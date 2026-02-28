from dataclasses import dataclass
import os
from pathlib import Path 
import yaml
import torch
import torchaudio

from torch import Tensor

from .tokenizer_ import MTLTokenizer

from .models import VoiceEncoder

from s3gen import S3Token2Wav
from s3tokenizers import S3Tokenizer

from s3gen import CAMPPlus
from modules import MEL_SPEC

from .models_.t3.modules.cond_enc import T3Cond
from .models_.t3.t3 import T3

import torch.nn.functional as F

from safetensors.torch import load_file as load_safetensors

@dataclass
class Conditionals : 

    t3 : T3Cond
    gen : dict

    def to(self , device) : 

        self.t3 = self.t3.to(device = device)

        for k , v in self.gen.items() : 

            if torch.is_tensor(v) : 
                self.gen[k] = v.to(device = device)

        return self

    def save(self , fpath : Path) -> None : 

        arg_dict = dict(
            t3 = self.t3.__dict__ , 
            gen = self.gen
        )

        torch.save(arg_dict , fpath)

    @classmethod
    def load(cls , fpath , map_location = "cpu") : 

        kwargs = torch.load(
            fpath , 
            map_location = map_location , 
            weights_only = True
        )

        return cls(T3Cond(**kwargs['t3']) , kwargs['gen'])

LLAMA_520M_CONFIG_DICT = dict(
    # Arbitrary small number that won't cause problems when loading.
    # These param are unused due to custom input layers.
    vocab_size=8,
    # default params needed for loading most pretrained 1B weights
    max_position_embeddings=131072,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="eager",
    head_dim=64,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    num_key_value_heads=16,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3"
    ),
    rope_theta=500000.0,
    torch_dtype="bfloat16",
    use_cache=True,
)

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
}



class T3Config:
    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)


def main() : 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('config.yaml') as config_file : 
        config : dict = yaml.safe_load(config_file)

    config = config['tests']['t3-generation']

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(config['t3']['model-path'])
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()


    tokenizer : MTLTokenizer = MTLTokenizer(config['tokenizer'])

    ve : VoiceEncoder = VoiceEncoder()
    
    ve.load_state_dict(
        torch.load(
            config['voice-encoder']['model-path'] , 
            weights_only = True , 
            map_location = torch.device('cpu')
        )
    )

    ve.eval()

    speaker_encoder : CAMPPlus = CAMPPlus().to(device)
    speaker_encoder.eval()

    mel_spec_fn : MEL_SPEC = MEL_SPEC(config['mel-spec']).to(device)
    mel_spec_fn.eval()

    audio_tokenizer = S3Tokenizer(config = config['audio_tokenizer']).to(device)
    audio_tokenizer.eval()

    ref_wav_tensor : Tensor = torchaudio.load(config['reference-audio-path'])[0]

    ref_mel : Tensor = mel_spec_fn(ref_wav_tensor)          # [1, 80, Frames]
    ref_mel = ref_mel.transpose(1, 2)                        # [1, Frames, 80]  ← flow expects [B, Frames, Mels]

    ref_x_vector : Tensor = speaker_encoder.inference(ref_wav_tensor)  # [1, X_Vector_Dim]
    ref_speech_tokens , _ = audio_tokenizer(ref_wav_tensor)        # [1, Token_Len]

    # prompt_feat is [B, Frames, Mels] so shape[1] is frames
    # we need prompt frames < minimum expected h.shape[1]
    # keep prompt short: 40 frames = 20 tokens (token_mel_ratio=2)
    TOKEN_LIMIT = 20
    PROMPT_FRAMES = TOKEN_LIMIT * 2  # = 40 frames

    ref_mel = ref_mel[:, :PROMPT_FRAMES, :]              # [1, 40, 80]
    ref_speech_tokens = ref_speech_tokens[:, :TOKEN_LIMIT]  # [1, 20]

    ref_speech_token_lens = torch.tensor([ref_speech_tokens.shape[1]], device=device)

    ref_dict : dict = {
        'prompt_token'     : ref_speech_tokens.to(device),
        'prompt_token_len' : ref_speech_token_lens.to(device),
        'prompt_feat'      : ref_mel.to(device),          # [1, 40, 80]
        'prompt_feat_len'  : None,
        'embedding'        : ref_x_vector.to(device)
    }

    text_tokens : Tensor = tokenizer.text_to_tokens(
        text = 'Hello, how are you doing today?' , 
        language_id = 'en'
    )

    ve_embed : Tensor = torch.from_numpy(
        ve.embeds_from_wavs(
            [ref_wav_tensor.squeeze(0).numpy()] , 
            sample_rate =16000
        )
    )

    ve_embed = ve_embed.mean(axis = 0 , keepdims = True).to(device)

    t3_cond_prompt_tokens , _ = audio_tokenizer(ref_wav_tensor[: 6 * 16_000] , max_len = 150)
    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(device)

    t3_cond = T3Cond(
        speaker_emb = ve_embed , 
        cond_prompt_speech_tokens = t3_cond_prompt_tokens , 
        emotion_adv = config['attributes']['exaggeration'] * torch.ones(1 , 1 , 1) , 
    ).to(device = device)

    conds = Conditionals(t3_cond , ref_dict)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

    sot = 255
    eot = 0
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    model : S3Token2Wav = S3Token2Wav(device , config['generation']).to(device)
    model.eval()

    with torch.inference_mode():
        speech_tokens = t3.inference(
            t3_cond=conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,  # TODO: use the value in config
            temperature=0.8,
            cfg_weight=0.5,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
        # Extract only the conditional batch.
        speech_tokens = speech_tokens[0]

        # # TODO: output becomes 1D
        # speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens.to(device)

        # print(speech_tokens.shape)

        output_wavs , output_sources = model.inference(
            speech_tokens = speech_tokens.unsqueeze(0) , 
            ref_dict = ref_dict , 
            # cache_source = output_sources , 
            # finalize = is_last
        )

        output_wavs = output_wavs.squeeze(0).detach().cpu()

        print(output_wavs)

        torchaudio.save('file.wav' , output_wavs , sample_rate = 16000)

    print(text_tokens)