import time
import traceback
from typing import Iterable
import torch
import numpy as np
import torchaudio
import yaml
from numpy import ndarray
from torch import Tensor

from ..tokenizer import S3Tokenizer
from ..generation.generation_ import S3Token2Wav

from ..generation.xvector import CAMPPlus
from ..modules import MEL_SPEC

def stream_bytes_to_s3_tokens(
    iterator : Iterable[bytes] , 
    config : dict 
) : 

    output_sources = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = S3Tokenizer(config = config['tokenizer']).to(device)
    tokenizer.eval()

    speaker_encoder : CAMPPlus = CAMPPlus().to(device)
    speaker_encoder.eval()

    mel_spec_fn : MEL_SPEC = MEL_SPEC(config['mel-spec']).to(device)
    mel_spec_fn.eval()

    model : S3Token2Wav = S3Token2Wav(device , config['generation']).to(device)
    model.eval()

    ref_wav_tensor : Tensor = torchaudio.load(config['reference-audio-path'])[0].to(device) # [1, Time]

    ref_mel : Tensor = mel_spec_fn(ref_wav_tensor)          # [1, 80, Frames]
    ref_mel = ref_mel.transpose(1, 2)                        # [1, Frames, 80]  ← flow expects [B, Frames, Mels]

    ref_x_vector : Tensor = speaker_encoder.inference(ref_wav_tensor)  # [1, X_Vector_Dim]
    ref_speech_tokens , _ = tokenizer(ref_wav_tensor)        # [1, Token_Len]

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
    audio_buffer = np.array([] , dtype = np.float32)

    for raw_bytes in iterator : 

        # * Convert to float32 normalized for the model
        raw_audio : ndarray = np.frombuffer(raw_bytes , dtype = np.int16).astype(np.float32) / 32768.0
        current_audio = np.concatenate([audio_buffer, raw_audio])

        torch_audio : Tensor = torch.from_numpy(current_audio).to(device)

        is_last = len(raw_audio) < config['chunk-size'] 

        with torch.no_grad() : 

            codes , codes_lens = tokenizer(torch_audio.unsqueeze(0)) # * Add batch dim

            if output_sources is not None and output_sources.shape[2] > codes_lens[0]:
                output_sources = None

            output_wavs , output_sources = model.inference(
                speech_tokens = codes[:, :codes_lens[0]] , 
                ref_dict = ref_dict , 
                cache_source = output_sources , 
                finalize = is_last
            )

        tokens = codes[0 , : codes_lens[0]].tolist()

        if tokens : 
            yield tokens

        # * Maintain context buffer (last 200ms) to prevent boundary artifacts
        audio_buffer = current_audio[-3200:] # 3200 samples = 200ms @ 16kHz

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

        config = config['tests']['s3gen-streaming']

    try : 

        with open(config['input-file-path'] , 'rb') as audio_file : 

            start_time : float = time.time()

            for token_batch in stream_bytes_to_s3_tokens(
                iter(
                    lambda : audio_file.read(config['chunk-size']) , b''
                ) , 
                config
            ) : 
                print(f"Emitted {len(token_batch)} tokens: {token_batch[:5]}...")

            print(f"Total streaming time: {time.time() - start_time:.2f} seconds")

    except Exception as e : 
        print(f"Error during streaming: {e} , {traceback.format_exc()}")