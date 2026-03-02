import torch
import torchaudio
from torchaudio.transforms import Resample

from torch import Tensor

from torchtyping import TensorType

from s3tokenizers import S3Tokenizer

from safetensors.torch import load_file
# from s3tokenizers.src.s3tokenizers import S3Tokenizer
from ..generation.generation_ import S3Token2Wav

from ..generation.xvector import CAMPPlus
from modules import MEL_SPEC
# from modules.src.modules import MEL_SPEC

def tokenize(
    audio_tensor : Tensor , 
    tokenizer : S3Tokenizer , 
    device : str
) -> tuple[Tensor , Tensor] : 

    audio_tensor = audio_tensor.to(device)

    with torch.no_grad() : 
        tokens , token_lens = tokenizer(audio_tensor)

    return tokens.cpu() , token_lens.cpu()

def encode_speaker(
    audio_tensor : Tensor , 
    speaker_encoder : CAMPPlus , 
    device : str
) -> Tensor : 

    audio_tensor = audio_tensor.to(device)

    with torch.no_grad() : 
        x_vector = speaker_encoder.inference(audio_tensor)

    return x_vector.cpu()

def load_all_client(
    device : str , 
    config : dict
) -> tuple[
    S3Tokenizer , 
    CAMPPlus , 
    MEL_SPEC , 
    S3Token2Wav
] : 

    mel_spec_fn_tokenizer : MEL_SPEC = MEL_SPEC(config['mel-spec-tokenizer'])
    mel_spec_fn_tokenizer.eval()

    tokenizer = S3Tokenizer(
        config = config['tokenizer'] , 
        mel_spec = mel_spec_fn_tokenizer
    ).to(device)
    tokenizer.eval()

    speaker_encoder : CAMPPlus = CAMPPlus().to(device)
    speaker_encoder.eval()

    mel_spec_fn_model : MEL_SPEC = MEL_SPEC(config['mel-spec-model']).to(device)
    mel_spec_fn_model.eval()

    model : S3Token2Wav = S3Token2Wav(device , config['generation']).to(device)
    model.load_state_dict(
        load_file('/teamspace/studios/this_studio/models/s3gen.safetensors') , 
        strict = False
    )

    return (
        tokenizer , 
        speaker_encoder , 
        mel_spec_fn_model , 
        model
    )

def get_ref_dict(
    audio_file_path : str , 
    device : str , 
    mel_spec_fn_model : MEL_SPEC , 
    speaker_encoder : CAMPPlus , 
    tokenizer : S3Tokenizer
) : 

    ref_wav_tensor , ref_sr = torchaudio.load(audio_file_path)
    # * ref_wav_tensor [channels , audio]

    max_samples = 10 * ref_sr
    if ref_wav_tensor.shape[1] > max_samples:
        ref_wav_tensor = ref_wav_tensor[:, :max_samples]

    resampler = Resample(orig_freq = ref_sr , new_freq = 16_000)
    ref_wav_tensor_16 : TensorType['channels' , 'audio'] = resampler(ref_wav_tensor)

    resampler = Resample(orig_freq = ref_sr , new_freq = 24_000)
    ref_wav_tensor_24 : TensorType['channels' , 'audio'] = resampler(ref_wav_tensor)

    # * Mel Spec gives us [channels , bins , frames], we transpose it to[channels , frames , bins] for the model
    ref_mel_24 : Tensor = mel_spec_fn_model(ref_wav_tensor_24).transpose(1 , 2)

    ref_x_vector : Tensor = encode_speaker(ref_wav_tensor_16 , speaker_encoder , device)
    ref_speech_tokens , _ = tokenize(ref_wav_tensor_16 , tokenizer , device)

    # * 2. Ensure mel_len = 2 * token_len
    # Instead of hardcoding to 20 tokens, we use the full length but ensure exact 2:1 alignment
    target_token_len = ref_mel_24.shape[1] // 2

    # ! Can be an issue
    
    ref_speech_tokens = ref_speech_tokens[:, :target_token_len]
    ref_mel = ref_mel_24[:, :target_token_len * 2, :]

    ref_speech_token_lens = torch.tensor([ref_speech_tokens.shape[1]] , device = device)

    ref_dict : dict = {
        'prompt_token'     : ref_speech_tokens.to(device),
        'prompt_token_len' : ref_speech_token_lens.to(device),
        'prompt_feat'      : ref_mel.to(device),
        'prompt_feat_len'  : None,
        'embedding'        : ref_x_vector.to(device)
    }

    return ref_dict