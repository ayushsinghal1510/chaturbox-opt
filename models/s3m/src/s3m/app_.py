import torch
import yaml
import torchaudio
from torch import Tensor
from torchaudio.transforms import Resample 

from .chatterbox.models.s3gen.generation_ import S3Token2Wav
from safetensors.torch import load_file

from s3t import S3Tokenizer
from services import MEL_SPEC

def generate_tts_audio(
    config : dict , 
    device : str
) -> None : 

    states = torch.load(
        config['voices']['_1'] , 
        map_location = device
    )
    ref_dict = states['gen']

    tokenizer_mel_spec_fn : MEL_SPEC = MEL_SPEC(
        config = config['tokenizer-mel-spec']
    )

    tokenizer : S3Tokenizer = S3Tokenizer(
        config = config['tokenizer'] , 
        mel_spec = tokenizer_mel_spec_fn ,
    )

    tokenizer.load_state_dict(
        torch.load(
            config['tokenizer']['state-dict-file'] , 
            map_location = device
        )
    )
    tokenizer.to(device).eval()

    s3gen : S3Token2Wav = S3Token2Wav(
        device = device , 
        config = config['generation']
    )
    s3gen.load_state_dict(
        load_file(config['generation']['state-dict-file']) , 
        strict = False
    )
    s3gen.to(device).eval()

    with torch.inference_mode() : 

        (
            content_audio_tensor , 
            content_audio_sr 
        ) = torchaudio.load(
            config['case']['content-audio-path'] , 
        )


        content_audio_tensor_16 : Tensor = Resample(
            orig_freq = content_audio_sr , 
            new_freq = 16_000
        ).to(device)(content_audio_tensor)

        s3_tokens, _ = tokenizer(content_audio_tensor_16)

        wav , _ = s3gen.inference(
            speech_tokens = s3_tokens , 
            ref_dict = ref_dict,
        )
        wav = wav.squeeze(0).detach().cpu()

    torchaudio.save(
        config['case']['output-audio-path'] , 
        wav , 
        sample_rate = config['case']['output-audio-sr']
    )

def main() : 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

    generate_tts_audio(
        config = config['generation'] , 
        device = device
    )