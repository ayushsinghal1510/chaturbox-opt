import torch
import librosa
import torchaudio

from .chatterbox.vc import ChatterboxVC
from .chatterbox.models.s3gen import S3Gen
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

vc_mode : ChatterboxVC = ChatterboxVC.from_pretrained(device)

def generate_tts_audio(
) -> None : 

    states = torch.load(
        '/teamspace/studios/this_studio/models/conds.pt' , 
        map_location = 'cpu'
    )

    ref_dict = states['gen']

    s3gen = S3Gen()
    s3gen.load_state_dict(
        load_file('/teamspace/studios/this_studio/models/s3gen.safetensors') , 
        strict = False
    )
    s3gen.to(device).eval()

    with torch.inference_mode():
        audio_16, _ = librosa.load('/teamspace/studios/this_studio/ElevenLabs_2025-06-05T07_18_58_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3', sr=16_000)
        audio_16 = torch.from_numpy(audio_16).float().to(device)[None, ]

        s3_tokens, _ = s3gen.tokenizer(audio_16)
        wav, _ = s3gen.inference(
            speech_tokens=s3_tokens,
            ref_dict=ref_dict,
        )
        wav = wav.squeeze(0).detach().cpu()

    torchaudio.save("output.wav", wav, sample_rate=vc_mode.sr)


def main() : 
    generate_tts_audio(
)