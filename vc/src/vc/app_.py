import torch
import numpy as np
import librosa
import torchaudio

from .chatterbox.vc import ChatterboxVC
from .chatterbox.models.s3gen import S3Gen
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# model : ChatterboxMultilingualTTS = ChatterboxMultilingualTTS.from_pretrained(device)
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
    # wav = vc_mode.generate(
    #     audio = '/teamspace/studios/this_studio/ElevenLabs_2025-06-05T07_18_58_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3' , 
    # )


    # chosen_prompt = '/teamspace/studios/this_studio/en_f1.flac'

    # generate_kwargs = {
    #     "exaggeration": exaggeration_input,
    #     "temperature": temperature_input,
    #     "cfg_weight": cfgw_input,
    # }
    # if chosen_prompt:
    #     generate_kwargs["audio_prompt_path"] = chosen_prompt
    #     print(f"Using audio prompt: {chosen_prompt}")
    # else:
    #     print("No audio prompt provided; using default voice.")
        
    # wav = model.generate(
    #     text_input[:300],  # Truncate text to max chars
    #     language_id=language_id,
    #     **generate_kwargs
    # )

    # print(wav)
    # print("Audio generation complete.")
    # return (model.sr, wav.squeeze(0).numpy())


def main() : 
    generate_tts_audio(
)