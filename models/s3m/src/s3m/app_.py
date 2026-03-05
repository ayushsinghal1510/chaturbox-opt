import torch
import librosa
import torchaudio

from .chatterbox.vc import ChatterboxVC
from .chatterbox.models.s3gen import S3Gen
from safetensors.torch import load_file

from s3t import S3Tokenizer
from services import MEL_SPEC , make_line_plot
# from services.src.services import MEL_SPEC

device = "cuda" if torch.cuda.is_available() else "cpu"

vc_mode : ChatterboxVC = ChatterboxVC.from_pretrained(device)

def generate_tts_audio() -> None : 

    states = torch.load(
        '/teamspace/studios/this_studio/models/conds.pt' , 
        map_location = 'cpu'
    )
    ref_dict = states['gen']

    tokenizer_mel_spec_fn : MEL_SPEC = MEL_SPEC(
        config = {
            'sample-rate' : 16_000 , 
            'n-fft' : 400 , 
            'n-mels' : 128 , 
            'win-size' : 400 , 
            'hop-size' : 160 , 
            'win-length' : 400 , 
            'center' : True , 
            'normalized' : False , 
            'spec-calc' : 'power' , 
            'drop-last-spec' : True ,
            'log' : 'log10' , 
            'clip-val' : 0.0000000001 , 
            'addition-amount' : 4 , 
            'division-amount' : 4
        }
    )

    tokenizer : S3Tokenizer = S3Tokenizer(
        config = {
            'model-name' : 'speech_tokenizer_v2_25hz' , 
            'sample-rate' : 16_000 , 
            'padding' : 0
        } , 
        mel_spec = tokenizer_mel_spec_fn ,
    )

    s3gen = S3Gen()
    s3gen.load_state_dict(
        load_file('/teamspace/studios/this_studio/models/s3gen.safetensors') , 
        strict = False
    )
    s3gen.to(device).eval()

    # * Load the tokenizer with the saved the state dict 

    tokenizer.load_state_dict(
        torch.load(
            'assets/tokenizers/s3/tokenizer.pt' , 
            map_location = 'cpu'
        )
    )


    with torch.inference_mode():
        audio_16, _ = librosa.load('/teamspace/studios/this_studio/ElevenLabs_2025-06-05T07_18_58_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3', sr=16_000)
        audio_16 = torch.from_numpy(audio_16).float().to(device)[None, ]

        s3_tokens, _ = tokenizer(audio_16)
        # print(s3_tokens.numpy().tolist())

        make_line_plot(
            data = s3_tokens.squeeze(0).cpu().numpy() , 
            title = 'S3 Tokens' , 
            xlabel = 'Time (tokens)' , 
            ylabel = 'Token Value' , 
            filename = 's3_tokens.png' , 
            figsize = (15 , 5) , 
            dpi = 400 , 
            save = True
        )
        wav, _ = s3gen.inference(
            speech_tokens=s3_tokens,
            ref_dict=ref_dict,
        )
        wav = wav.squeeze(0).detach().cpu()

    torchaudio.save("output.wav", wav, sample_rate=vc_mode.sr)


def main() : 
    generate_tts_audio(
)