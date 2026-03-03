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

    working_sd = s3gen.tokenizer.state_dict()

    tokenizer.load_state_dict(working_sd, strict=False)
    torch.save(tokenizer.state_dict(), "s3_tokenizer_modular_fixed.pt")

    compare_tokenizers(s3gen.tokenizer , tokenizer)

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

import torch

def compare_tokenizers(tokenizer_original, tokenizer_modular):
    """
    Compares the weights of two tokenizer instances to ensure 
    the modular version loaded pretrained weights correctly.
    """
    print("--- Starting Weight Comparison ---")
    
    orig_sd = tokenizer_original.state_dict()
    mod_sd = tokenizer_modular.state_dict()

    # 1. Check if they have the same number of parameter entries
    if len(orig_sd) != len(mod_sd):
        print(f"CRITICAL: Key count mismatch! Original: {len(orig_sd)}, Modular: {len(mod_sd)}")
    
    mismatched_keys = []
    missing_keys = []
    
    for key in orig_sd.keys():
        if key not in mod_sd:
            missing_keys.append(key)
            continue
            
        # 2. Check if the actual weights are identical
        if not torch.equal(orig_sd[key], mod_sd[key]):
            # Check for small epsilon difference if strictly equal fails
            diff = (orig_sd[key] - mod_sd[key]).abs().max()
            if diff > 1e-6:
                mismatched_keys.append((key, diff.item()))

    # Reporting results
    if not missing_keys and not mismatched_keys:
        print("SUCCESS: All weights are identical across both models.")
    else:
        if missing_keys:
            print(f"MISSING KEYS in Modular version: {missing_keys}")
        if mismatched_keys:
            print("WEIGHT MISMATCH detected in following keys:")
            for key, diff in mismatched_keys:
                print(f" - {key} (Max Diff: {diff})")
    
    print("--- Comparison Finished ---")

# Usage:
# compare_tokenizers(script1_tokenizer, script2_tokenizer)


def main() : 
    generate_tts_audio(
)