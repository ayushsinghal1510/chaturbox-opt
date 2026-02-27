# import time
from torchaudio.transforms import Resample
import yaml
# import torch
# import librosa

# from .tokenizer import S3Tokenizer

# def run_inference(config : dict) : 

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     tokenizer = S3Tokenizer(config = config).to(device)

#     tokenizer.eval()

#     speech , _ = librosa.load(
#         config['file-name'] , 
#         sr = config['sample-rate']
#     )
    
#     # Convert to tensor and add batch dimension [1, Time]
#     speech_tensor = torch.from_numpy(speech).unsqueeze(0).to(device)


#     with torch.no_grad() : 
#         tokens , lengths = tokenizer(speech_tensor)

#     # 4. Process Output
#     # tokens shape: [Batch, Seq_Len]
#     token_list = tokens[0][:lengths[0]].tolist()
    
#     return token_list

# def main() : 

#     with open('config.yml') as config_file : 
#         config : dict = yaml.safe_load(config_file)

#     try : 

#         start_time : float = time.time()

#         compressed_tokens = run_inference(config)

#         print(time.time() - start_time)

#         print(compressed_tokens)

#     except Exception as e : 
#         print(f"Error during inference: {e}")

import yaml
import torch
import torchaudio

from .generation import S3Token2Wav
from .tokenizer import S3Tokenizer
from .modules import MEL_SPEC, mel_

def clone_voice(config : dict) : 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = S3Tokenizer(config = config['tokenizer']).to(device)
    mel_spec : MEL_SPEC = MEL_SPEC(config = config['mel-spec']).to(device)

    model = S3Token2Wav(
        tokenizer = tokenizer , 
        device = device , 
        config = config , 
        mel_spec = mel_spec
    ).to(device)


    model.eval()

    ref_tensor , ref_sr = torchaudio.load(config['tests']['ref']['file-name'])
    ref_tensor = Resample(
        orig_freq = ref_sr , 
        new_freq = config['tests']['ref']['sample-rate']
    )(ref_tensor)

    content_tensor , content_sr = torchaudio.load(config['tests']['content']['file-name'])
    content_tensor = Resample(
        orig_freq = content_sr , 
        new_freq = config['tests']['content']['sample-rate']
    )(content_tensor)

    with torch.no_grad() : 

        speech_tokens , _ = tokenizer(content_tensor)
        
        output_wav , _ = model.inference(
            speech_tokens = speech_tokens , 
            ref_wav = ref_tensor , 
            ref_sr = 16_000 , 
            finalize = True
        )

    output_np = output_wav.squeeze().cpu().numpy()
    torchaudio.save(
        uri = config['tests']['output']['file-name'] , 
        src = output_wav , 
        sample_rate = config['tests']['output']['sample-rate']
    )

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

    clone_voice(config)