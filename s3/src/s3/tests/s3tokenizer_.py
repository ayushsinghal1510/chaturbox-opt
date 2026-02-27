import time
import traceback
from torchaudio.transforms import Resample
import yaml
import torch
import librosa

from ..tokenizer import S3Tokenizer

def run_inference(config : dict) : 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = S3Tokenizer(config = config['tokenizer']).to(device)

    tokenizer.eval()

    speech , _ = librosa.load(
        config['file-name'] , 
        sr = config['sample-rate']
    )
    
    # Convert to tensor and add batch dimension [1, Time]
    speech_tensor = torch.from_numpy(speech).unsqueeze(0).to(device)


    with torch.no_grad() : 
        tokens , lengths = tokenizer(speech_tensor)

    # 4. Process Output
    # tokens shape: [Batch, Seq_Len]
    token_list = tokens[0][:lengths[0]].tolist()
    
    return token_list

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)['tests']['s3tokenizer-streaming']

    try : 

        start_time : float = time.time()

        compressed_tokens = run_inference(config)

        print(time.time() - start_time)

        print(compressed_tokens)

    except Exception as e : 
        print(f"Error during inference: {e} : {traceback.format_exc()}")
