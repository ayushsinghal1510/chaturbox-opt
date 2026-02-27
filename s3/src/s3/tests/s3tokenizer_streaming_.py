import subprocess
import time
from typing import Iterable
import torch
import numpy as np
import yaml
from numpy import ndarray
from torch import Tensor

from ..tokenizer import S3Tokenizer

def stream_bytes_to_s3_tokens(
    iterator : Iterable[bytes] , 
    config : dict 
) : 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = S3Tokenizer(config = config['tokenizer']).to(device)

    tokenizer.eval()

    audio_buffer = np.array([] , dtype = np.float32)

    for raw_bytes in iterator : 

        # * Convert to float32 normalized for the model
        raw_audio : ndarray = np.frombuffer(raw_bytes , dtype = np.int16).astype(np.float32) / 32768.0
        current_audio = np.concatenate([audio_buffer, raw_audio])

        torch_audio : Tensor = torch.from_numpy(current_audio).to(device)

        with torch.no_grad() : 
            codes , codes_lens = tokenizer(torch_audio.unsqueeze(0)) # * Add batch dim

        tokens = codes[0 , : codes_lens[0]].tolist()

        if tokens : 
            yield tokens

        # * Maintain context buffer (last 200ms) to prevent boundary artifacts
        audio_buffer = current_audio[-3200:] # 3200 samples = 200ms @ 16kHz

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

        config = config['tests']['s3tokenizer-streaming']

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
        print(f"Error during streaming: {e}")