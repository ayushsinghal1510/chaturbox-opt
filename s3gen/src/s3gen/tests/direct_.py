import time
import traceback
import torch
import torchaudio
from torchaudio.transforms import Resample
import yaml
from torch import Tensor


from .services_ import load_all_client , get_ref_dict

def direct_bytes_to_s3_tokens(
    config : dict 
) -> Tensor : 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    (
        tokenizer , 
        speaker_encoder , 
        mel_spec_fn_model , 
        model 
    ) = load_all_client(device , config)

    ref_dict = get_ref_dict(
        audio_file_path = config['reference-audio-path'] , 
        device = device , 
        mel_spec_fn_model = mel_spec_fn_model , 
        speaker_encoder = speaker_encoder , 
        tokenizer = tokenizer
    )

    # states = torch.load(
    #     '/teamspace/studios/this_studio/models/conds.pt' , 
    #     map_location = 'cpu'
    # )

    # ref_dict = states['gen']

    content_audio , content_sr = torchaudio.load(config['input-file-path'])
    content_audio_16 = Resample(content_sr , 16_000)(content_audio)

    with torch.no_grad() : 

        tokens , tokens_length = tokenizer(content_audio_16) # * Add batch dim

        output_wavs , output_sources = model.inference(
            speech_tokens = tokens , 
            # speech_tokens = codes , 
            ref_dict = ref_dict , 
            cache_source = None , 
            # cache_source = None , 
            finalize = True
        )

        return output_wavs.detach().cpu()

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

        config = config['tests']['s3gen-streaming']

    try : 

        start_time : float = time.time()

        audio : Tensor = direct_bytes_to_s3_tokens(config)

        output_filename = "streamed_output.wav"
        torchaudio.save(output_filename , audio , 24_000)
        
        print(f"--- Streaming Complete ---")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Saved audio to: {output_filename}")

    except Exception as e : 
        print(f"Error during streaming: {e} , {traceback.format_exc()}")