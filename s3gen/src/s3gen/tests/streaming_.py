import time
import traceback
from typing import Generator, Iterable
import torch
import numpy as np
import torchaudio
import yaml
from numpy import ndarray
from torch import Tensor

from modules import make_line_plot
# from modules.src.modules import make_line_plot

from .services_ import load_all_client , get_ref_dict

def stream_bytes_to_s3_tokens(
    iterator : Iterable[bytes] , 
    config : dict 
) -> Generator[Tensor , None , None] : 

    output_sources = None

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

    audio_buffer = np.array([] , dtype = np.float32)

    for raw_bytes in iterator : 

        # * Convert to float32 normalized for the model
        raw_audio : ndarray = np.frombuffer(raw_bytes , dtype = np.int16).astype(np.float32) / 32768.0
        current_audio = np.concatenate([audio_buffer, raw_audio])

        torch_audio : Tensor = torch.from_numpy(current_audio).to(device)

        is_last = len(raw_audio) * 2 < config['chunk-size']

        with torch.no_grad() : 

            tokens , tokens_length = tokenizer(torch_audio.unsqueeze(0)) # * Add batch dim

            output_wavs , output_sources = model.inference(
                speech_tokens = tokens[: , :tokens_length[0]] , 
                # speech_tokens = codes , 
                ref_dict = ref_dict , 
                cache_source = output_sources , 
                # cache_source = None , 
                finalize = is_last
            )

            yield output_wavs.cpu()

        # * Maintain context buffer (last 200ms) to prevent boundary artifacts
        audio_buffer = current_audio[-3200:] # 3200 samples = 200ms @ 16kHz

def main() : 

    with open('config.yml') as config_file : 
        config : dict = yaml.safe_load(config_file)

        config = config['tests']['s3gen-streaming']

    all_chunks : list[Tensor] = []

    try : 

        with open(config['input-file-path'] , 'rb') as audio_file : 

            start_time : float = time.time()

            for audio  in stream_bytes_to_s3_tokens(
                iter(
                    lambda : audio_file.read(config['chunk-size']) , b''
                ) , 
                config
            ) : 
                print('Streaming')
                all_chunks.append(audio)

            print('Streaming Complete. Concatenating chunks and saving output...')

            if all_chunks : 

                full_audio = torch.cat(all_chunks , dim = -1)

                if full_audio.ndim == 1 : 
                    full_audio = full_audio.unsqueeze(0)

                elif full_audio.ndim == 3 : 
                    full_audio = full_audio.squeeze(0)

                print('Saving the figure')

                make_line_plot(
                    data = full_audio[0].cpu() , 
                    title = 'Generated Audio Waveform' , 
                    xlabel = 'Samples' , 
                    ylabel = 'Amplitude' , 
                    filename = 'final_audio.png' , 
                    save = True
                )

                # make_heatmap(
                #     data = full_mel[0].cpu() , 
                #     title = 'Generated Mel Spectrogram' , 
                #     xlabel = 'Frames' , 
                #     ylabel = 'Mel Bins' , 
                #     filename = 'final_mel.png' , 
                #     save = True
                # )

                output_filename = "streamed_output.wav"
                torchaudio.save(output_filename , full_audio.cpu() , 24_000)
                
                print(f"--- Streaming Complete ---")
                print(f"Total time: {time.time() - start_time:.2f}s")
                print(f"Saved audio to: {output_filename}")
            else:
                print("No audio chunks were generated.")

            print(f"Total streaming time: {time.time() - start_time:.2f} seconds")

    except Exception as e : 
        print(f"Error during streaming: {e} , {traceback.format_exc()}")