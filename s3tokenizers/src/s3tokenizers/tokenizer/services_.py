import torch

import torch.nn.functional as F

from numpy import ndarray 
from torch import Tensor

from torch.types import Device 

class SERVICES : 

    def __init__(self) -> None : 
        pass

    def _prepare_audio(self , wavs : ndarray | Tensor) -> list[Tensor] : 

        '''Prepare a list of audios for s3tokenizer processing.'''

        processed_wavs : list[Tensor] = []

        for wav in wavs : 

            # * if wav is ndarray, convert to torch
            wav_tensor : Tensor = torch.as_tensor(wav)

            # * if wav is flat, make it 2d
            if wav_tensor.dim() == 1 : 
                wav_tensor = wav_tensor.unsqueeze(0)

            processed_wavs.append(wav_tensor)

        return processed_wavs

    def log_mel_spectrogram(
        self , 
        audio : Tensor | ndarray , 
        device : Device ,
        n_fft : int , 
        window : Tensor , 
        _mel_filters : Tensor , 
        hop_length : int , 
        padding : int , 
    ) -> Tensor :

        # * Convert to Tensor
        audio_tensor : Tensor = torch.as_tensor(audio)

        audio_tensor = audio_tensor.to(device)

        if padding > 0 : 
            audio_tensor = F.pad(audio_tensor , (0 , padding))

        # * Perform the STFT (Short Term Fourier Transform, basically fourier transform with sliding window and overlapping)
        stft_tensor = torch.stft(
            input = audio_tensor , 
            n_fft = n_fft , # * Length of the window
            hop_length = hop_length , # * Slide of the window
            window = window , # * Window function / strategy
            return_complex = True
        )

        # even_bins : Tensor = stft_tensor[... , : -1] # * Leave the last bin, as it can be odd
        magnitudes : Tensor = stft_tensor.abs()**2 # * Get the real part and square it 

        mel_spec : Tensor = _mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec , min = 1e-10).log10()
        log_spec = torch.maximum(log_spec , log_spec.max() - 8.0)

        log_spec = (log_spec + 4.0) / 4.0

        return log_spec