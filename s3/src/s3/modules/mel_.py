import torch

import torch.nn as nn
import torchaudio.functional as F

from torch import Tensor

class MEL_SPEC(nn.Module) : 

    n_fft : Tensor
    num_mels : Tensor
    sample_rate : Tensor
    hop_size : Tensor
    win_size : Tensor
    fmin : Tensor
    fmax : Tensor
    center : Tensor
    C : Tensor
    clip_val : Tensor
    mel_basis : Tensor
    hann_window : Tensor

    def __init__(self , config : dict) -> None : 

        super().__init__()

        # * Creating values and scaling them to tensors for easy device management. These will be registered as buffers so they move with the model.
        self.register_buffer('n_fft' , torch.tensor(config['n-fft'] , dtype = torch.long))
        self.register_buffer('num_mels' , torch.tensor(config['num-mels'] , dtype = torch.long))
        self.register_buffer('sample_rate' , torch.tensor(config['sample-rate'] , dtype = torch.long))
        self.register_buffer('hop_size' , torch.tensor(config['hop-size'] , dtype = torch.long))
        self.register_buffer('win_size' , torch.tensor(config['win-size'] , dtype = torch.long))
        self.register_buffer('fmin' , torch.tensor(config['f-min'] , dtype = torch.float))
        self.register_buffer('fmax' , torch.tensor(config['f-max'] , dtype = torch.float))
        self.register_buffer('center' , torch.tensor(config['center'] , dtype = torch.bool))
        self.register_buffer('C' , torch.tensor(config.get('c' , 1) , dtype = torch.float))
        self.register_buffer('clip_val' , torch.tensor(config.get('clip-val' , 1e-5), dtype = torch.float))
        
        self.pad_mode : str = config.get('pad-mode' , 'reflect')

        # * Pre-calculate Mel Filterbank
        mel_fb = F.melscale_fbanks(
            n_freqs = (self.n_fft.item() // 2) + 1 , 
            f_min = self.fmin.item() , 
            f_max = self.fmax.item() , 
            n_mels = self.num_mels.item() , 
            sample_rate = self.sample_rate.item() , 
            norm = 'slaney' , 
            mel_scale = 'htk'
        ).transpose(0, 1) 

        # * Pre-calculate hann Window
        window = torch.hann_window(self.win_size.item())

        self.register_buffer('mel_basis' , mel_fb)
        self.register_buffer('hann_window' , window)

    def forward(self , audio : Tensor) -> Tensor : 

        # * Range Validation
        if torch.max(torch.abs(audio)) > 1.0 : 
            print("Audio magnitude exceeds 1.0. Results may differ from training.") # ! Change with logger

        # * Manual Padding (Center=False logic)
        pad_size = (self.n_fft.item() - self.hop_size.item()) // 2

        audio = torch.nn.functional.pad(
            audio.unsqueeze(1) , 
            (pad_size , pad_size) ,  
            mode = self.pad_mode
        ).squeeze(1)

        # * STFT
        spec = torch.stft(
            audio , 
            n_fft = self.n_fft.item() , 
            hop_length = self.hop_size.item() , 
            win_length = self.win_size.item() , 
            window = self.hann_window , 
            center = self.center.item() , 
            pad_mode = self.pad_mode , 
            normalized = False , 
            onesided = True , 
            return_complex = True
        )

        # * Magnitude Calculation
        magnitudes = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

        # * Mel Projection
        mel_output = self.mel_basis @ magnitudes

        # * Log Compression
        output : Tensor = torch.log(
            torch.clamp(
                mel_output , 
                min = self.clip_val.item()
            ) * self.C.item()
        )

        return output