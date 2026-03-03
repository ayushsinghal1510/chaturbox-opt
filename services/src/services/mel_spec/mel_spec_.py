import torch
import librosa 

import torch.nn as nn
import torchaudio.functional as F

from torch import Tensor

class MEL_SPEC(nn.Module) : 

    n_fft : Tensor
    hop_size : Tensor
    mel_basis : Tensor
    hann_window : Tensor
    clip_value : Tensor

    def __init__(
        self , 
        config : dict , 
        mel_basis : Tensor | None = None , 
        hann_window : Tensor | None = None
    ) -> None : 

        super().__init__()


        self.n_fft_val : int = config['n-fft']
        self.hop_size_val : int = config['hop-size']
        self.win_size : int | None = config.get('win-length')
        self.center : bool = config.get('center' , True)
        self.pad_mode : str = config.get('pad-mode' , 'reflect')
        self.clip_value_val : float = config.get('clip-val' , 1e-5)
        self.C : float = config.get('c' , 1.0)
        self.normalized : bool = config.get('normalized' , False)
        self.onsided : float | None = config.get('onesided' , None)
        self.spec_calc : str = config.get('spec-calc' , 'power')
        self.log : str = config.get('log' , 'log10')
        self.drop_last_spec : bool = config.get('drop-last-spec' , False)

        self.addition_amount : float = config.get('addition-amount' , 0.0)
        self.division_amount : float = config.get('division-amount' , 1.0)

        # * Creating values and scaling them to tensors for easy device management. These will be registered as buffers so they move with the model.
        self.register_buffer('n_fft' , torch.tensor(self.n_fft_val , dtype = torch.long))
        self.register_buffer('hop_size' , torch.tensor(self.hop_size_val , dtype = torch.long))
        self.register_buffer('clip_value' , torch.tensor(self.clip_value_val , dtype = torch.float))

        # * Check if mel basis is provided, if not calculate it from config parameters.
        if mel_basis is None : 

            if 'n-mels' not in config : 
                raise ValueError("Missing 'n-mels' in config for MEL_SPEC initialization.")
            if 'sample-rate' not in config : 
                raise ValueError("Missing 'sample-rate' in config for MEL_SPEC initialization.")

            # # * Calculate Mel Filterbank
            # mel_basis = F.melscale_fbanks(
            #     n_freqs = (self.n_fft_val // 2) + 1 , 
            #     f_min =  , 
            #     f_max =  , 
            #     n_mels = config['n-mels'] , 
            #     sample_rate =  , 
            #     norm = config.get('norm' , 'slaney') ,
            #     mel_scale = config.get('mel-scale' , 'htk')
            # ).transpose(0 , 1) 

            mel_basis_np = librosa.filters.mel(
                sr=config['sample-rate'], 
                n_fft=self.n_fft_val, 
                n_mels=config['n-mels'], 
                fmin=config.get('f-min' , 0.0), 
                fmax=config.get('f-max' , config['sample-rate'] // 2), 
                # htk=True, 
                # norm='slaney'
            )

            mel_basis = torch.from_numpy(mel_basis_np).float()

        if hann_window is None : 

            if self.win_size is None : 
                raise ValueError("win-length must be specified in config if hann_window is not provided.")

            window = torch.hann_window(self.win_size)

        self.register_buffer('hann_window' , window)
        self.register_buffer('mel_basis' , mel_basis)

    def forward(self , audio : Tensor) -> Tensor : 

        if not self.center and self.pad_mode == 'reflect' : 

            pad_amount = int((self.n_fft_val - self.hop_size_val) / 2)

            audio = torch.nn.functional.pad(
                audio.unsqueeze(1) , 
                (pad_amount , pad_amount) ,  
                mode = 'reflect'
            ).squeeze(1)

        # * STFT
        spec = torch.stft(
            audio , 
            n_fft = self.n_fft_val , 
            hop_length = self.hop_size_val , 
            win_length = self.win_size , 
            window = self.hann_window , 
            center = self.center , 
            pad_mode = self.pad_mode , 
            normalized = self.normalized , 
            onesided = None , 
            return_complex = True
        )

        if self.drop_last_spec : 
            spec = spec[... , : -1]

        if self.spec_calc == 'power' : 
            spec = torch.abs(spec)**2

        elif self.spec_calc == 'mod' : 

            spec = torch.view_as_real(spec) 
            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9)) 

        # * Mel Projection
        mel_output : Tensor = self.mel_basis @ spec

        clamped_output = torch.clamp(
            mel_output , 
            min = self.clip_value
        )

        if self.log == 'log10' : 

            clamped_output = clamped_output.log10() * self.C
            clamped_output = torch.maximum(clamped_output , clamped_output.max() - 8.0)

        elif self.log == 'natural' : 
            clamped_output = clamped_output.log() * self.C

        output = (clamped_output + self.addition_amount) / self.division_amount

        return output