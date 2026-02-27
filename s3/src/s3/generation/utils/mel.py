"""mel-spectrogram extraction in Matcha-TTS"""
import logging
from librosa.filters import mel as librosa_mel_fn
import torch
import numpy as np

from torch import Tensor

logger = logging.getLogger(__name__)


# NOTE: they decalred these global vars
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

"""

def mel_spectrogram(
    audio : Tensor , 
    n_fft = 1920 , 
    num_mels = 80 , 
    sampling_rate = 24000 , 
    hop_size = 480 , 
    win_size = 1920 , 
    fmin = 0 , 
    fmax = 8000 , 
    center = False
) -> Tensor : 

    min_val = torch.min(audio)
    max_val = torch.max(audio)

    if min_val < -1.0 or max_val > 1.0 : 
        logger.warning(f"Audio values outside normalized range: min={min_val.item():.4f}, max={max_val.item():.4f}")

    global mel_basis, hann_window 

    if f"{str(fmax)}_{str(audio.device)}" not in mel_basis : 

        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(audio.device)] = torch.from_numpy(mel).float().to(audio.device)
        hann_window[str(audio.device)] = torch.hann_window(win_size).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    audio = audio.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            audio,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(audio.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(audio.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
