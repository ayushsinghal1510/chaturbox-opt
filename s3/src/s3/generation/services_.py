from typing import Callable

from torch import Tensor
from torch.types import Device
from torchaudio.transforms import Resample


class SERVICES : 

    def __init__(self) -> None : 
        pass

    def get_mel_spectrogram(
        self , 
        mel_spectrogram_func : Callable , 
        audio : Tensor , 
        device : Device , 
        audio_sr : int = 16_000
    ) -> Tensor : 

        self._16_to_24 : Resample = Resample(
            orig_freq = 16_000 , 
            new_freq = 24_000
        )

        if audio_sr == 16_000 : 
            audio = self._16_to_24(audio)

        mel_tensor : Tensor = mel_spectrogram_func(audio).transpose(1 , 2).to(device)

        return mel_tensor


