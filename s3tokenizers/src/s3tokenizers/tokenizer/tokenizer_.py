import torch
import librosa

from accelerate import Accelerator

from torch import Tensor , LongTensor

from s3tokenizer.utils import padding

from s3tokenizer.model_v2 import (
    S3TokenizerV2 , 
    ModelConfig , 
)

from .services_ import SERVICES

class S3Tokenizer(S3TokenizerV2 , SERVICES) : 

    _mel_filters : Tensor 
    window : Tensor 

    '''
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    '''

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self , 
        config : dict , 
        model_config : ModelConfig = ModelConfig()
    ) : 

        super().__init__(config['model-name'])

        self.config : dict = config

        self.n_fft = self.config['n_fft']

        _mel_filters = librosa.filters.mel(
            sr = self.config['sample-rate'] , 
            n_fft = self.n_fft , 
            n_mels = model_config.n_mels
        )

        self.register_buffer(
            '_mel_filters' , 
            torch.FloatTensor(_mel_filters)
        )

        self.register_buffer(
            'window' , 
            torch.hann_window(self.n_fft)
        )

    @torch.no_grad()
    def forward(
        self , 
        wav : Tensor , # * [audio]
        accelerator : Accelerator | None = None , 
        max_len : int | None = None
    ) -> tuple[Tensor , LongTensor | Tensor] : 

        assert wav.dim() == 2 and wav.size(0) == 1 , 'Input must be a single audio sample with shape [1, Time]'

        wav = wav.to(self.device)

        mel = self.log_mel_spectrogram(
            audio = wav , 
            device = self.device , 
            n_fft = self.n_fft , 
            window = self.window , 
            _mel_filters = self._mel_filters , 
            hop_length = self.config['hop_length'] , 
            padding = self.config['padding']
        ).to(self.device)  # [B=1, F, T]

        mel_len = torch.tensor([mel.shape[-1]] , device = self.device)

        if accelerator is None : 
            tokenizer = self

        else : 
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens , speech_token_lens = tokenizer.quantize(
            mel , 
            mel_len
        )

        return (
            speech_tokens.long().detach() , 
            speech_token_lens.long().detach()
        )
