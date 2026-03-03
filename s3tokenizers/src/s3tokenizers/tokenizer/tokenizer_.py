import torch

import torch.nn as nn

from accelerate import Accelerator

from torch import Tensor , LongTensor

from s3tokenizer.model_v2 import (
    S3TokenizerV2 , 
    ModelConfig , 
)

from modules import MEL_SPEC
# from modules.src.modules import MEL_SPEC

from .services_ import SERVICES

class S3Tokenizer(S3TokenizerV2 , SERVICES , nn.Module) : 

    def __init__(
        self , 
        config : dict , 
        mel_spec : MEL_SPEC , 
        model_config : ModelConfig = ModelConfig() , 
        accelerator : Accelerator | None = None , 
    ) : 

        nn.Module.__init__(self)
        S3TokenizerV2.__init__(self , name = config['model-name'])

        self.config : dict = config

        self.n_fft = self.config['n_fft']
        self.mel_spec_fn : MEL_SPEC = mel_spec

    # ! Add batch processing here
    @torch.inference_mode()
    def forward(
        self , 
        wav : Tensor , # * [channel , audio]

        max_len : int | None = None
    ) -> tuple[Tensor , LongTensor | Tensor] : 

        assert wav.dim() == 2 and wav.size(0) == 1 , 'Input must be a single audio sample with shape [1, Time]'

        wav = wav.to(self.device)

        mel = self.mel_spec_fn(wav)

        mel_len = torch.tensor([mel.shape[-1]] , device = self.device)

        speech_tokens , speech_token_lens = self.quantize(
            mel , 
            mel_len
        )

        return (
            speech_tokens.long().detach() , 
            speech_token_lens.long().detach()
        )
