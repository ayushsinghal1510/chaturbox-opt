import torch

import torch.nn as nn

from accelerate import Accelerator

from torch import Tensor , LongTensor

from s3tokenizer.model_v2 import (
    S3TokenizerV2 , 
    ModelConfig , 
)

from s3tokenizer.utils import padding as s3_padding

from services import MEL_SPEC
# from modules.src.modules import MEL_SPEC

from .services_ import SERVICES

class S3Tokenizer(S3TokenizerV2) : 

    def __init__(
        self , 
        config : dict , 
        mel_spec : MEL_SPEC , 
        model_config : ModelConfig = ModelConfig() , 
        accelerator : Accelerator | None = None , 
    ) : 

        super().__init__(name=config['model-name'])

        self.config : dict = config

        self.mel_spec_fn : MEL_SPEC = mel_spec

    # ! Add batch processing here
    @torch.no_grad()
    def forward(
        self , 
        wav : Tensor , # * [channel , audio]
        max_len : int | None = None
    ) -> tuple[Tensor , LongTensor | Tensor] : 

        assert wav.dim() == 2 and wav.size(0) == 1 , 'Input must be a single audio sample with shape [1, Time]'

        wav = wav.to(self.device)

        mel = self.mel_spec_fn(wav)

        mel , mel_len = s3_padding([mel.squeeze(0)])

        print(mel.shape)

        speech_tokens , speech_token_lens = self.quantize(
            mel , 
            mel_len
        )

        return (
            speech_tokens.long().detach() , 
            speech_token_lens.long().detach()
        )
