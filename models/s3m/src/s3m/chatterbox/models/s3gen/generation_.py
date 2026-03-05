# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.types import Device

from torch import Tensor

from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS
import torch.nn as nn

class S3Token2Wav(nn.Module) : 

    def __init__(
        self , 
        device : Device , 
        config : dict , 
    ) : 

        super().__init__()

        self.device : Device = device

        encoder : UpsampleConformerEncoder = UpsampleConformerEncoder()

        estimator : ConditionalDecoder = ConditionalDecoder()
        cfm_params = CFM_PARAMS

        decoder : CausalConditionalCFM = CausalConditionalCFM(
            spk_emb_dim = 80 , 
            cfm_params = cfm_params , 
            estimator = estimator
        )

        self.flow : CausalMaskedDiffWithXvec = CausalMaskedDiffWithXvec(
            encoder = encoder , 
            decoder = decoder
        )

        f0_predictor = ConvRNNF0Predictor()

        self.mel2wav = HiFTGenerator(
            sampling_rate = S3GEN_SR , 
            upsample_rates = [8 , 5 , 3] , 
            upsample_kernel_sizes = [16 , 11 , 7] , 
            source_resblock_kernel_sizes = [7 , 7 , 11] , 
            source_resblock_dilation_sizes = [[1 , 3 , 5] , [1 , 3 , 5] , [1 , 3 , 5]] , 
            f0_predictor = f0_predictor
        )

        # * silence out a few ms and fade audio in to reduce artifacts

        n_trim = S3GEN_SR // 50  # * 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)

        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi , 0 , n_trim)) + 1) / 2

        self.register_buffer("trim_fade", trim_fade, persistent=False) # * (buffers get automatic device casting)

    @torch.inference_mode()
    def inference(
        self , 
        speech_tokens , # * S3 Tokens of what to say
        ref_dict : dict , 
        cache_source : Tensor | None = None , 
        finalize : bool = True
    ) -> tuple[Tensor , Tensor] : 

        assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"

        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        output_mels , _ = self.flow.inference(
            token = speech_tokens , 
            token_len = speech_token_lens , 
            finalize = finalize , 
            **ref_dict
        )

        if cache_source is None : 
            cache_source = torch.zeros(1 , 1 , 0).to(self.device)

        output_wavs , output_sources = self.mel2wav.inference(
            speech_feat = output_mels , 
            cache_source = cache_source
        )

        # * NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs , output_sources