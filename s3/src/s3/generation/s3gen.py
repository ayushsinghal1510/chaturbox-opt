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

import logging

import numpy as np
import torch
from torch.types import Device
import torchaudio as ta
from functools import lru_cache

from torch import Tensor

from ..tokenizer import S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS
import torch.nn as nn

from ..modules import MEL_SPEC

S3_SR = 16_000
SPEECH_VOCAB_SIZE = 6561


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)

from .services_ import SERVICES


class S3Token2Mel(nn.Module , SERVICES) : 

    def __init__(
        self , 
        tokenizer : S3Tokenizer , 
        device : Device , 
        config : dict , 
        mel_spec : MEL_SPEC
    ) -> None : 
        super().__init__()

        self.tokenizer : S3Tokenizer = tokenizer
        self.mel_spec : MEL_SPEC = mel_spec
        
        self.device : Device = device

        self.speaker_encoder : CAMPPlus = CAMPPlus()

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

        self.resamplers = {}

    def embed_ref(self , ref_wav : Tensor
    ) -> dict : 

        assert len(ref_wav.shape) == 2 , f'Invalid Shape , {ref_wav.shape}'

        if ref_wav.size(1) > 10 * 16_000 : 
            print("WARNING: cosydec received ref longer than 10s")

        mel : Tensor = self.mel_spec(ref_wav)

        # * Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav)

        # * Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav)

        # * Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if mel.shape[1] != 2 * ref_speech_tokens.shape[1] : 
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :mel.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token = ref_speech_tokens.to(self.device) , 
            prompt_token_len = ref_speech_token_lens , 
            prompt_feat = mel , 
            prompt_feat_len = None , 
            embedding = ref_x_vector
        )

    def forward(
        self , 
        speech_tokens : Tensor , # * S3 Tokens of what to say
        ref_wav : Tensor , # * S3 Tokens of how to say
        ref_sr : int = 24_000 , # * Sample rate of how to say audio
        ref_dict : dict | None = None , 
        finalize : bool = True
    ) -> Tensor : 

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav)
        else:
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(self.device)

        assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"

        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        output_mels , _ = self.flow.inference(
            token = speech_tokens , 
            token_len = speech_token_lens , 
            finalize = finalize , 
            **ref_dict
        )

        return output_mels

class S3Token2Wav(S3Token2Mel) : 

    def __init__(
        self , 
        tokenizer : S3Tokenizer , 
        device : Device , 
        config : dict , 
        mel_spec : MEL_SPEC
    ) : 

        super().__init__(
            tokenizer = tokenizer , 
            device = device , 
            config = config , 
            mel_spec = mel_spec
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

    def forward(
        self , 
        speech_tokens : Tensor , 
        ref_wav : Tensor , 
        ref_sr : int , 
        ref_dict : dict | None = None  ,
        finalize : bool = False
    ) -> Tensor : 

        output_mels = super().forward(
            speech_tokens , 
            ref_wav = ref_wav , 
            ref_sr = ref_sr , 
            ref_dict = ref_dict , 
            finalize = finalize
        )

        hift_cache_source = torch.zeros(1 , 1 , 0).to(self.device)

        output_wavs , *_ = self.mel2wav.inference(
            speech_feat = output_mels , 
            cache_source = hift_cache_source
        )

        if not self.training : 
            # * NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self , 
        speech_tokens : Tensor , # * S3 Tokens of what to say
        ref_wav : Tensor , # * S3 Tokens of how to say
        ref_sr : int = 24_000 , # * Sample rate of how to say audio
        ref_dict : dict | None = None , 
        finalize : bool = True
    ) : 
        return super().forward(
            speech_tokens , 
            ref_wav = ref_wav , 
            ref_sr = ref_sr , 
            ref_dict = ref_dict , 
            finalize = finalize
        )

    @torch.inference_mode()
    def hift_inference(
        self , 
        speech_feat : Tensor , 
        cache_source : Tensor | None = None
    ) -> Tensor : 

        if cache_source is None : 
            cache_source = torch.zeros(1 , 1 , 0).to(self.device)

        return self.mel2wav.inference(
            speech_feat = speech_feat , 
            cache_source = cache_source
        )

    @torch.inference_mode()
    def inference(
        self , 
        speech_tokens , # * S3 Tokens of what to say
        ref_wav : Tensor , # * S3 Tokens of how to say
        ref_sr : int = 16_000 , # * Sample rate of how to say audio
        ref_dict : dict | None = None , 
        cache_source : Tensor | None = None , 
        finalize : bool = True
    ) -> tuple : 

        output_mels = self.flow_inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)
        output_wavs , output_sources = self.hift_inference(
            output_mels , 
            cache_source
        )

        # * NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources
