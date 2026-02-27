import torch

import torch.nn as nn 

import torch.nn.functional as F

from torch import Tensor

from .relative_position_bias import RelativePositionBias

class AttentionQKV(nn.Module) : 

    def __init__(
        self , 
        n_heads : int , 
        head_dim : int , 
        dropout_rate : float = 0.1 , 
        scale : None | float = None , 
        flash : bool = False
    ) -> None : 

        super().__init__()

        self.n_heads : int = n_heads
        self.head_dim : int = head_dim
        self.scale : float = scale if scale is not None else head_dim ** -0.5
        self.flash : bool = flash
        self.dropout_rate : float = dropout_rate

        self.dropout : nn.Dropout = nn.Dropout(dropout_rate)

        self.flash_config : None | dict = self.setup_flash_config() if flash else None

    # ! Add flash attention here
    def setup_flash_config(self) -> dict : 

        flash_config = {
            'enable_flash' : True , 
            'enable_math' : True , 
            'enable_mem_efficient' : True
        }

        return flash_config

    def forward(
        self , 
        q : Tensor , 
        k : Tensor , 
        v : Tensor , 
        mask : None = None
    ) -> Tensor : 

        q , k , v = [self.split_heads(tensor) for tensor in [q , k , v]]

        if self.flash : 
            out = self.flash_attention(q , k , v , mask = mask)

        else : 
            out = self.scaled_dot_product_attention(q , k , v , mask = mask)

        return self.combine_heads(out)

    def scaled_dot_product_attention(
        self , 
        q : Tensor , 
        k : Tensor , 
        v : Tensor , 
        mask : None = None
    ) -> Tensor : 

        sim = torch.einsum("bhlt,bhls->bhts" , q , k) * self.scale

        if mask is not None : 
            sim = sim.masked_fill(mask == 0 , float('-inf'))

        attn = torch.softmax(sim , dim = -1)
        attn : Tensor = self.dropout(attn)

        return torch.einsum("bhts,bhls->bhlt" , attn , v)

    def flash_attention(
        self , 
        q : Tensor , 
        k : Tensor , 
        v : Tensor , 
        mask : None = None
    ) -> Tensor : 

        config = self.flash_config if self.flash_config else {}

        with torch.backends.cuda.sdp_kernel(**config):

            out : Tensor = F.scaled_dot_product_attention(
                q , k , v , 
                attn_mask = mask , 
                dropout_p = self.dropout_rate if self.training else 0
            )

        return out

    def split_heads(self , x : Tensor) -> Tensor : 

        bs , length , _ = x.shape

        x = x.view(bs , length , self.n_heads , self.head_dim)

        return x.permute(0 , 2 , 1 , 3)

    def combine_heads(self , x : Tensor) -> Tensor : 

        bs , _ , length , _ = x.shape

        x = x.permute(0 , 2 , 1 , 3).contiguous()

        return x.view(bs , length , -1)

class AttentionBlock2(nn.Module) : 

    """
    An attention block that allows spatial positions to attend to each other,
    using AttentionQKV and separate linear transformations for Q, K, and V.
    """

    def __init__(
        self , 
        channels : int , 
        num_heads : int = 1 , 
        num_head_channels : int = -1 , 
        relative_pos_embeddings : bool = False , 
        flash_attention : bool = True , 
        dropout_rate : float = 0.2 , 
        scale : None = None
    ) -> None : 

        super().__init__()

        self.channels : int = channels

        if num_head_channels == -1 : 
            self.num_heads = num_heads

        else : 
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm : nn.LayerNorm = nn.LayerNorm(channels)

        # * Separate linear layers for Q, K, and V
        self.to_q : nn.Linear = nn.Linear(channels , channels)
        self.to_k : nn.Linear = nn.Linear(channels , channels)
        self.to_v : nn.Linear = nn.Linear(channels , channels)

        self.attention = AttentionQKV(self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale)

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x1, x2, mask=None):
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)