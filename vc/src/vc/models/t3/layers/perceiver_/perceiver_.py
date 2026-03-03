import math
import torch

import torch.nn as nn

from torch import Tensor

from .layers import AttentionBlock2

class Perceiver(nn.Module) : 
    '''
    Paper : Perceiver: General Perception with Iterative Attention (https://arxiv.org/abs/2103.03206)
    '''

    def __init__(
        self , 
        pre_attention_query_token : int = 32 , 
        pre_attention_query_size : int = 1024 , 
        embedding_dim : int = 1024 , 
        num_attn_heads : int = 4
    ) -> None : 

        '''
        Initialize the perceiver module.

        :param pre_attention_query_token: Number of query tokens for pre-attention
        :param pre_attention_query_size: Size of each query token
        :param embedding_dim: Dimension of the embedding space
        :param num_attn_heads: Number of attention heads
        '''
        super().__init__()

        # * Pre-attention query parameter
        self.pre_attention_query : nn.Parameter = nn.Parameter(
            torch.empty(1 , pre_attention_query_token , pre_attention_query_size)
        )

        # * Calculate the variance for uniform initialization
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))

        # * Initialize the pre-attention query with uniform distribution
        self.pre_attention_query.data.uniform_(-query_variance , query_variance)

        # * Initialize the attention block
        self.attn = AttentionBlock2(embedding_dim , num_attn_heads)

    def forward(self, h : Tensor) -> Tensor : 
        '''
        Forward pass of the perceiver module.
        :param h: Input tensor
        :return: Output after applying attention mechanisms
        '''

        # * Expand the pre-attention query to match the batch size of the input
        query_ = self.pre_attention_query.expand(h.shape[0] , -1 , -1)

        # * Apply the first attention mechanism (cross-attention)
        pre_att = self.attn(query_ , h)

        # * Apply the second attention mechanism (self-attention)
        attn = self.attn(pre_att , pre_att)

        return attn