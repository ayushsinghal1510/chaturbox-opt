import torch

import torch.nn as nn

from torch import Tensor

class LearnedPositionEmbeddings(nn.Module) : 

    def __init__(
        self , 
        seq_len , 
        model_dim , 
        init = .02
    ) : 

        super().__init__()

        self.emb : nn.Embedding = nn.Embedding(seq_len , model_dim)

        self.emb.weight.data.normal_(mean = 0.0 , std = init)

    def forward(self , x : Tensor) -> Tensor : 
        '''
        Returns positional embeddings for index 0 up to the length of x
        '''

        sl = x.shape[1]

        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self , idx : int | Tensor) -> Tensor : 
        '''
        Args:
            idx: scalar int or an integer tensor of shape (T,) or (B, T)
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        '''

        device = self.emb.weight.device

        tensor_idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)

        tensor_idx = torch.atleast_2d(tensor_idx)

        assert tensor_idx.ndim == 2

        return self.emb(tensor_idx)  # (B, T, dim)
