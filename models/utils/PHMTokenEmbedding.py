import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch.nn.functional import *
class PHMEmbedding(Module):
   
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings: int, embedding_dim: int, phm: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PHMEmbedding, self).__init__()
        self.num = num_embeddings
        self.emb = embedding_dim

        rest = num_embeddings % phm
        if rest != 0 :
            num_embeddings += (phm-rest)

        rest0 = embedding_dim % phm
        if rest0 != 0 :
            embedding_dim += (phm-rest0)
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            ##########################################################################################
            self.a = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((phm, phm, phm))))

            self.s = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((phm, num_embeddings//phm, embedding_dim//phm))))

            self.weight = torch.zeros((num_embeddings, embedding_dim))
            ##########################################################################################
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        #init.normal_(self.weight)
        #self._fill_padding_idx_with_zero() #not used
        pass

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
            
    ##########################################################################################
    def kronecker_product1(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out
    ##########################################################################################
    def forward(self, input: Tensor) -> Tensor:
        ##########################################################################################
        self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
        #print("input.shape:",input.shape)
        #if self.weight.size(-1) != input.size(-1):
        #    self.weight = self.weight[:,:input.size(-1)]

        
        ##########################################################################################
        #print("print(self.weight.shape)_before:",self.weight.shape)
        #self._fill_padding_idx_with_zero()
        self.weight = self.weight[:self.num,:self.emb]
        #print(self.weight.shape)
        #print("print(self.weight.shape)_after:",self.weight.shape)
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
"""
layer = PHMEmbedding(53,64,2,0)
indices = torch.tensor([1, 2, 3], dtype=torch.long)
layer(indices)
"""