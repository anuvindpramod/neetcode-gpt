import torch
import torch.nn as nn
from torchtyping import TensorType
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key=nn.Linear(embedding_dim,attention_dim,bias=False) # (B,Tensor,atten_dim)
        self.query=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.value=nn.Linear(embedding_dim,attention_dim,bias=False)

        
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K=self.key(embedded)
        Q=self.query(embedded)
        V=self.value(embedded)

        scores=Q @ torch.transpose(K,1,2)
        context_len,attn_dim=K.shape[1],K.shape[2]
        scores=scores/math.sqrt(attn_dim)
        lower_triangle=torch.tril(torch.ones(context_len,context_len))
        mask=lower_triangle==0

        scores=scores.masked_fill(mask,float('-inf'))
        scores=F.softmax(scores,dim=2) @ V

        return torch.round(scores,decimals=4)
      
    