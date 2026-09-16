import torch
from torch.nn import Module, MultiheadAttention, Parameter
from torch import Tensor
from torch.nn.functional import normalize


class CrossAttention(Module):
    def __init__(self, num_heads: int, embed_dim: int, seq_len: int,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()

        self.cross_attention = MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True
        )

        self.query = Parameter(torch.randn(1, seq_len, embed_dim))

        self.device = device

    def forward(self, x: Tensor, padding_mask=None) -> Tensor:
        batch_size = x.size(0)

        query = self.query.to(self.device).repeat(batch_size, 1, 1)

        compressed, _ = self.cross_attention(
            query=query,
            value=x, key=x,
            key_padding_mask=padding_mask,
        )

        return normalize(compressed, dim=-1)