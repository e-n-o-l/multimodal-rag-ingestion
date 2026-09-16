from torch.nn import Module, Linear
from .cross_attention import CrossAttention
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch


class EmbeddingCompressor(Module):
    def __init__(self, input_dim: int, target_dim: int, num_clusters: int, num_heads: int,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()

        self.linear = Linear(in_features=input_dim, out_features=target_dim)

        self.attention = CrossAttention(
            num_heads=num_heads,
            embed_dim=target_dim,
            seq_len=num_clusters
        ).to(device)

        self.device = device

    def forward(self, x: Tensor, imgs2offer: list[int]):
        x_comp = self.linear(x)

        grouped = torch.split(
            tensor=x_comp,
            split_size_or_sections=imgs2offer,
            dim=0
        )

        padded = pad_sequence(grouped, batch_first=True, padding_value=0)

        max_len = padded.size(1)

        lengths = torch.tensor(imgs2offer, device=self.device).unsqueeze(1)

        mask = torch.arange(max_len, device=self.device).unsqueeze(0) >= lengths

        centroids = self.attention(padded, padding_mask=mask)

        return x_comp, centroids