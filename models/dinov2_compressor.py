from torch import Tensor
from torch.nn import Module
import torch
from .embedding_compressor import EmbeddingCompressor
from transformers import AutoModel


class DINOv2EmbeddingCompressor(Module):

    def __init__(self, model_name: str = "facebook/dinov2-base", embed_dim: int = 768,
                 compressor_model_path: str = "compressor_model.pt",
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.device = device

        self.embedder = AutoModel.from_pretrained(model_name).to(device)
        self.embedder.eval()

        self.compressor = EmbeddingCompressor(
            input_dim=embed_dim,
            target_dim=64,
            num_clusters=8,
            num_heads=4
        ).to(device)

        self.compressor.load_state_dict(
            torch.load(compressor_model_path, map_location=device)
        )
        self.compressor.requires_grad_(False)
        self.compressor.eval()


    def forward(self, imgs: Tensor, imgs2offer: list) -> Tensor:
        outputs = self.embedder(imgs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        normalized_embeddings = cls_embeddings / torch.linalg.norm(cls_embeddings, axis=-1, keepdims=True)

        _, centroids = self.compressor(normalized_embeddings, imgs2offer)

        return centroids