"""Trainable classification heads; no generative model dependencies."""

import torch
from torch import nn

from llm_specialist.config import EMBEDDING_DIM, SpecialistConfig


class ClassificationHead(nn.Module):
    """Linear baseline or configurable GELU MLP on frozen embeddings."""

    def __init__(self, config: SpecialistConfig, kind: str = "mlp") -> None:
        super().__init__()
        if kind not in ("linear", "mlp"):
            raise ValueError("head kind must be linear or mlp.")
        self.kind = kind
        self.num_labels = len(config.labels)
        if kind == "linear":
            self.layers = nn.Linear(EMBEDDING_DIM, self.num_labels)
        else:
            self.layers = nn.Sequential(
                nn.Linear(EMBEDDING_DIM, config.head.hidden_size),
                nn.GELU(),
                nn.Dropout(config.head.dropout),
                nn.Linear(config.head.hidden_size, self.num_labels),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of FP32 embeddings."""
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Head input must have shape [batch, {EMBEDDING_DIM}].")
        if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
            raise ValueError("Head input must contain finite FP32 embeddings.")
        return self.layers(embeddings)


def loss_function(config: SpecialistConfig, device: torch.device) -> nn.CrossEntropyLoss:
    """Construct the configured single-label loss on the head device."""
    weights = config.training.class_weights
    tensor = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=tensor, label_smoothing=config.training.label_smoothing)
