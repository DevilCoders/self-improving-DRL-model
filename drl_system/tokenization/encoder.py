"""Sequence encoder using a small Transformer-style stack."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding(tokens)
        return self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.forward(tokens)
        pooled = encoded.mean(dim=1)
        return encoded, pooled


__all__ = ["SequenceEncoder"]
