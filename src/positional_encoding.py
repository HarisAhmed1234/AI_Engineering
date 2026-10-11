"""Position information for attention layers.

Plain attention is permutation-equivariant: swapping tokens and swapping their
rows/columns produces the same pattern. These modules add a vector that identifies
each position before attention is calculated.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position vectors to `(B,T,D)` token embeddings.

    For position ``p`` and paired feature index ``i``:

    ``PE[p, 2i] = sin(p / 10000^(2i / D))``
    ``PE[p, 2i+1] = cos(p / 10000^(2i / D))``

    The buffer is not learned or optimized. Different frequencies let the model
    represent both short and long relative changes in position.
    """

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequencies = torch.exp((-math.log(10000.0) / d_model) * even_dimensions)
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, d_model)")
        if x.size(-1) != self.encoding.size(-1):
            raise ValueError("x final dimension must equal d_model")
        if x.size(1) > self.encoding.size(1):
            raise ValueError("sequence length exceeds configured max_length")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class LearnedPositionalEmbedding(nn.Module):
    """Optional learned-position baseline for the M2 comparison experiment."""

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)
        self.max_length = max_length

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(1) > self.max_length:
            raise ValueError("x must be (batch, time, d_model) within max_length")
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions).unsqueeze(0).to(dtype=x.dtype)

