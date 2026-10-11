"""Self-attention written with elementary PyTorch tensor operations.

This module deliberately avoids PyTorch's high-level attention/Transformer wrappers
and fused scaled-attention helper. Multi-head attention is added in M2; this M1 class
is one projected self-attention head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SelfAttention(nn.Module):
    """Projected scaled dot-product self-attention for a single head.

    Given ``x`` with shape ``(batch, time, d_model)``, the query and key projections
    have shape ``(batch, time, d_k)`` and the value projection has shape
    ``(batch, time, d_v)``.  Their score matrix is ``(batch, time, time)``.  The
    returned output is projected back to ``(batch, time, d_model)`` and the returned
    attention weights remain ``(batch, time, time)`` for inspection and tests.
    """

    def __init__(
        self,
        d_model: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        if self.d_k <= 0 or self.d_v <= 0:
            raise ValueError("d_k and d_v must be positive")

        self.q_proj = nn.Linear(d_model, self.d_k, bias=bias)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=bias)
        self.v_proj = nn.Linear(d_model, self.d_v, bias=bias)
        self.out_proj = nn.Linear(self.d_v, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

    @staticmethod
    def causal_mask(sequence_length: int, device: torch.device) -> Tensor:
        """Return True exactly where a query must not see a future key."""
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x: Tensor, causal: bool = False) -> tuple[Tensor, Tensor]:
        """Compute attention and return ``(output, attention_weights)``.

        ``causal=True`` replaces every score above the diagonal with negative
        infinity *before* softmax, making its resulting probability exactly zero.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, d_model)")
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"expected final dimension {self.d_model}, got {x.size(-1)}"
            )

        _, time, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            scores = scores.masked_fill(self.causal_mask(time, x.device), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        # Dropout is applied only while training; use eval mode for probability tests.
        weights = self.attention_dropout(weights)
        context = weights @ values  # (B, T, T) @ (B, T, d_v) -> (B, T, d_v)
        return self.out_proj(context), weights


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head causal self-attention with explicit split and merge steps.

    Input: ``(B, T, D)`` where ``D=d_model``.  Q, K, and V are projected to
    ``(B, T, D)``, split into ``(B, H, T, d_head)``, attended independently by
    head, concatenated back to ``(B, T, D)``, then passed through ``out_proj``.
    ``d_model`` must divide evenly across the heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """Reshape `(B,T,D)` to `(B,H,T,d_head)` without mixing token order."""
        batch, time, _ = tensor.shape
        return tensor.reshape(batch, time, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """Reverse the head split: `(B,H,T,d_head)` becomes `(B,T,D)`."""
        batch, _, time, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().reshape(batch, time, self.d_model)

    def forward(
        self, x: Tensor, causal: bool = False, return_details: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Return output/weights, plus named intermediate tensors when requested."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, d_model)")
        if x.size(-1) != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.size(-1)}")

        _, time, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries_by_head = self._split_heads(queries)
        keys_by_head = self._split_heads(keys)
        values_by_head = self._split_heads(values)

        scores = (queries_by_head @ keys_by_head.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal:
            scores = scores.masked_fill(SelfAttention.causal_mask(time, x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        per_head_outputs = weights @ values_by_head
        concatenated = self._merge_heads(per_head_outputs)
        output = self.out_proj(concatenated)

        if not return_details:
            return output, weights
        details = {
            "input": x,
            "queries": queries,
            "keys": keys,
            "values": values,
            "queries_by_head": queries_by_head,
            "keys_by_head": keys_by_head,
            "values_by_head": values_by_head,
            "scores": scores,
            "weights": weights,
            "per_head_outputs": per_head_outputs,
            "concatenated": concatenated,
            "output": output,
        }
        return output, weights, details

    def shape_trace(self, x: Tensor, causal: bool = True) -> tuple[str, dict[str, Tensor]]:
        """Return a complete human-readable shape/value trace for Figure 3.

        The trace lists all tensor transformations and prints each head's score,
        probability, and output tensors for the first batch item. It is intended
        for a small instructional input, not for a full training batch.
        """
        _, _, details = self(x, causal=causal, return_details=True)
        lines = [
            "MULTI-HEAD ATTENTION SHAPE TRACE",
            f"input:                 {tuple(details['input'].shape)} = (B, T, D)",
            f"Q projection:          {tuple(details['queries'].shape)} = (B, T, D)",
            f"K projection:          {tuple(details['keys'].shape)} = (B, T, D)",
            f"V projection:          {tuple(details['values'].shape)} = (B, T, D)",
            f"split Q/K/V:           {tuple(details['queries_by_head'].shape)} = (B, H, T, d_head)",
            f"per-head scores:       {tuple(details['scores'].shape)} = (B, H, T, T)",
            f"per-head weights:      {tuple(details['weights'].shape)} = (B, H, T, T)",
            f"per-head outputs:      {tuple(details['per_head_outputs'].shape)} = (B, H, T, d_head)",
            f"concatenated heads:    {tuple(details['concatenated'].shape)} = (B, T, D)",
            f"final output projection:{tuple(details['output'].shape)} = (B, T, D)",
        ]
        for head in range(self.n_heads):
            lines.extend(
                [
                    f"\nHead {head} scores (batch 0):\n{details['scores'][0, head].detach().cpu()}",
                    f"Head {head} attention weights (batch 0):\n{details['weights'][0, head].detach().cpu()}",
                    f"Head {head} output (batch 0):\n{details['per_head_outputs'][0, head].detach().cpu()}",
                ]
            )
        return "\n".join(lines), details

