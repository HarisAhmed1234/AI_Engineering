"""Pre-LayerNorm decoder block for a causal Transformer language model."""

from __future__ import annotations

from torch import Tensor, nn

from src.attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Causal decoder block: pre-LN attention + residual, then pre-LN FFN + residual.

    Pre-LayerNorm normalizes the input to each sublayer *before* the residual branch.
    This leaves a direct residual gradient path and is generally more stable in deep
    language models, reducing the need for unusually careful warmup settings.
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        self.ln_attention = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Apply causal attention and a position-wise FFN without changing shape."""
        attention_output, weights = self.attention(self.ln_attention(x), causal=True)
        x = x + self.residual_dropout(attention_output)
        x = x + self.feed_forward(self.ln_ffn(x))
        if return_attention:
            return x, weights
        return x

