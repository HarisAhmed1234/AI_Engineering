"""M1 input-validation tests; M2 will extend this file with multi-head tests."""

import pytest
import torch

from src.attention import MultiHeadSelfAttention, SelfAttention
from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


def test_attention_rejects_wrong_rank_or_model_dimension() -> None:
    attention = SelfAttention(d_model=4)
    with pytest.raises(ValueError, match="shape"):
        attention(torch.randn(2, 4))
    with pytest.raises(ValueError, match="expected final dimension"):
        attention(torch.randn(2, 3, 5))


def test_multi_head_shapes_and_divisibility_assertion() -> None:
    attention = MultiHeadSelfAttention(d_model=8, n_heads=2, dropout=0.0).eval()
    output, weights = attention(torch.randn(3, 4, 8), causal=True)
    assert output.shape == (3, 4, 8)
    assert weights.shape == (3, 2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 2, 4), atol=1e-6)
    with pytest.raises(AssertionError, match="divisible"):
        MultiHeadSelfAttention(d_model=7, n_heads=2)


def test_sinusoidal_encoding_shape_and_known_position_zero_values() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=6, max_length=10)
    encoded = encoding(torch.zeros(2, 4, 6))
    assert encoded.shape == (2, 4, 6)
    assert torch.allclose(encoded[0, 0, 0::2], torch.zeros(3), atol=1e-6)
    assert torch.allclose(encoded[0, 0, 1::2], torch.ones(3), atol=1e-6)
    assert not torch.allclose(encoded[0, 0], encoded[0, 1])


def test_transformer_block_preserves_shape_and_uses_four_x_ffn_by_default() -> None:
    block = TransformerBlock(d_model=8, n_heads=2, dropout=0.0).eval()
    output, weights = block(torch.randn(2, 5, 8), return_attention=True)
    assert output.shape == (2, 5, 8)
    assert weights.shape == (2, 2, 5, 5)
    assert block.feed_forward[0].out_features == 32

