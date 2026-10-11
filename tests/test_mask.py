"""M1 causal-mask correctness tests."""

import torch

from src.attention import MultiHeadSelfAttention, SelfAttention
from src.transformer_block import TransformerBlock


def _identity_attention() -> SelfAttention:
    attention = SelfAttention(d_model=2, d_k=2, d_v=2, dropout=0.0, bias=False).eval()
    with torch.no_grad():
        identity = torch.eye(2)
        attention.q_proj.weight.copy_(identity)
        attention.k_proj.weight.copy_(identity)
        attention.v_proj.weight.copy_(identity)
        attention.out_proj.weight.copy_(identity)
    return attention


def test_causal_mask_assigns_exactly_zero_weight_to_future_positions() -> None:
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    _, weights = _identity_attention()(x, causal=True)
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[0][future], torch.zeros_like(weights[0][future]))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_changing_future_tokens_cannot_change_earlier_causal_outputs() -> None:
    attention = _identity_attention()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]])
    changed_future = x.clone()
    changed_future[:, 2:] = torch.tensor([[[100.0, -100.0], [-50.0, 50.0]]])

    original_output, _ = attention(x, causal=True)
    changed_output, _ = attention(changed_future, causal=True)

    assert torch.equal(original_output[:, :2], changed_output[:, :2])


def test_multi_head_causal_weights_are_zero_above_the_diagonal() -> None:
    torch.manual_seed(11)
    attention = MultiHeadSelfAttention(d_model=4, n_heads=2, dropout=0.0).eval()
    _, weights = attention(torch.randn(1, 4, 4), causal=True)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.equal(weights[0, :, future], torch.zeros_like(weights[0, :, future]))


def test_transformer_block_is_causal_when_future_tokens_change() -> None:
    torch.manual_seed(29)
    block = TransformerBlock(d_model=6, n_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 5, 6)
    altered = x.clone()
    altered[:, 3:] = torch.randn(1, 2, 6) * 50
    original = block(x)
    changed = block(altered)
    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6)

