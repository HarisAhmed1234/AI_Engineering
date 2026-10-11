"""M1 tests for manually implemented scaled dot-product attention."""

import torch

from src.attention import SelfAttention


def test_attention_output_and_weight_shapes_and_row_sums() -> None:
    torch.manual_seed(7)
    attention = SelfAttention(d_model=6, d_k=4, d_v=5, dropout=0.0).eval()
    x = torch.randn(2, 5, 6)

    output, weights = attention(x)

    assert output.shape == (2, 5, 6)
    assert weights.shape == (2, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_numeric_attention_matches_hand_calculation() -> None:
    """Use projections to reproduce the explicit Q, K, V from M1 exactly."""
    attention = SelfAttention(d_model=3, d_k=2, d_v=2, dropout=0.0, bias=False).eval()
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # One-hot inputs let each projection's columns store one explicit token vector.
    with torch.no_grad():
        attention.q_proj.weight.copy_(q.T)
        attention.k_proj.weight.copy_(k.T)
        attention.v_proj.weight.copy_(v.T)
        attention.out_proj.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        )

    one_hot_tokens = torch.eye(3).unsqueeze(0)
    output, weights = attention(one_hot_tokens)

    expected_weights = torch.tensor(
        [
            [0.4011121, 0.1977758, 0.4011121],
            [0.1977758, 0.4011121, 0.4011121],
            [0.2482551, 0.2482551, 0.5034899],
        ]
    )
    expected_context = expected_weights @ v
    assert torch.allclose(weights[0], expected_weights, atol=1e-6)
    assert torch.allclose(output[0, :, :2], expected_context, atol=1e-6)

