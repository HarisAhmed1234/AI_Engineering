"""Worked 3-token attention calculation and verification against ``SelfAttention``.

Run with ``python -m experiments.hand_calculation``.  It prints screenshot-ready
tables, saves Figure 1, saves terminal evidence, and checks that the hand values
match the same manually implemented PyTorch module used by the project.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.attention import SelfAttention


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig1_hand_calculated_attention.png"
MASK_FIGURE_PATH = ROOT / "outputs" / "figures" / "fig2_causal_mask.png"
EVIDENCE_PATH = ROOT / "outputs" / "logs" / "m1_hand_calculation.txt"


def matrix_text(name: str, matrix: torch.Tensor) -> str:
    """Format a small tensor as a compact terminal table."""
    rows = []
    for row in matrix.tolist():
        cells = ["-inf" if value == float("-inf") else f"{value:8.4f}" for value in row]
        rows.append("[" + ", ".join(cells) + "]")
    return f"{name} =\n" + "\n".join(rows)


def make_identity_projection_module(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> SelfAttention:
    """Build a projected module that emits supplied Q, K, V from one-hot tokens."""
    module = SelfAttention(d_model=3, d_k=2, d_v=2, dropout=0.0, bias=False).eval()
    with torch.no_grad():
        module.q_proj.weight.copy_(q.T)
        module.k_proj.weight.copy_(k.T)
        module.v_proj.weight.copy_(v.T)
        module.out_proj.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        )
    return module


def create_figure(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_scores: torch.Tensor,
    scaled_scores: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    masked_scores: torch.Tensor,
    masked_weights: torch.Tensor,
    masked_output: torch.Tensor,
) -> None:
    """Save a clear one-page visual record of all seven hand-calculation steps."""
    figure, axes = plt.subplots(3, 4, figsize=(17, 12))
    panels = [
        ("Q", q),
        ("K", k),
        ("V", v),
        ("QK^T", raw_scores),
        ("Scaled QK^T", scaled_scores),
        ("Softmax weights", weights),
        ("Output = weights @ V", output),
        ("Causally masked scores", masked_scores),
        ("Masked weights / output", masked_weights),
        ("Masked output = weights @ V", masked_output),
    ]
    for axis, (title, values) in zip(axes.flat, panels):
        shown = torch.nan_to_num(values.detach().cpu(), neginf=-2.0).numpy()
        image = axis.imshow(shown, cmap="viridis")
        axis.set_title(title, fontsize=12, fontweight="bold")
        if values.size(1) == 3:
            axis.set_xticks(range(3), ["T1", "T2", "T3"])
        else:
            axis.set_xticks(range(values.size(1)), [f"d{i + 1}" for i in range(values.size(1))])
        axis.set_yticks(range(values.size(0)), ["T1", "T2", "T3"])
        axis.set_xlabel("key token" if values.size(1) == 3 else "feature")
        axis.set_ylabel("query token")
        for row in range(values.size(0)):
            for column in range(values.size(1)):
                value = values[row, column].item()
                label = "-∞" if value == float("-inf") else f"{value:.3f}"
                axis.text(column, row, label, ha="center", va="center", color="white", fontsize=9)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.suptitle("Figure 1 - Hand-calculated scaled dot-product attention (d_k=d_v=2)", fontsize=15)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_causal_mask_figure(mask: torch.Tensor) -> None:
    """Save the standalone causal mask required as Figure 2."""
    figure, axis = plt.subplots(figsize=(6.5, 5.4))
    allowed = (~mask).to(dtype=torch.float32).cpu().numpy()
    image = axis.imshow(allowed, cmap="RdYlGn", vmin=0, vmax=1)
    axis.set_xticks(range(3), ["T1", "T2", "T3"])
    axis.set_yticks(range(3), ["T1", "T2", "T3"])
    axis.set_xlabel("Key token (information source)")
    axis.set_ylabel("Query token (current position)")
    axis.set_title("Figure 2 - Causal mask: green=allowed, red=blocked")
    for row in range(3):
        for column in range(3):
            label = "allowed" if allowed[row, column] else "blocked"
            axis.text(column, row, label, ha="center", va="center", fontsize=10, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["blocked", "allowed"])
    figure.tight_layout()
    figure.savefig(MASK_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    raw_scores = q @ k.T
    scaled_scores = raw_scores / math.sqrt(2)
    weights = torch.softmax(scaled_scores, dim=-1)
    output = weights @ v

    mask = SelfAttention.causal_mask(sequence_length=3, device=q.device)
    masked_scores = scaled_scores.masked_fill(mask, float("-inf"))
    masked_weights = torch.softmax(masked_scores, dim=-1)
    masked_output = masked_weights @ v

    module = make_identity_projection_module(q, k, v)
    one_hot_tokens = torch.eye(3).unsqueeze(0)
    module_output, module_weights = module(one_hot_tokens)
    module_masked_output, module_masked_weights = module(one_hot_tokens, causal=True)
    assert torch.allclose(module_weights[0], weights, atol=1e-6)
    assert torch.allclose(module_output[0, :, :2], output, atol=1e-6)
    assert torch.allclose(module_masked_weights[0], masked_weights, atol=1e-6)
    assert torch.allclose(module_masked_output[0, :, :2], masked_output, atol=1e-6)

    report = "\n\n".join(
        [
            "M1 HAND CALCULATION: SCALED DOT-PRODUCT SELF-ATTENTION",
            matrix_text("Q", q),
            matrix_text("K", k),
            matrix_text("V", v),
            f"Step 3 scaling constant: sqrt(d_k) = sqrt(2) = {math.sqrt(2):.4f}",
            "Step 2: QK^T (each row is one query compared with all keys)\n" + matrix_text("QK^T", raw_scores),
            "Step 3: QK^T / sqrt(d_k), where sqrt(2) = 1.4142\n" + matrix_text("scaled scores", scaled_scores),
            "Step 4: row-wise softmax (each row sums to 1)\n" + matrix_text("weights", weights),
            "Step 5: weights @ V\n" + matrix_text("unmasked output", output),
            "Step 6: set future positions above the diagonal to -inf before softmax\n" + matrix_text("masked scores", masked_scores),
            "Masked softmax weights\n" + matrix_text("masked weights", masked_weights),
            "Masked output\n" + matrix_text("masked output", masked_output),
            "Step 7: token 1 can access only token 1; token 2 can access tokens 1-2; token 3 can access tokens 1-3.",
            "Verification: every unmasked and masked value matches src.attention.SelfAttention (allclose at 1e-6).",
        ]
    )
    print(report)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(report + "\n", encoding="utf-8")
    create_figure(q, k, v, raw_scores, scaled_scores, weights, output, masked_scores, masked_weights, masked_output)
    create_causal_mask_figure(mask)
    print(f"\nSaved terminal evidence: {EVIDENCE_PATH.relative_to(ROOT)}")
    print(f"Saved Figure 1: {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Saved Figure 2: {MASK_FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

