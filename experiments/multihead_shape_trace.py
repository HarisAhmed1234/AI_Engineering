"""Create the complete M2 multi-head tensor-shape trace (Figure 3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.attention import MultiHeadSelfAttention


ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "outputs" / "logs" / "m2_multihead_shape_trace.txt"
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig3_multihead_shape_trace.png"


def main() -> None:
    torch.manual_seed(17)
    x = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]]
    )
    attention = MultiHeadSelfAttention(d_model=4, n_heads=2, dropout=0.0).eval()
    trace, details = attention.shape_trace(x, causal=True)
    print(trace)
    print("\nFinal output (batch 0):\n", details["output"][0])

    TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_trace = trace + "\n\nFinal output (batch 0):\n" + str(details["output"][0]) + "\n"
    TEXT_PATH.write_text(full_trace, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(15, 15))
    axis.axis("off")
    axis.text(0.01, 0.99, full_trace, va="top", family="monospace", fontsize=9)
    axis.set_title("Figure 3 - Complete multi-head attention tensor-shape trace", fontsize=15, fontweight="bold")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved terminal evidence: {TEXT_PATH.relative_to(ROOT)}")
    print(f"Saved Figure 3: {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

