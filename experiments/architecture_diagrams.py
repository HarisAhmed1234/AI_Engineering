"""Create clean M3 architecture diagrams from the implemented design."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
BLOCK_PATH = ROOT / "outputs" / "figures" / "fig_m3_transformer_block.png"
MODEL_PATH = ROOT / "outputs" / "figures" / "fig_m3_minigpt_architecture.png"


def draw_box(axis: plt.Axes, y: float, text: str, color: str = "#dbeafe") -> None:
    box = FancyBboxPatch((0.17, y), 0.66, 0.08, boxstyle="round,pad=0.012", facecolor=color, edgecolor="#1e3a8a", linewidth=1.4)
    axis.add_patch(box)
    axis.text(0.5, y + 0.04, text, ha="center", va="center", fontsize=11, fontweight="bold")


def arrow(axis: plt.Axes, start: float, end: float, label: str = "") -> None:
    axis.annotate("", xy=(0.5, end), xytext=(0.5, start), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    if label:
        axis.text(0.54, (start + end) / 2, label, fontsize=9, color="#334155")


def block_diagram() -> None:
    figure, axis = plt.subplots(figsize=(8, 10))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    entries = [
        (0.89, "Input x  (B, T, D)", "#e2e8f0"),
        (0.76, "LayerNorm (pre-LN)", "#fef3c7"),
        (0.63, "Causal Multi-Head Attention", "#dbeafe"),
        (0.49, "Residual add: x + attention", "#dcfce7"),
        (0.35, "LayerNorm (pre-LN)", "#fef3c7"),
        (0.22, "FFN: Linear -> GELU -> Linear -> Dropout", "#dbeafe"),
        (0.08, "Residual add: x + FFN  -> output (B, T, D)", "#dcfce7"),
    ]
    for y, text, color in entries:
        draw_box(axis, y, text, color)
    for upper, lower in zip(entries[:-1], entries[1:]):
        arrow(axis, upper[0], lower[0] + 0.08)
    axis.set_title("Transformer decoder block - pre-LayerNorm", fontsize=16, fontweight="bold")
    axis.text(0.5, 0.01, "Causal attention blocks future tokens; residual paths preserve an identity route for gradients.", ha="center", fontsize=9)
    figure.tight_layout(); BLOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(BLOCK_PATH, dpi=180, bbox_inches="tight"); plt.close(figure)


def model_diagram() -> None:
    figure, axis = plt.subplots(figsize=(9, 11))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    entries = [
        (0.90, "Token IDs  (B, T)", "#e2e8f0"),
        (0.78, "Token embedding  (B, T, D)", "#dbeafe"),
        (0.66, "+ sinusoidal positional encoding  (B, T, D)", "#fef3c7"),
        (0.52, "N x pre-LN causal Transformer blocks  (B, T, D)", "#c7d2fe"),
        (0.38, "Final LayerNorm  (B, T, D)", "#fef3c7"),
        (0.25, "LM head: Linear(D, vocab_size)", "#dbeafe"),
        (0.11, "Vocabulary logits  (B, T, V)", "#dcfce7"),
    ]
    for y, text, color in entries:
        draw_box(axis, y, text, color)
    for upper, lower in zip(entries[:-1], entries[1:]):
        arrow(axis, upper[0], lower[0] + 0.08)
    axis.set_title("MiniGPT architecture (implemented across M2-M4)", fontsize=16, fontweight="bold")
    axis.text(0.5, 0.01, "The M3 block exists now; token embedding, final norm, and LM head are added in M4.", ha="center", fontsize=9)
    figure.tight_layout(); MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(MODEL_PATH, dpi=180, bbox_inches="tight"); plt.close(figure)


if __name__ == "__main__":
    block_diagram(); model_diagram()
    print(f"Saved: {BLOCK_PATH.relative_to(ROOT)}")
    print(f"Saved: {MODEL_PATH.relative_to(ROOT)}")

