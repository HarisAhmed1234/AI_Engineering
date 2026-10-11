"""Compare bidirectional MLM attention with causal next-token attention for M3.

The small experiment is intentionally synthetic and transparent. The target at
position ``t`` is the input token at ``t+1``. A bidirectional attention row may
inspect that future input, so a direct-copy shortcut exists. A causal row cannot.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig_m3_mlm_vs_clm.png"
LOG_PATH = ROOT / "outputs" / "logs" / "m3_mlm_vs_clm.txt"
CSV_PATH = ROOT / "outputs" / "logs" / "m3_mlm_vs_clm_losses.csv"


class TinyLeakModel(nn.Module):
    """A single attention layer plus classifier; mask choice is the only change."""

    def __init__(self, vocab_size: int, d_model: int, causal: bool) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(8, d_model)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size)
        self.causal = causal

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.embedding(ids) + self.position(positions).unsqueeze(0)
        scores = self.query(x) @ self.key(x).transpose(-2, -1) / (x.size(-1) ** 0.5)
        if self.causal:
            mask = torch.triu(torch.ones(ids.size(1), ids.size(1), device=ids.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.head(weights @ self.value(x))


def train_condition(causal: bool, inputs: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Train with identical initialization/data except for the attention mask."""
    torch.manual_seed(101)
    model = TinyLeakModel(vocab_size=16, d_model=16, causal=causal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    losses = []
    for _ in range(150):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    torch.manual_seed(41)
    vocab_size, batch, time = 16, 64, 8
    inputs = torch.randint(0, vocab_size, (batch, time))
    shifted_targets = torch.cat([inputs[:, 1:], inputs[:, :1]], dim=1)
    # In a bidirectional layer, target[t] is exactly the visible input at key t+1.
    bidirectional_losses = train_condition(False, inputs, shifted_targets)
    causal_next_token_losses = train_condition(True, inputs, shifted_targets)

    # Deterministic direct-copy proof: a hypothetical classifier that reads key
    # t+1 can put all of its logit mass on the visible target. A causal row cannot
    # use that future key and its no-information baseline is uniform.
    high_logit = 30.0
    leaked_logits = torch.full((batch, time, vocab_size), -high_logit)
    leaked_logits.scatter_(-1, shifted_targets.unsqueeze(-1), high_logit)
    direct_copy_loss = float(nn.functional.cross_entropy(leaked_logits.reshape(-1, vocab_size), shifted_targets.reshape(-1)))
    uniform_logits = torch.zeros_like(leaked_logits)
    causal_uniform_loss = float(nn.functional.cross_entropy(uniform_logits.reshape(-1, vocab_size), shifted_targets.reshape(-1)))

    masked_positions = torch.tensor([1, 5])
    bidirectional_mask = torch.ones(time, time)
    causal_mask = torch.tril(torch.ones(time, time))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for axis, matrix, title, cmap in [
        (axes[0], bidirectional_mask, "MLM: bidirectional attention\n[MASK] positions: 1, 5", "Greens"),
        (axes[1], causal_mask, "GPT: causal lower-triangular attention", "RdYlGn"),
    ]:
        image = axis.imshow(matrix.numpy(), vmin=0, vmax=1, cmap=cmap)
        axis.set_xlabel("Key position"); axis.set_ylabel("Query position"); axis.set_title(title)
        if axis is axes[0]:
            input_tokens = ["A", "[MASK]", "C", "D", "E", "[MASK]", "G", "H"]
            axis.set_xticks(range(time), input_tokens, rotation=0)
            axis.set_xlabel("Input token at key position ([MASK] is an MLM training input)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[2].plot(bidirectional_losses, label="Bidirectional next-token target (future visible)", linewidth=2)
    axes[2].plot(causal_next_token_losses, label="Causal next-token target", linewidth=2)
    axes[2].set_title("Real 150-step tiny-model losses")
    axes[2].set_xlabel("Training step"); axes[2].set_ylabel("Cross-entropy loss")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    figure.suptitle("M3 - MLM versus CLM masks and information leakage", fontsize=15)
    figure.tight_layout(); FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight"); plt.close(figure)

    rows = []
    for name, losses in [("bidirectional_next_token_future_visible", bidirectional_losses), ("causal_next_token_target", causal_next_token_losses)]:
        rows.extend({"condition": name, "step": step + 1, "loss": loss} for step, loss in enumerate(losses))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "step", "loss"]); writer.writeheader(); writer.writerows(rows)
    log = (
        "M3 MLM VS CLM EXPERIMENT\n"
        f"Deterministic bidirectional direct-copy loss (target token visible at key t+1): {direct_copy_loss:.4f}\n"
        f"Causal uniform no-future-information baseline loss: {causal_uniform_loss:.4f}\n"
        f"Trained bidirectional next-token loss (future visible): initial={bidirectional_losses[0]:.4f}, final={bidirectional_losses[-1]:.4f}\n"
        f"Causal next-token loss: initial={causal_next_token_losses[0]:.4f}, final={causal_next_token_losses[-1]:.4f}\n"
        "Interpretation: with a bidirectional mask, query t can attend to key t+1, which contains its next-token label. "
        "A causal next-token LM cannot access key t+1, so this direct-copy shortcut is unavailable.\n"
    )
    LOG_PATH.write_text(log, encoding="utf-8")
    print(log)
    print(f"Saved figure: {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Saved log: {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

