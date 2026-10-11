"""Fresh-process validation and loss-curve evidence for the M5 trained model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.dataset import create_dataloaders
from src.generate import load_model_checkpoint


@torch.no_grad()
def validation_metrics(checkpoint: str, data_path: str, batches: int) -> dict:
    model, tokenizer, config, device_name = load_model_checkpoint(checkpoint)
    device = next(model.parameters()).device
    text = Path(data_path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)  # The checkpoint vocabulary is authoritative.
    _, validation_loader = create_dataloaders(tokens, config["context_length"], config["batch_size"], config.get("train_stride", 32))
    losses: list[float] = []
    for index, (x, y) in enumerate(validation_loader):
        if index >= batches:
            break
        logits = model(x.to(device))
        losses.append(float(nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.to(device).reshape(-1))))
    val_loss = sum(losses) / len(losses)
    return {"checkpoint": checkpoint, "device": device_name, "batches": len(losses), "val_loss": val_loss,
            "perplexity": math.exp(val_loss), "parameters": model.parameter_count()}


def plot_loss_curve(log_path: Path, output_path: Path) -> None:
    with log_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    steps = [int(row["step"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["val_loss"]) for row in rows]
    best_index = min(range(len(validation)), key=validation.__getitem__)
    best_step, best_loss = steps[best_index], validation[best_index]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(steps, train, marker="o", linewidth=1.8, label="average training loss since previous evaluation")
    axis.plot(steps, validation, marker="s", linewidth=1.8, label="validation loss")
    axis.axvline(best_step, color="tab:green", linestyle="--", linewidth=1.3,
                label=f"best validation: step {best_step} ({best_loss:.6f})")
    axis.scatter([best_step], [best_loss], color="tab:green", zorder=3)
    axis.set(title="Figure 6 - MiniGPT full-training loss curve", xlabel="training step", ylabel="cross-entropy loss")
    axis.grid(alpha=0.28); axis.legend(); figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--log", default="outputs/logs/training_log.csv")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", default="outputs/logs/m5_fresh_evaluation.json")
    arguments = parser.parse_args()
    metrics = validation_metrics(arguments.checkpoint, arguments.data, arguments.batches)
    plot_loss_curve(Path(arguments.log), Path("outputs/figures/fig6_loss_curve.png"))
    Path(arguments.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved outputs/figures/fig6_loss_curve.png (best-validation step marked)")


if __name__ == "__main__":
    main()

