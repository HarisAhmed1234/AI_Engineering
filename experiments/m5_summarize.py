"""Export the compact M5 checkpoint and summarize measured full-run evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.generate import load_model_checkpoint


def main() -> None:
    log_path = Path("outputs/logs/training_log.csv")
    with log_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError("training log is empty")
    final = rows[-1]
    best = min(rows, key=lambda row: float(row["val_loss"]))
    source_checkpoint = Path("checkpoints/best.pt")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    # This deployment-friendly checkpoint deliberately contains only inference
    # weights plus the architecture configuration and tokenizer vocabulary.
    # In particular, it carries no optimizer state, step counter, or AMP state.
    compact = {key: checkpoint[key] for key in ("model_state", "config", "vocabulary")}
    compact_path = Path("checkpoints/minigpt_final.pt")
    torch.save(compact, compact_path)
    model, _, config, _ = load_model_checkpoint(compact_path, torch.device("cpu"))
    cuda_available = torch.cuda.is_available()
    summary = {
        "run": "M5 full Tiny Shakespeare training",
        "final_parameter_count": model.parameter_count(),
        "training_config": config,
        "hardware": {
            "device": torch.cuda.get_device_name(0) if cuda_available else "CPU",
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda if cuda_available else None,
            "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda_available else None,
        },
        "precision": final["precision"],
        "wall_clock_seconds": float(final["elapsed_seconds"]),
        "final_step": int(final["step"]),
        "final_average_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "final_perplexity": float(final["perplexity"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_step": int(best["step"]),
        "compact_checkpoint": str(compact_path),
        "compact_checkpoint_source": str(source_checkpoint),
        "compact_checkpoint_mb": round(compact_path.stat().st_size / 2**20, 3),
        "measurement_note": "Training loss is the average of all optimization-step losses since the previous evaluation.",
    }
    json_path = Path("outputs/logs/m5_training_summary.json")
    text_path = Path("outputs/logs/m5_training_summary.txt")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path.write_text("M5 TRAINING SUMMARY\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
