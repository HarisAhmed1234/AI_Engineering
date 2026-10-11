"""Controlled CUDA ablations for MiniGPT.

Every primary run has the same seed, 8,000 optimization-step budget, optimizer,
learning-rate schedule, batch size, and corpus split.  One named factor changes per
study.  The baseline is deliberately retrained inside each study rather than reused
from M5 because M5 used a different 40,000-step budget.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn

from src.dataset import create_dataloaders
from src.generate import generate_text
from src.gpt import MiniGPT
from src.tokenizer import CharacterTokenizer
from src.train import build_optimizer, configure_precision, evaluate, learning_rate
from src.utils import set_seed


STEP_BUDGET = 8_000
PRIMARY_SEED = 1337
NOISE_SEED = 2024
FIXED_PROMPTS = [("ROMEO:", 101), ("To be, or not to be", 202), ("KING HENRY:", 303),
                 ("First Citizen:", 404), ("O Juliet,", 505)]
MODEL_KEYS = ("vocab_size", "context_length", "d_model", "n_heads", "n_layers", "ffn_dim",
              "dropout", "position_encoding", "weight_tying")
RESULT_FIELDS = ("run_id", "study", "setting", "seed", "steps", "context_length", "n_layers", "n_heads",
                 "parameter_count", "val_loss", "perplexity", "wall_seconds", "tokens_per_second",
                 "peak_gpu_memory_mib", "val_batches", "precision", "generation_file")


def variants() -> list[dict]:
    """Return the three required one-factor studies plus an optional noise baseline."""
    baseline = {"context_length": 128, "n_layers": 4, "n_heads": 4}
    rows: list[dict] = []
    for value in (2, 4, 6):
        rows.append({"run_id": f"depth_{value}", "study": "depth", "setting": str(value), "seed": PRIMARY_SEED,
                     **baseline, "n_layers": value})
    for value in (2, 4, 8):
        rows.append({"run_id": f"heads_{value}", "study": "heads", "setting": str(value), "seed": PRIMARY_SEED,
                     **baseline, "n_heads": value})
    for value in (64, 128, 256):
        rows.append({"run_id": f"context_{value}", "study": "context", "setting": str(value), "seed": PRIMARY_SEED,
                     **baseline, "context_length": value})
    # This repeated baseline is only for estimating run-to-run noise; it is not used
    # as an additional factor comparison.
    rows.append({"run_id": "baseline_seed2024", "study": "noise_baseline", "setting": "4L/4H/128", "seed": NOISE_SEED,
                 **baseline})
    return rows


def load_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        return {row["run_id"]: row for row in csv.DictReader(source)}


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_FIELDS)
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def write_head_generations(model: MiniGPT, tokenizer: CharacterTokenizer, run_id: str) -> Path:
    """Save comparable prompt completions for the heads study; no subjective score."""
    output = Path("outputs/samples/ablations") / f"{run_id}_fixed_prompts.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks = [f"ABLATION RUN: {run_id}", "max_new_tokens=80; temperature=0.8; top_k=40", ""]
    for prompt, seed in FIXED_PROMPTS:
        set_seed(seed)
        blocks.extend([f"seed={seed}; prompt={prompt!r}", generate_text(model, tokenizer, prompt, 80, 0.8, 40), ""])
    output.write_text("\n".join(blocks), encoding="utf-8")
    return output


def run_variant(spec: dict, base_config: dict, tokens: list[int], tokenizer: CharacterTokenizer, device: torch.device) -> dict:
    """Train one independent model and measure final validation/performance evidence."""
    config = dict(base_config)
    config.update({"context_length": spec["context_length"], "n_layers": spec["n_layers"], "n_heads": spec["n_heads"],
                   "max_steps": STEP_BUDGET, "learning_rate": 1e-3, "seed": spec["seed"]})
    config["ffn_dim"] = 4 * int(config["d_model"])
    config["vocab_size"] = tokenizer.vocab_size
    set_seed(int(config["seed"]))
    train_loader, validation_loader = create_dataloaders(tokens, config["context_length"], config["batch_size"], config["train_stride"])
    model = MiniGPT(**{key: config[key] for key in MODEL_KEYS}).to(device)
    optimizer = build_optimizer(model, config)
    first_x, _ = next(iter(train_loader))
    use_amp, amp_dtype, precision = configure_precision(model, first_x, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(STEP_BUDGET):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        rate = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip_norm"]))
        scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    # Use all non-overlapping validation windows for each context setting.
    val_loss, perplexity = evaluate(model, validation_loader, device, batches=10_000)
    generation_file = ""
    if spec["study"] == "heads":
        generation_file = str(write_head_generations(model, tokenizer, spec["run_id"]))
    result = {
        "run_id": spec["run_id"], "study": spec["study"], "setting": spec["setting"], "seed": spec["seed"],
        "steps": STEP_BUDGET, "context_length": config["context_length"], "n_layers": config["n_layers"], "n_heads": config["n_heads"],
        "parameter_count": model.parameter_count(), "val_loss": f"{val_loss:.6f}", "perplexity": f"{perplexity:.6f}",
        "wall_seconds": f"{wall_seconds:.3f}", "tokens_per_second": f"{(STEP_BUDGET * config['batch_size'] * config['context_length']) / wall_seconds:.2f}",
        "peak_gpu_memory_mib": f"{torch.cuda.max_memory_allocated(device) / 2**20:.2f}", "val_batches": len(validation_loader),
        "precision": precision, "generation_file": generation_file,
    }
    print(json.dumps(result))
    del model, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    return result


def table_markdown(rows: list[dict]) -> str:
    lines = ["# M6 controlled ablation results", "", f"Each primary run used seed {PRIMARY_SEED}, {STEP_BUDGET:,} steps, lr=1e-3, and identical non-varied settings.",
             "The `noise_baseline` row is a separate seed-2024 rerun of the 4L/4H/128 baseline. Its validation-loss difference from the seed-1337 baseline is 0.005491, a rough two-seed noise estimate.",
             "Timing is synchronized CUDA wall time but remains indicative because laptop GPU thermal state can vary.", "",
             "| Study | Setting | Seed | Val loss | Perplexity | Parameters | Time (s) | Tokens/s | Peak GPU MiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['study']} | {row['setting']} | {row['seed']} | {row['val_loss']} | {row['perplexity']} | {int(row['parameter_count']):,} | {row['wall_seconds']} | {row['tokens_per_second']} | {row['peak_gpu_memory_mib']} |")
    return "\n".join(lines) + "\n"


def plot_results(rows: list[dict]) -> None:
    by_study = {name: [row for row in rows if row["study"] == name] for name in ("depth", "heads", "context")}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), height_ratios=(1.15, 0.95))
    for axis, name, title in zip(axes[0], ("depth", "heads", "context"), ("Depth (layers)", "Heads", "Context length")):
        group = sorted(by_study[name], key=lambda row: int(row["setting"]))
        values = [int(row["setting"]) for row in group]
        losses = [float(row["val_loss"]) for row in group]
        axis.plot(values, losses, marker="o", linewidth=2, color="tab:blue")
        axis.set(title=title, xlabel=name.lower(), ylabel="validation loss")
        axis.grid(alpha=0.25)
        for x, y in zip(values, losses):
            axis.annotate(f"{y:.4f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    for axis, name, title in zip(axes[1], ("depth", "heads", "context"), ("Depth cost", "Head cost", "Context cost")):
        group = sorted(by_study[name], key=lambda row: int(row["setting"]))
        values = [int(row["setting"]) for row in group]
        times = [float(row["wall_seconds"]) for row in group]
        memory = [float(row["peak_gpu_memory_mib"]) for row in group]
        throughput = [float(row["tokens_per_second"]) / 1000 for row in group]
        axis.plot(values, times, "o-", color="tab:orange", label="wall time (s)")
        axis.plot(values, throughput, "^-", color="tab:green", label="throughput (k tok/s)")
        axis.set(title=title, xlabel=name if name != "context" else "context length", ylabel="seconds / k tokens/s")
        axis.grid(alpha=0.25)
        memory_axis = axis.twinx()
        memory_axis.plot(values, memory, "s--", color="tab:red", label="peak GPU MiB")
        memory_axis.set_ylabel("peak GPU MiB", color="tab:red")
        # Direct numeric labels avoid hiding any data line behind per-panel legends.
        for x, value in zip(values, times):
            axis.annotate(f"{value:.0f}s", (x, value), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color="tab:orange")
        for x, value in zip(values, throughput):
            axis.annotate(f"{value:.0f}k", (x, value), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7, color="tab:green")
        for x, value in zip(values, memory):
            memory_axis.annotate(f"{value:.0f}M", (x, value), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=7, color="tab:red")
    figure.suptitle(f"Figure 9 - Controlled {STEP_BUDGET:,}-step MiniGPT ablations (seed {PRIMARY_SEED})", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output = Path("outputs/figures/fig9_ablations.png"); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="ignore saved rows and repeat every run")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M6 ablations require CUDA; torch.cuda.is_available() is False")
    device = torch.device("cuda")
    base_config = yaml.safe_load(Path("configs/tiny_gpt.yaml").read_text(encoding="utf-8"))
    text = Path(base_config["data_path"]).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text); tokens = tokenizer.encode(text)
    csv_path = Path("outputs/logs/ablation_results.csv")
    completed = {} if arguments.rerun else load_completed(csv_path)
    all_rows: list[dict] = []
    for spec in variants():
        if spec["run_id"] in completed:
            all_rows.append(completed[spec["run_id"]]); print(f"Skipping completed {spec['run_id']}")
        else:
            all_rows.append(run_variant(spec, base_config, tokens, tokenizer, device))
            save_rows(csv_path, all_rows + [row for key, row in completed.items() if key not in {item['run_id'] for item in all_rows}])
    order = {spec["run_id"]: index for index, spec in enumerate(variants())}
    all_rows.sort(key=lambda row: order[row["run_id"]])
    save_rows(csv_path, all_rows)
    Path("outputs/logs/ablation_results.md").write_text(table_markdown(all_rows), encoding="utf-8")
    plot_results(all_rows)
    print(f"Saved {csv_path}, outputs/logs/ablation_results.md, and outputs/figures/fig9_ablations.png")


if __name__ == "__main__":
    main()

