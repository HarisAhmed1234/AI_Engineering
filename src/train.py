"""Train MiniGPT with warmup, cosine decay, checkpoints, logs, and samples.

Run `python -m src.train --config configs/tiny_gpt.yaml`. M4 uses this command only
for tiny checks; M5 performs the documented full run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from src.dataset import create_dataloaders
from src.gpt import MiniGPT, save_model_summary_figure
from src.tokenizer import CharacterTokenizer
from src.utils import set_seed


def learning_rate(step: int, config: dict) -> float:
    """Linear warmup followed by cosine decay to 10% of the base learning rate."""
    base = float(config["learning_rate"])
    warmup = int(config["warmup_steps"])
    maximum = int(config["max_steps"])
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: MiniGPT, loader, device: torch.device, batches: int) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        logits = model(x.to(device))
        loss = nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.to(device).reshape(-1))
        losses.append(float(loss))
    average = sum(losses) / max(1, len(losses))
    return average, math.exp(average)


@torch.no_grad()
def sample_text(model: MiniGPT, tokenizer: CharacterTokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    """Greedy generation for training evidence; full sampler is added in M5."""
    model.eval()
    ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
        next_id = int(torch.argmax(model(context)[0, -1]))
        ids.append(next_id)
    return tokenizer.decode(ids)


def configure_precision(model: MiniGPT, batch: Tensor, config: dict, device: torch.device) -> tuple[bool, torch.dtype | None, str]:
    """Use fp16 only after a finite CUDA autocast forward/backward stability probe."""
    requested = str(config.get("mixed_precision", "auto"))
    if requested.lower() == "false" or device.type != "cuda":
        return False, None, "float32 (disabled or CUDA unavailable)"
    try:
        probe_ids = batch[: min(2, len(batch))].to(device)
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(probe_ids)
            probe_loss = logits.float().square().mean()
        probe_loss.backward()
        stable = bool(torch.isfinite(probe_loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
        )
        model.zero_grad(set_to_none=True)
        if stable:
            return True, torch.float16, "float16 CUDA AMP (finite forward/backward stability probe passed)"
    except RuntimeError as error:
        model.zero_grad(set_to_none=True)
        return False, None, f"float32 (AMP probe failed: {error})"
    return False, None, "float32 (AMP probe produced non-finite values)"


def save_checkpoint(path: Path, model: MiniGPT, optimizer, step: int, config: dict, tokenizer: CharacterTokenizer, precision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
         "config": config, "vocabulary": tokenizer.characters, "precision": precision},
        path,
    )


def build_optimizer(model: MiniGPT, config: dict) -> torch.optim.AdamW:
    """Apply weight decay only to matrix weights of non-embedding layers.

    Biases and LayerNorm scale/shift parameters are small normalization/offset terms
    for which weight decay is usually unhelpful. Embeddings are token lookup vectors,
    so excluding them preserves their learned symbol representations.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "embedding" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(config["weight_decay"])}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(config["learning_rate"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_gpt.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="write an isolated tuning run beneath outputs/logs and checkpoints")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if arguments.max_steps is not None:
        config["max_steps"] = arguments.max_steps
    if arguments.learning_rate is not None:
        config["learning_rate"] = arguments.learning_rate
    if arguments.eval_interval is not None:
        config["eval_interval"] = arguments.eval_interval
    if arguments.run_name:
        config["log_dir"] = str(Path(config["log_dir"]) / arguments.run_name)
        config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]) / arguments.run_name)
        config["sample_dir"] = str(Path(config["sample_dir"]) / arguments.run_name)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU.")
    text = Path(config["data_path"]).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    tokens = tokenizer.encode(text)
    config["vocab_size"] = tokenizer.vocab_size
    train_loader, val_loader = create_dataloaders(tokens, config["context_length"], config["batch_size"], config["train_stride"])
    model = MiniGPT(**{key: config[key] for key in ("vocab_size", "context_length", "d_model", "n_heads", "n_layers", "ffn_dim", "dropout", "position_encoding", "weight_tying")}).to(device)
    optimizer = build_optimizer(model, config)
    start_step = 0
    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"]) + 1
        print(f"Resumed from {arguments.resume} at step {start_step}")
    first_x, _ = next(iter(train_loader))
    use_amp, amp_dtype, precision = configure_precision(model, first_x, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device={device}; precision={precision}; parameters={model.parameter_count():,}")
    save_model_summary_figure(model)
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["sample_dir"]).mkdir(parents=True, exist_ok=True)
    log_path = Path(config["log_dir"]) / "training_log.csv"
    config_path = Path(config["log_dir"]) / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint_dir = Path(config["checkpoint_dir"])
    best_val = float("inf"); started = time.perf_counter(); train_iterator = iter(train_loader); train_losses_since_eval: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with log_path.open("a" if start_step else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "train_loss", "val_loss", "perplexity", "lr", "elapsed_seconds", "tokens_per_second", "gpu_memory_mib", "precision"])
        if not start_step: writer.writeheader()
        for step in range(start_step, int(config["max_steps"])):
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader); x, y = next(train_iterator)
            x, y = x.to(device), y.to(device)
            lr = learning_rate(step, config)
            for group in optimizer.param_groups: group["lr"] = lr
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip_norm"]))
            scaler.step(optimizer); scaler.update()
            train_losses_since_eval.append(float(loss.detach()))
            if (step + 1) % int(config["eval_interval"]) == 0 or step == int(config["max_steps"]) - 1:
                val_loss, perplexity = evaluate(model, val_loader, device, int(config["eval_batches"]))
                elapsed = time.perf_counter() - started
                token_count = (step - start_step + 1) * x.numel()
                memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
                average_train_loss = sum(train_losses_since_eval) / len(train_losses_since_eval)
                row = {"step": step + 1, "epoch": (step + 1) / max(1, len(train_loader)), "train_loss": f"{average_train_loss:.6f}", "val_loss": f"{val_loss:.6f}", "perplexity": f"{perplexity:.6f}", "lr": f"{lr:.8f}", "elapsed_seconds": f"{elapsed:.3f}", "tokens_per_second": f"{token_count / elapsed:.2f}", "gpu_memory_mib": f"{memory:.2f}", "precision": precision}
                writer.writerow(row); handle.flush(); print(row)
                train_losses_since_eval.clear()
                generated = sample_text(model, tokenizer, "ROMEO:", max_new_tokens=80)
                (Path(config["sample_dir"]) / f"step_{step + 1}.txt").write_text(generated, encoding="utf-8")
                save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, config, tokenizer, precision)
                if val_loss < best_val:
                    best_val = val_loss; save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, step, config, tokenizer, precision)


if __name__ == "__main__":
    main()

