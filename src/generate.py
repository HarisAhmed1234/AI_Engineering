"""Autoregressive MiniGPT inference from a self-contained project checkpoint.

Example:
    python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 \
        --temperature 0.8 --top-k 40 --seed 1337 --checkpoint checkpoints/minigpt_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.gpt import MiniGPT
from src.tokenizer import CharacterTokenizer
from src.utils import set_seed


MODEL_CONFIG_KEYS = (
    "vocab_size", "context_length", "d_model", "n_heads", "n_layers", "ffn_dim",
    "dropout", "position_encoding", "weight_tying",
)


def load_model_checkpoint(checkpoint_path: str | Path, device: torch.device | None = None) -> tuple[MiniGPT, CharacterTokenizer, dict, str]:
    """Load a full-training or small weights-only checkpoint in a fresh process."""
    chosen_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=chosen_device, weights_only=False)
    config = dict(checkpoint["config"])
    config["vocab_size"] = len(checkpoint["vocabulary"])
    model = MiniGPT(**{key: config[key] for key in MODEL_CONFIG_KEYS}).to(chosen_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # Returned attention is not post-dropout during later inspection.
    return model, CharacterTokenizer(checkpoint["vocabulary"]), config, str(chosen_device)


def _sample_token(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None, greedy: bool) -> int:
    """Sample one token after temperature and optional top-k/top-p filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if greedy:
        return int(torch.argmax(logits))
    filtered = logits / temperature
    if top_k is not None and top_k > 0:
        keep = min(top_k, filtered.numel())
        cutoff = torch.topk(filtered, keep).values[-1]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[0] = False  # Always retain one legal candidate.
        filtered = filtered.clone()
        filtered[sorted_indices[remove_sorted]] = float("-inf")
    return int(torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1))


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    greedy: bool = False,
) -> str:
    """Generate left-to-right, cropping the input context to the model limit."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    token_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    model.eval()
    for _ in range(max_new_tokens):
        context = torch.tensor([token_ids[-model.context_length :]], dtype=torch.long, device=device)
        logits = model(context)[0, -1]
        token_ids.append(_sample_token(logits, temperature, top_k, top_p, greedy))
    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint", default="checkpoints/minigpt_final.pt")
    arguments = parser.parse_args()
    set_seed(arguments.seed)
    model, tokenizer, config, device = load_model_checkpoint(arguments.checkpoint)
    generated = generate_text(model, tokenizer, arguments.prompt, arguments.max_new_tokens, arguments.temperature,
                              arguments.top_k, arguments.top_p, arguments.greedy)
    print(f"Checkpoint: {arguments.checkpoint}")
    print(f"Device: {device}")
    print(f"Seed: {arguments.seed}")
    print(f"Model parameters: {model.parameter_count():,}")
    print(f"Context length: {config['context_length']}")
    print(f"Temperature: {arguments.temperature}")
    print(f"Top-k: {arguments.top_k}; Top-p: {arguments.top_p}; Greedy: {arguments.greedy}")
    print(f"Prompt: {arguments.prompt}")
    print("Generated:")
    print(generated)


if __name__ == "__main__":
    main()

