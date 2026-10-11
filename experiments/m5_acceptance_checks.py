"""M5 acceptance evidence: fresh load, three samples, and trained-model causality."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.generate import generate_text, load_model_checkpoint
from src.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/logs/m5_acceptance_checks.txt")
    arguments = parser.parse_args()
    evidence: list[str] = []

    def record(message: str) -> None:
        evidence.append(message)
        print(message)

    model, tokenizer, config, device = load_model_checkpoint("checkpoints/minigpt_final.pt")
    record(f"FRESH LOAD OK: device={device}; parameters={model.parameter_count():,}; context={config['context_length']}")
    for number, prompt in enumerate(("ROMEO:", "KING HENRY:", "O Juliet,"), start=1):
        set_seed(900 + number)
        text = generate_text(model, tokenizer, prompt, max_new_tokens=32, temperature=0.8, top_k=40)
        record(f"SAMPLE {number} prompt={prompt!r}: {text!r}")
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=next(model.parameters()).device)
    changed = ids.clone(); changed[:, 5:] = torch.tensor([[12, 13, 14]], device=ids.device)
    with torch.no_grad():
        original_logits = model(ids)
        changed_logits = model(changed)
    maximum_difference = float((original_logits[:, :5] - changed_logits[:, :5]).abs().max())
    record(f"TRAINED CAUSALITY: max |earlier-logit difference| after future perturbation = {maximum_difference:.8f}")
    if maximum_difference != 0.0:
        raise AssertionError("future tokens changed earlier logits")
    record("M5 ACCEPTANCE CHECKS PASSED")
    output_path = Path(arguments.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
