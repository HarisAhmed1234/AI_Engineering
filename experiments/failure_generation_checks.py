"""Inference-only long-context and repetition evidence for report failure analysis."""

from __future__ import annotations

import json
from pathlib import Path

from src.generate import generate_text, load_model_checkpoint
from src.utils import set_seed


def run(prompt: str, seed: int, new_tokens: int, temperature: float, greedy: bool) -> dict:
    model, tokenizer, _, _ = load_model_checkpoint("checkpoints/minigpt_final.pt")
    set_seed(seed)
    text = generate_text(model, tokenizer, prompt, new_tokens, temperature, 40, None, greedy)
    return {"prompt": prompt, "seed": seed, "max_new_tokens": new_tokens, "temperature": temperature,
            "top_k": 40, "greedy": greedy, "generated": text}


def main() -> None:
    # MONTAGUE is distinctive, present in the character vocabulary, and deliberately
    # placed early so it has fallen outside a 128-character generation window later.
    records = [run("MONTAGUE:\nRemember MONTAGUE, my lord.\n", 111, 400, 0.8, False)]
    for prompt, seed in (("ROMEO:", 301), ("KING HENRY:", 302)):
        records.append(run(prompt, seed, 300, 1.0, True))
        records.append(run(prompt, seed, 300, 0.2, False))
    payload = {"checkpoint": "checkpoints/minigpt_final.pt", "records": records}
    json_path = Path("outputs/logs/m6_failure_generation_checks.json")
    text_path = Path("outputs/logs/m6_failure_generation_checks.txt")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    blocks = []
    for index, record in enumerate(records, start=1):
        blocks.extend([f"RUN {index}", *(f"{key}: {value}" for key, value in record.items() if key != "generated"), "Generated:", record["generated"], ""])
    text_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Saved {json_path} and {text_path}")


if __name__ == "__main__":
    main()
