"""Make reproducible fixed-prompt M5 text evidence and Figure 8."""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt

from src.generate import generate_text, load_model_checkpoint
from src.utils import set_seed


SAMPLES = [("ROMEO:", 101), ("To be, or not to be", 202), ("KING HENRY:", 303),
           ("First Citizen:", 404), ("O Juliet,", 505)]


def preserve_line_breaks(text: str, width: int = 92) -> str:
    """Wrap long source lines without collapsing the model's newline characters."""
    wrapped_lines: list[str] = []
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    return "\n".join(wrapped_lines)


def trim_trailing_word(prompt: str, generated: str) -> str:
    """Keep the 200-token request but omit a terminal fragment in display evidence.

    Character models may stop in the middle of a word at an arbitrary token budget.
    Removing only the final alphabetic fragment makes the saved sample readable while
    remaining at or below the requested `max_new_tokens=200`.
    """
    completion = generated[len(prompt):]
    return prompt + re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?$", "", completion)


def main() -> None:
    checkpoint = "checkpoints/minigpt_final.pt"
    model, tokenizer, config, device = load_model_checkpoint(checkpoint)
    output_dir = Path("outputs/samples"); output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for index, (prompt, seed) in enumerate(SAMPLES, start=1):
        set_seed(seed)
        generated = trim_trailing_word(prompt, generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40))
        evidence = "\n".join([
            f"M5 fixed sample {index}", f"Checkpoint: {checkpoint}", f"Device: {device}",
            f"Seed: {seed}", "Max new tokens: 200 (terminal word fragment omitted for readability)",
            "Temperature: 0.8", "Top-k: 40", "Top-p: None", "Greedy: False",
            f"Context length: {config['context_length']}",
            f"Model parameters: {model.parameter_count():,}", f"Prompt: {prompt}", "Generated:", generated, "",
        ])
        (output_dir / f"m5_sample_{index}.txt").write_text(evidence, encoding="utf-8")
        rendered.append(f"[{index}] seed={seed} | prompt={prompt}\n" + preserve_line_breaks(generated))
    line_count = sum(item.count("\n") + 1 for item in rendered) + 3
    figure_height = max(6.5, 0.185 * line_count)
    figure, axis = plt.subplots(figsize=(14, figure_height)); axis.axis("off")
    axis.set_title("Figure 8 - Five reproducible MiniGPT samples (200 new tokens; T=0.8, top-k=40)", fontsize=15, fontweight="bold", pad=16)
    header = (f"checkpoint={checkpoint} | device={device} | parameters={model.parameter_count():,} | "
              f"context={config['context_length']} | full settings in outputs/samples/m5_sample_*.txt")
    axis.text(0.02, 0.95, header, va="top", family="monospace", fontsize=7.5)
    axis.text(0.02, 0.86, "\n\n".join(rendered), va="top", family="monospace", fontsize=8.2, linespacing=1.20)
    figure.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.04)
    figure.savefig("outputs/figures/fig8_samples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Saved five outputs/samples/m5_sample_*.txt files and outputs/figures/fig8_samples.png")


if __name__ == "__main__":
    main()
