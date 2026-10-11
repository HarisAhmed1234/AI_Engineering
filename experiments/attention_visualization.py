"""Inspect actual causal attention from the selected trained MiniGPT checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.generate import load_model_checkpoint


PROMPT = "ROMEO:\nMy lord, I can"


def visible_character(character: str) -> str:
    """Keep whitespace visible in compact axis labels."""
    return {"\n": "NL", " ": "SP", "\t": "TAB"}.get(character, character)


def main() -> None:
    model, tokenizer, config, device = load_model_checkpoint("checkpoints/minigpt_final.pt")
    model.eval()  # Essential: dropout would otherwise alter displayed weights.
    token_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        _, all_weights = model(token_ids, return_attention=True)
    maximum_future_weight = max(float(weights[0].triu(diagonal=1).abs().max()) for weights in all_weights)
    maximum_row_sum_error = max(float((weights.sum(dim=-1) - 1.0).abs().max()) for weights in all_weights)
    if maximum_future_weight != 0.0:
        raise AssertionError(f"causal triangle failed: maximum future weight {maximum_future_weight}")
    if maximum_row_sum_error > 1e-6:
        raise AssertionError(f"attention rows are not normalized: maximum error {maximum_row_sum_error}")
    labels = [f"{index}:{visible_character(character)}" for index, character in enumerate(PROMPT)]
    layers, heads = len(all_weights), all_weights[0].shape[1]
    figure, axes = plt.subplots(layers, heads, figsize=(4.2 * heads + 1, 3.25 * layers + 1), squeeze=False)
    image = None
    for layer_index, weights in enumerate(all_weights):
        for head_index in range(heads):
            axis = axes[layer_index][head_index]
            image = axis.imshow(weights[0, head_index].detach().cpu(), cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
            axis.set_title(f"Layer {layer_index + 1}, head {head_index + 1}", fontsize=10)
            axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            axis.set_yticks(range(len(labels)), labels, fontsize=6)
            if layer_index == layers - 1:
                axis.set_xlabel("key / attended token", fontsize=8)
            if head_index == 0:
                axis.set_ylabel("query token", fontsize=8)
    figure.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.88, hspace=0.43, wspace=0.24)
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="attention weight")
    figure.suptitle("Figure 7 - Real causal attention from minigpt_final.pt\n"
                     f"prompt={PROMPT!r}; eval mode; future max={maximum_future_weight:.1f}; row-sum error={maximum_row_sum_error:.1e}",
                     fontsize=14, fontweight="bold")
    output_path = Path("outputs/figures/fig7_attention.png"); output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180); plt.close(figure)
    evidence = {"checkpoint": "checkpoints/minigpt_final.pt", "device": device, "prompt": PROMPT,
                "layers": layers, "heads": heads, "max_forbidden_future_weight": maximum_future_weight,
                "max_attention_row_sum_error": maximum_row_sum_error,
                "eval_mode": not model.training, "context_length": config["context_length"]}
    Path("outputs/logs/m6_attention_visualization.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
