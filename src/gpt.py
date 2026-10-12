"""Small GPT-style causal language model assembled from project components."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """Embedding -> position -> N causal decoder blocks -> norm -> vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        position_encoding: str = "sinusoidal",
        weight_tying: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # We deliberately do not multiply embeddings by sqrt(D): GPT-style learned
        # token embeddings are initialized directly at the model's working scale.
        if position_encoding == "sinusoidal":
            self.position = SinusoidalPositionalEncoding(d_model, context_length)
        elif position_encoding == "learned":
            self.position = LearnedPositionalEmbedding(d_model, context_length)
        else:
            raise ValueError("position_encoding must be 'sinusoidal' or 'learned'")
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.weight_tying = weight_tying
        self.apply(self._initialize_weights)
        # Residual projections sum contributions from 2*n_layers branches. Scaling
        # their initial standard deviation keeps that accumulation controlled.
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.out_proj.weight") or name.endswith("feed_forward.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, time)")
        if token_ids.size(1) > self.context_length:
            raise ValueError("sequence length exceeds context_length")
        hidden = self.position(self.token_embedding(token_ids))
        attention_weights: list[Tensor] = []
        for block in self.blocks:
            if return_attention:
                hidden, weights = block(hidden, return_attention=True)
                attention_weights.append(weights)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_layer_norm(hidden))
        if return_attention:
            return logits, attention_weights
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def model_summary(model: MiniGPT) -> str:
    """Return a compact, reproducible architecture and parameter summary."""
    embedding_parameters = sum(parameter.numel() for parameter in model.token_embedding.parameters())
    one_block_parameters = sum(parameter.numel() for parameter in model.blocks[0].parameters())
    all_blocks_parameters = sum(parameter.numel() for parameter in model.blocks.parameters())
    final_norm_parameters = sum(parameter.numel() for parameter in model.final_layer_norm.parameters())
    lm_head_parameters = sum(parameter.numel() for parameter in model.lm_head.parameters())
    return "\n".join(
        [
            "MINIGPT MODEL SUMMARY",
            f"vocab_size: {model.vocab_size}",
            f"context_length: {model.context_length}",
            f"d_model: {model.d_model}",
            f"n_layers: {model.n_layers}",
            f"position_encoding: {model.position.__class__.__name__}",
            f"weight_tying: {model.weight_tying}",
            f"parameters: {model.parameter_count():,}",
            "PARAMETER BREAKDOWN",
            f"token embedding: {embedding_parameters:,}",
            f"one Transformer block: {one_block_parameters:,}",
            f"all {model.n_layers} Transformer blocks: {all_blocks_parameters:,}",
            f"final LayerNorm: {final_norm_parameters:,}",
            f"LM head: {lm_head_parameters:,}",
            "architecture:",
            "  token embedding -> position -> N pre-LN causal blocks",
            "  -> final LayerNorm -> LM head",
        ]
    )


def save_model_summary_figure(model: MiniGPT, output_dir: str | Path = "outputs") -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    """Save Figure 5 plus matching terminal-style text evidence."""
    output_root = Path(output_dir)
    figure_path = output_root / "figures" / "fig5_model_summary.png"
    text_path = output_root / "logs" / "m4_model_summary.txt"
    text = model_summary(model)
    # Keep the terminal-like evidence compact: no decorative empty lower panel.
    figure, axis = plt.subplots(figsize=(10, 4.5)); axis.axis("off")
    axis.text(0.06, 0.90, text, va="top", family="monospace", fontsize=10.5, linespacing=1.22)
    axis.set_title("Figure 5 - Actual MiniGPT architecture and parameter count", fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03); figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight"); plt.close(figure)
    text_path.parent.mkdir(parents=True, exist_ok=True); text_path.write_text(text + "\n", encoding="utf-8")
    return figure_path, text_path

