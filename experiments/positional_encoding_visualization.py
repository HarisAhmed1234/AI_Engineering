"""Generate Figure 4 and a careful fixed-versus-learned position comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional

from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig4_positional_encoding.png"
TEXT_PATH = ROOT / "outputs" / "logs" / "m2_positional_encoding.txt"


def main() -> None:
    torch.manual_seed(23)
    positions, d_model = 64, 32
    sinusoidal = SinusoidalPositionalEncoding(d_model=d_model, max_length=positions)
    learned = LearnedPositionalEmbedding(d_model=d_model, max_length=positions)
    zero_embeddings = torch.zeros(1, positions, d_model)
    fixed_matrix = sinusoidal(zero_embeddings)[0].detach()
    learned_matrix = learned(zero_embeddings)[0].detach()
    cosine_similarity = functional.normalize(fixed_matrix, dim=-1) @ functional.normalize(fixed_matrix, dim=-1).T

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    fixed_image = axes[0, 0].imshow(fixed_matrix.T.numpy(), aspect="auto", cmap="coolwarm")
    axes[0, 0].set_title("Sinusoidal position encoding heatmap")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Embedding dimension")
    figure.colorbar(fixed_image, ax=axes[0, 0])

    for dimension in range(4):
        axes[0, 1].plot(fixed_matrix[:, dimension].numpy(), label=f"dimension {dimension}")
    axes[0, 1].set_title("Sin/cos curves across positions")
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_ylabel("Encoding value")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    similarity_image = axes[1, 0].imshow(cosine_similarity.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 0].set_title("Cosine similarity between sinusoidal positions")
    axes[1, 0].set_xlabel("Key position")
    axes[1, 0].set_ylabel("Query position")
    figure.colorbar(similarity_image, ax=axes[1, 0])

    learned_image = axes[1, 1].imshow(learned_matrix.T.numpy(), aspect="auto", cmap="coolwarm")
    axes[1, 1].set_title("Optional learned-position baseline (untrained)")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("Embedding dimension")
    figure.colorbar(learned_image, ax=axes[1, 1])
    figure.suptitle("Figure 4 - Positional encodings add order information", fontsize=15)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    text = (
        "M2 POSITIONAL ENCODING EVIDENCE\n"
        f"sinusoidal shape: {tuple(fixed_matrix.shape)}\n"
        f"learned baseline shape: {tuple(learned_matrix.shape)}\n"
        f"position 0 first four fixed values: {fixed_matrix[0, :4].tolist()}\n"
        f"position 1 first four fixed values: {fixed_matrix[1, :4].tolist()}\n"
        "The learned panel is an untrained structural baseline, not a quality comparison. "
        "A training-quality comparison is deferred until the M4/M6 model and ablation pipeline exist.\n"
    )
    TEXT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved Figure 4: {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

