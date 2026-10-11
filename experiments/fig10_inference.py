"""Render Figure 10 from saved real CLI and local-API call transcripts.

This rendering-only script never starts Uvicorn, uses TestClient, opens sockets,
loads the model, or alters either evidence log.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig10_inference.png"

# The commands caption the saved calls. CLI flags match the settings printed in
# fig10_cli.txt; max_new_tokens uses its documented CLI default, so it is not
# presented as a separately logged command-line flag.
CLI_COMMAND = (
    '$ python -m src.generate --prompt "The future of artificial intelligence" '
    "--seed 7 --temperature 0.8 --top-k 40 "
    "--checkpoint checkpoints/minigpt_final.pt"
)
API_COMMAND = (
    "$ curl -X POST http://127.0.0.1:8000/generate \\\n  -H 'Content-Type: application/json' \\\n  --data '{\"prompt\":\"ROMEO:\",\"max_new_tokens\":80,\"temperature\":0.8,\"top_k\":40}'"
)


def terminal_text(command: str, saved_output: str, width: int) -> str:
    """Wrap only for image layout while retaining the saved text content."""

    command_lines = "\n".join(textwrap.wrap(command, width=width, break_long_words=False))
    output_lines: list[str] = []
    for line in saved_output.rstrip("\n").splitlines():
        output_lines.extend(textwrap.wrap(line, width=width, break_long_words=False) or [""])
    return command_lines + "\n\n" + "\n".join(output_lines)


def draw_terminal(axis: plt.Axes, title: str, transcript: str) -> None:
    axis.set_facecolor("#101418")
    axis.set_title(title, fontsize=12, weight="bold", pad=10)
    axis.text(0.025, 0.97, transcript, transform=axis.transAxes, va="top", ha="left",
              fontsize=8.1, family="monospace", color="#d9e2f1")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#3e4b59")


def main() -> None:
    """Read immutable saved evidence and write the figure only."""

    cli_log = LOG_DIR / "fig10_cli.txt"
    api_log = LOG_DIR / "fig10_api.txt"
    cli_output = cli_log.read_text(encoding="utf-8")
    api_output = api_log.read_text(encoding="utf-8")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # The saved outputs are short, so a compact canvas avoids empty terminal area.
    figure, axes = plt.subplots(1, 2, figsize=(17, 3.6), constrained_layout=True)
    figure.patch.set_facecolor("white")
    draw_terminal(axes[0], "Saved real CLI call", terminal_text(CLI_COMMAND, cli_output, width=59))
    draw_terminal(axes[1], "Saved real local FastAPI call", terminal_text(API_COMMAND, api_output, width=59))
    figure.suptitle("Figure 10. MiniGPT inference evidence: CLI and local HTTP API", fontsize=15, weight="bold")
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Rendered {FIGURE_PATH} from {cli_log.name} and {api_log.name}; logs were read only.")


if __name__ == "__main__":
    main()
