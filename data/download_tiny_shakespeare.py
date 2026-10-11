"""Download the public Tiny Shakespeare corpus without committing it to Git."""

from __future__ import annotations

import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "input.txt"
SOURCES = [
    ("Karpathy char-rnn raw corpus", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
    ("Hugging Face mirror", "https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/input.txt?download=true"),
]


def download() -> Path:
    """Try the requested Karpathy source, then the Hugging Face mirror."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name, url in SOURCES:
        try:
            print(f"Trying {name}: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "transformer-gpt-assignment/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
            if len(text) < 100_000:
                raise ValueError(f"download was unexpectedly small ({len(text)} characters)")
            OUTPUT_PATH.write_text(text, encoding="utf-8")
            print(f"Downloaded {OUTPUT_PATH.name}: {OUTPUT_PATH.stat().st_size:,} bytes, {text.count(chr(10)):,} lines")
            return OUTPUT_PATH
        except Exception as error:  # Keep the fallback attempt visible to the student.
            failures.append(f"{name}: {error}")
            print(f"Failed: {error}")
    raise RuntimeError("All dataset sources failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    download()

