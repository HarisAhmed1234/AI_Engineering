"""Deterministic character-level tokenizer for Tiny Shakespeare.

Special-token policy: none. Every character observed in the public corpus,
including whitespace and punctuation, is a valid token. This avoids inventing an
unknown or padding token for a fixed character corpus; batches all have equal length.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


class CharacterTokenizer:
    """Map sorted unique characters to stable integer IDs and back."""

    def __init__(self, characters: list[str]) -> None:
        if characters != sorted(set(characters)):
            raise ValueError("characters must be sorted and unique for deterministic IDs")
        self.characters = characters
        self.stoi = {character: index for index, character in enumerate(characters)}
        self.itos = {index: character for character, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character not in vocabulary: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.itos[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"token ID not in vocabulary: {error.args[0]}") from error

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"characters": self.characters}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["characters"])

    def printable_vocabulary(self) -> str:
        """Return a repr-safe complete vocabulary for logs and report evidence."""
        return "\n".join(f"{index:3d}: {character!r}" for index, character in enumerate(self.characters))


def main() -> None:
    """Print the reproducible vocabulary and optionally save it as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", nargs="?", default="data/input.txt")
    parser.add_argument("--save", default="outputs/logs/tiny_shakespeare_vocab.json")
    arguments = parser.parse_args()
    tokenizer = CharacterTokenizer.from_text(Path(arguments.text_path).read_text(encoding="utf-8"))
    tokenizer.save(arguments.save)
    print("SPECIAL TOKENS: none (every observed character is valid; windows need no padding)")
    print(f"VOCAB SIZE: {tokenizer.vocab_size}")
    print(tokenizer.printable_vocabulary())
    print(f"Saved vocabulary: {arguments.save}")


if __name__ == "__main__":
    main()

