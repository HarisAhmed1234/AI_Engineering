"""M4 data-pipeline and deterministic-tokenizer tests."""

from pathlib import Path

import torch

from src.dataset import NextTokenDataset, split_tokens
from src.tokenizer import CharacterTokenizer


def test_targets_are_inputs_shifted_by_one() -> None:
    dataset = NextTokenDataset(torch.arange(20), context_length=5, stride=5)
    x, y = dataset[1]
    assert torch.equal(x, torch.tensor([5, 6, 7, 8, 9]))
    assert torch.equal(y, torch.tensor([6, 7, 8, 9, 10]))
    assert torch.equal(y[:-1], x[1:])


def test_split_is_positional_and_tokenizer_is_sorted_and_roundtrips() -> None:
    train, validation = split_tokens(torch.arange(10))
    assert torch.equal(train, torch.arange(9))
    assert torch.equal(validation, torch.tensor([9]))
    tokenizer = CharacterTokenizer.from_text("cab\n")
    assert tokenizer.characters == ["\n", "a", "b", "c"]
    path = Path("outputs/logs/test_vocab_roundtrip.json")
    try:
        tokenizer.save(path)
        loaded = CharacterTokenizer.load(path)
        assert loaded.decode(loaded.encode("cab\n")) == "cab\n"
    finally:
        path.unlink(missing_ok=True)
