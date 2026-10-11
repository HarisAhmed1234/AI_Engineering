"""Contiguous fixed-length next-token datasets with a positional 90/10 split."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Return `x=tokens[i:i+T]` and its one-position-shifted target `y`."""

    def __init__(self, tokens: list[int] | Tensor, context_length: int, stride: int | None = None) -> None:
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        if context_length <= 0 or self.stride <= 0:
            raise ValueError("context_length and stride must be positive")
        if len(self.tokens) <= context_length:
            raise ValueError("not enough tokens for one input/target window")
        self.starts = list(range(0, len(self.tokens) - context_length, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        end = start + self.context_length
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


def split_tokens(tokens: list[int] | Tensor, train_fraction: float = 0.9) -> tuple[Tensor, Tensor]:
    """Split once by corpus position; no sample may cross the train/validation boundary."""
    all_tokens = torch.as_tensor(tokens, dtype=torch.long)
    split_index = int(len(all_tokens) * train_fraction)
    return all_tokens[:split_index], all_tokens[split_index:]


def create_dataloaders(
    tokens: list[int] | Tensor, context_length: int, batch_size: int, train_stride: int = 32
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create overlapping training windows and non-overlapping validation windows.

    Training uses a smaller stride for more varied overlapping examples. Validation
    stays at one full context-length stride so its estimate avoids near-duplicates.
    Both loaders use `num_workers=0` for reliable Windows execution.
    """
    train_tokens, validation_tokens = split_tokens(tokens)
    train_dataset = NextTokenDataset(train_tokens, context_length, train_stride)
    validation_dataset = NextTokenDataset(validation_tokens, context_length, context_length)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )

