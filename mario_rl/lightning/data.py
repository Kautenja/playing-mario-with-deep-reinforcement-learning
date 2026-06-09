"""Minimal iterable dataset that lets Lightning drive online DQN updates."""
from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import DataLoader, IterableDataset


class StepDataset(IterableDataset):
    """Yield placeholder step indices for a bounded online training loop."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

    def __iter__(self) -> Iterator[int]:
        yield from range(self.max_steps)


def build_step_dataloader(max_steps: int) -> DataLoader:
    """Return the dataloader used by ``DQNLightningModule.train_dataloader``."""
    return DataLoader(StepDataset(max_steps), batch_size=None)


__all__ = ["StepDataset", "build_step_dataloader"]
