"""Batch sampling for multi-scale training on native-LR datasets.

A native-LR model's input size depends on the scale, so a batch cannot mix
scales. :class:`ScaleBatchSampler` groups the indices of an ordinary index
sampler into batches and draws one scale per batch, yielding ``(index, scale)``
pairs that :class:`~sisr.datasets.native_lr.TrainDataset` understands.
"""

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class ScaleBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yields batches of ``(index, scale)``, one scale per batch.

    Wraps an index sampler instead of replacing it, so shuffling and
    distributed sharding stay exactly what they were: the same index sampler
    (``RandomSampler`` or ``DistributedSampler``) decides *which* items, this
    decides *how they are grouped and at what scale*. The wrapped sampler is
    exposed as :attr:`sampler`, which is where Lightning looks to call
    ``set_epoch`` on a sharded run.

    The scale of each batch is drawn from the global torch RNG, so
    ``seed_everything`` makes a run reproducible. Ranks draw independently:
    their batches may differ in scale at the same step, which data-parallel
    training does not mind (gradients have the same shape either way).

    Args:
        sampler: Index sampler to batch.
        scales: Scales to draw from, uniformly.
        batch_size: Items per batch.
        drop_last: Drop a final batch smaller than ``batch_size``.

    Raises:
        ValueError: If ``scales`` is empty or ``batch_size`` is not positive.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        scales: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        if not scales:
            raise ValueError("ScaleBatchSampler needs at least one scale.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        self.sampler = sampler
        self.scales = list(scales)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield self._tag(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._tag(batch)

    def _tag(self, batch: list[int]) -> list[tuple[int, int]]:
        scale = self.scales[int(torch.randint(len(self.scales), (1,)))]
        return [(i, scale) for i in batch]

    def __len__(self) -> int:
        n = len(self.sampler)  # type: ignore[arg-type]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)
