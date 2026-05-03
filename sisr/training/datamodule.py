from typing import Any, Dict, Optional

import lightning
from torch.utils.data import DataLoader, Dataset

from ..datasets.srcnn import TrainDataset, ValidationDataset


class SRDataModule(lightning.LightningDataModule):
    """
    Generic LightningDataModule for single-image super-resolution.

    Owns the train / validation / benchmark Dataset constructions and exposes
    them as DataLoaders.  Datasets are instantiated lazily in :meth:`setup`
    so that expensive work (e.g. LMDB cache builds in :class:`TrainDataset`)
    only runs once trainer subcommands are dispatched.

    Validation dataloaders are returned as a list:
    ``[primary_val] + [benchmarks...]``.  Index 0 is the primary loader that
    drives ``val_loss``, ``val_psnr``, etc.; subsequent indices are
    benchmark sets (Set5, Set14, …) consumed by
    :class:`~sisr.callbacks.BenchmarkImageLogger`.

    Architecture-agnostic: the dataset classes themselves are constructor
    parameters, so swapping in an alternative pipeline (e.g. random-crop
    style for SRResNet) only requires pointing the YAML at a different class.

    Args:
        train_dataset: Keyword arguments forwarded to ``train_dataset_class``
            in :meth:`setup` (``stage='fit'``).
        val_dataset: Keyword arguments forwarded to ``val_dataset_class``.
        benchmark_datasets: Mapping ``{name: dataset_kwargs}`` for benchmark
            evaluation sets.  ``None`` disables benchmark evaluation.
        train_dataloader_kwargs: DataLoader kwargs for the training loader
            (excluding ``shuffle``, which is forced to ``True``).
        val_dataloader_kwargs: DataLoader kwargs for the primary validation
            loader.  Defaults to ``{'batch_size': 1, 'num_workers': 1}``.
        benchmark_dataloader_kwargs: DataLoader kwargs reused for every
            benchmark loader.  Defaults to the same as the validation loader.
        train_dataset_class: Dataset class used for training.  Defaults to
            :class:`sisr.datasets.srcnn.TrainDataset`.
        val_dataset_class: Dataset class used for validation.  Defaults to
            :class:`sisr.datasets.srcnn.ValidationDataset`.
        benchmark_dataset_class: Dataset class used for benchmark sets.  When
            ``None``, falls back to ``val_dataset_class``.
    """

    def __init__(
        self,
        train_dataset: Dict[str, Any],
        val_dataset: Dict[str, Any],
        benchmark_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        val_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        benchmark_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        train_dataset_class: type = TrainDataset,
        val_dataset_class: type = ValidationDataset,
        benchmark_dataset_class: Optional[type] = None,
    ):
        super().__init__()
        self._train_kwargs = train_dataset
        self._val_kwargs = val_dataset
        self._benchmark_kwargs = benchmark_datasets or {}
        self._train_dl_kwargs = train_dataloader_kwargs or {}
        self._val_dl_kwargs = val_dataloader_kwargs or {'batch_size': 1, 'num_workers': 1}
        self._benchmark_dl_kwargs = benchmark_dataloader_kwargs or self._val_dl_kwargs
        self._train_cls = train_dataset_class
        self._val_cls = val_dataset_class
        self._benchmark_cls = benchmark_dataset_class or val_dataset_class

        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._benchmark_ds: Dict[str, Dataset] = {}

    @property
    def benchmark_names(self) -> list:
        """Ordered list of benchmark dataset names — drives BenchmarkImageLogger auto-discovery."""
        return list(self._benchmark_kwargs.keys())

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets lazily based on the trainer stage."""
        if stage in ('fit', None) and self._train_ds is None:
            self._train_ds = self._train_cls(**self._train_kwargs)
        if stage in ('fit', 'validate', None) and self._val_ds is None:
            self._val_ds = self._val_cls(**self._val_kwargs)
            self._benchmark_ds = {
                name: self._benchmark_cls(**kwargs)
                for name, kwargs in self._benchmark_kwargs.items()
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_ds, shuffle=True, **self._train_dl_kwargs)

    def val_dataloader(self) -> list:
        loaders = [DataLoader(self._val_ds, shuffle=False, **self._val_dl_kwargs)]
        for ds in self._benchmark_ds.values():
            loaders.append(DataLoader(ds, shuffle=False, **self._benchmark_dl_kwargs))
        return loaders
