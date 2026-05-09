from typing import Any, Dict, Optional

import lightning
from torch.utils.data import DataLoader, Dataset

from ..datasets.srcnn import TrainDataset, ValidationDataset


class SRDataModule(lightning.LightningDataModule):
    """
    Generic LightningDataModule for single-image super-resolution.

    Owns the train / validation / test Dataset constructions and exposes them
    as DataLoaders.  Datasets are instantiated lazily in :meth:`setup` so that
    expensive work (e.g. LMDB cache builds in :class:`TrainDataset`) only runs
    once trainer subcommands are dispatched.

    Test sets (Set5, Set14, …) are surfaced through *both* dataloader hooks:

    * :meth:`val_dataloader` returns ``[primary_val] + [test_loaders...]``,
      so they fire during every val cycle of ``cli fit`` for monitoring
      (driven by :class:`~sisr.training.callbacks.BenchmarkImageLogger`).
      ``dataloader_idx == 0`` is the primary held-out validation loader and
      drives ``val_loss`` / ``val_psnr`` / ``val_ssim``.
    * :meth:`test_dataloader` returns the test loaders only, for
      ``cli test --ckpt_path <path>`` final evaluation.

    Architecture-agnostic: the dataset classes themselves are constructor
    parameters, so swapping in an alternative pipeline (e.g. random-crop
    style for SRResNet) only requires pointing the YAML at a different class.

    Args:
        train_dataset: Keyword arguments forwarded to ``train_dataset_class``
            in :meth:`setup` (``stage='fit'``).
        val_dataset: Keyword arguments forwarded to ``val_dataset_class``.
        test_datasets: Mapping ``{name: dataset_kwargs}`` for held-out test
            sets (Set5, Set14, …).  ``None`` disables test evaluation.
        train_dataloader_kwargs: DataLoader kwargs for the training loader
            (excluding ``shuffle``, which is forced to ``True``).
        val_dataloader_kwargs: DataLoader kwargs for the primary validation
            loader.  Defaults to ``{'batch_size': 1, 'num_workers': 1}``.
        test_dataloader_kwargs: DataLoader kwargs reused for every test
            loader.  Defaults to the same as the validation loader.
        train_dataset_class: Dataset class used for training.  Defaults to
            :class:`sisr.datasets.srcnn.TrainDataset`.
        val_dataset_class: Dataset class used for validation.  Defaults to
            :class:`sisr.datasets.srcnn.ValidationDataset`.
        test_dataset_class: Dataset class used for test sets.  When ``None``,
            falls back to ``val_dataset_class``.
    """

    def __init__(
        self,
        train_dataset: Dict[str, Any],
        val_dataset: Dict[str, Any],
        test_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        val_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        train_dataset_class: type = TrainDataset,
        val_dataset_class: type = ValidationDataset,
        test_dataset_class: Optional[type] = None,
    ):
        super().__init__()
        self._train_kwargs = train_dataset
        self._val_kwargs = val_dataset
        self._test_kwargs = test_datasets or {}
        self._train_dl_kwargs = train_dataloader_kwargs or {}
        self._val_dl_kwargs = val_dataloader_kwargs or {'batch_size': 1, 'num_workers': 1}
        self._test_dl_kwargs = test_dataloader_kwargs or self._val_dl_kwargs
        self._train_cls = train_dataset_class
        self._val_cls = val_dataset_class
        self._test_cls = test_dataset_class or val_dataset_class

        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Dict[str, Dataset] = {}

    @property
    def test_names(self) -> list:
        """Ordered list of test dataset names — drives BenchmarkImageLogger auto-discovery."""
        return list(self._test_kwargs.keys())

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets lazily based on the trainer stage."""
        if stage in ('fit', None) and self._train_ds is None:
            self._train_ds = self._train_cls(**self._train_kwargs)
        if stage in ('fit', 'validate', 'test', None) and not self._test_ds and self._test_kwargs:
            self._test_ds = {
                name: self._test_cls(**kwargs)
                for name, kwargs in self._test_kwargs.items()
            }
        if stage in ('fit', 'validate', None) and self._val_ds is None:
            self._val_ds = self._val_cls(**self._val_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_ds, shuffle=True, **self._train_dl_kwargs)

    def val_dataloader(self) -> list:
        """Primary validation loader followed by every test loader.

        Index 0 is the held-out primary val set; indices 1+ are the test sets,
        which lets `BenchmarkImageLogger` log Set5/Set14 progress during
        every val cycle of `cli fit`.
        """
        loaders = [DataLoader(self._val_ds, shuffle=False, **self._val_dl_kwargs)]
        for ds in self._test_ds.values():
            loaders.append(DataLoader(ds, shuffle=False, **self._test_dl_kwargs))
        return loaders

    def test_dataloader(self) -> list:
        """Test loaders only — for `cli test --ckpt_path <path>` final eval."""
        return [
            DataLoader(ds, shuffle=False, **self._test_dl_kwargs)
            for ds in self._test_ds.values()
        ]
