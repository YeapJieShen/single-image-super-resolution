from typing import Any

import lightning
from lightning.pytorch.cli import instantiate_class
from torch.utils.data import DataLoader, Dataset


class SRDataModule(lightning.LightningDataModule):
    """Generic LightningDataModule for single-image super-resolution.

    Owns the train / validation / test Dataset constructions and exposes them
    as DataLoaders. Datasets are described by ``{class_path, init_args}`` specs
    and instantiated lazily in :meth:`setup` so that expensive work (e.g. LMDB
    cache builds in ``TrainDataset``) only runs once trainer subcommands are
    dispatched.

    Test sets (Set5, Set14, …) are surfaced through *both* dataloader hooks:

    * :meth:`val_dataloader` returns ``[primary_val] + [test_loaders...]``,
      so they fire during every val cycle of ``cli fit`` for monitoring
      (driven by :class:`~sisr.training.callbacks.BenchmarkImageLogger`).
      ``dataloader_idx == 0`` is the primary held-out validation loader and
      drives ``val_loss`` / ``val_psnr`` / ``val_ssim``.
    * :meth:`test_dataloader` returns the test loaders only, for
      ``cli test --ckpt_path <path>`` final evaluation.

    Architecture-agnostic: each dataset spec carries its own ``class_path``,
    so swapping in an alternative pipeline (e.g. SRResNet's random-crop dataset)
    only requires pointing the YAML at a different class.

    Args:
        train_dataset: ``{class_path, init_args}`` spec for the training dataset.
        val_dataset: ``{class_path, init_args}`` spec for the primary
            held-out validation dataset.
        test_datasets: ``{name: {class_path, init_args}}`` mapping for held-out
            test sets (Set5, Set14, …). ``None`` disables test evaluation.
        train_dataloader_kwargs: DataLoader kwargs for the training loader
            (excluding ``shuffle``, which is forced to ``True``).
        val_dataloader_kwargs: DataLoader kwargs for the primary validation
            loader. Defaults to ``{'batch_size': 1, 'num_workers': 1}``.
        test_dataloader_kwargs: DataLoader kwargs reused for every test
            loader. Defaults to the same as the validation loader.
    """

    def __init__(
        self,
        train_dataset: dict[str, Any],
        val_dataset: dict[str, Any],
        test_datasets: dict[str, dict[str, Any]] | None = None,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._train_spec = train_dataset
        self._val_spec = val_dataset
        self._test_specs = test_datasets or {}
        self._train_dl_kwargs = train_dataloader_kwargs or {}
        self._val_dl_kwargs = val_dataloader_kwargs or {'batch_size': 1, 'num_workers': 1}
        self._test_dl_kwargs = test_dataloader_kwargs or self._val_dl_kwargs

        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: dict[str, Dataset] = {}

    @property
    def test_names(self) -> list:
        """Ordered list of test dataset names — drives BenchmarkImageLogger auto-discovery."""
        return list(self._test_specs.keys())

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets lazily based on the trainer stage."""
        if stage in ('fit', None) and self._train_ds is None:
            self._train_ds = instantiate_class((), self._train_spec)
        if stage in ('fit', 'validate', 'test', None) and not self._test_ds and self._test_specs:
            self._test_ds = {
                name: instantiate_class((), spec)
                for name, spec in self._test_specs.items()
            }
        if stage in ('fit', 'validate', None) and self._val_ds is None:
            self._val_ds = instantiate_class((), self._val_spec)

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader.

        ``shuffle=True`` is forced — any ``shuffle`` entry the user supplies
        in ``train_dataloader_kwargs`` would be a TypeError (duplicate
        kwarg) and the YAML schema does not expose ``shuffle``.

        Returns:
            DataLoader over the train spec, shuffled.
        """
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
