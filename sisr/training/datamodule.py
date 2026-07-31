"""Generic Lightning DataModule for SR training.

Dataset specs (``{class_path, init_args}``) are materialized lazily in
:meth:`SRDataModule.setup` so expensive constructors (LMDB cache builds)
only run for the stages that need them. Test sets are surfaced through
*both* :meth:`val_dataloader` (for ``cli fit`` monitoring) and
:meth:`test_dataloader` (for ``cli test --ckpt_path`` final eval).
"""

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

    :meth:`predict_dataloader` serves a separate, optional LR-only dataset
    (e.g. :class:`~sisr.datasets.predict.PredictDataset`) for ``cli predict``
    — the only stage with no HR reference at all.

    Architecture-agnostic: each dataset spec carries its own ``class_path``,
    so swapping in an alternative pipeline (e.g. SRResNet's random-crop dataset)
    only requires pointing the YAML at a different class.

    Args:
        train_dataset: ``{class_path, init_args}`` spec for the training dataset.
        val_dataset: ``{class_path, init_args}`` spec for the primary
            held-out validation dataset.
        test_datasets: ``{name: {class_path, init_args}}`` mapping for held-out
            test sets (Set5, Set14, …). ``None`` disables test evaluation.
        predict_dataset: ``{class_path, init_args}`` spec for the LR-only
            prediction dataset. ``None`` (default) leaves ``cli predict``
            unconfigured — :meth:`predict_dataloader` raises if called.
        train_dataloader_kwargs: DataLoader kwargs for the training loader
            (excluding ``shuffle``, which is forced to ``True``).
        val_dataloader_kwargs: DataLoader kwargs for the primary validation
            loader. Defaults to ``{'batch_size': 1, 'num_workers': 1}``.
        test_dataloader_kwargs: DataLoader kwargs reused for every test
            loader. Defaults to the same as the validation loader.
        predict_dataloader_kwargs: DataLoader kwargs for the prediction
            loader. Defaults to ``{'batch_size': 1, 'num_workers': 0}`` —
            predict images vary in size so batch_size 1 is the safe default,
            and num_workers 0 keeps a one-off inference call free of
            multiprocessing startup cost.
    """

    def __init__(
        self,
        train_dataset: dict[str, Any],
        val_dataset: dict[str, Any],
        test_datasets: dict[str, dict[str, Any]] | None = None,
        predict_dataset: dict[str, Any] | None = None,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
        predict_dataloader_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._train_spec = train_dataset
        self._val_spec = val_dataset
        self._test_specs = test_datasets or {}
        self._predict_spec = predict_dataset
        self._train_dl_kwargs = train_dataloader_kwargs or {}
        self._val_dl_kwargs = val_dataloader_kwargs or {"batch_size": 1, "num_workers": 1}
        self._test_dl_kwargs = test_dataloader_kwargs or self._val_dl_kwargs
        self._predict_dl_kwargs = predict_dataloader_kwargs or {"batch_size": 1, "num_workers": 0}

        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: dict[str, Dataset] = {}
        self._predict_ds: Dataset | None = None

    @property
    def test_names(self) -> list[str]:
        """Ordered list of test dataset names — drives BenchmarkImageLogger auto-discovery.

        Returns:
            List of names in the order they appear in
            ``test_datasets``. Empty when no test sets are configured.
        """
        return list(self._test_specs.keys())

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets lazily based on the trainer stage.

        Called by Lightning at the start of every subcommand. Specs are
        materialized via :func:`lightning.pytorch.cli.instantiate_class`
        so expensive constructors (LMDB cache builds, etc.) only run for
        the stages that need them.

        Args:
            stage: Lightning trainer stage — ``'fit'``, ``'validate'``,
                ``'test'``, ``'predict'``, or ``None`` (for all). Determines
                which datasets get instantiated.
        """
        if stage in ("fit", None) and self._train_ds is None:
            self._train_ds = instantiate_class((), self._train_spec)
        if stage in ("fit", "validate", "test", None) and not self._test_ds and self._test_specs:
            self._test_ds = {
                name: instantiate_class((), spec) for name, spec in self._test_specs.items()
            }
        if stage in ("fit", "validate", None) and self._val_ds is None:
            self._val_ds = instantiate_class((), self._val_spec)
        if stage in ("predict", None) and self._predict_ds is None and self._predict_spec:
            self._predict_ds = instantiate_class((), self._predict_spec)

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader.

        ``shuffle=True`` is forced — any ``shuffle`` entry the user supplies
        in ``train_dataloader_kwargs`` would be a TypeError (duplicate
        kwarg) and the YAML schema does not expose ``shuffle``.

        Returns:
            DataLoader over the train spec, shuffled.
        """
        return DataLoader(self._train_ds, shuffle=True, **self._train_dl_kwargs)

    def val_dataloader(self) -> list[DataLoader]:
        """Primary validation loader followed by every test loader.

        Index 0 is the held-out primary val set; indices 1+ are the test
        sets, which lets :class:`~sisr.training.callbacks.BenchmarkImageLogger`
        log Set5 / Set14 progress during every val cycle of ``cli fit``.

        Returns:
            List ``[primary_val_loader, test_loader_1, test_loader_2,
            ...]``. Length is ``1 + len(test_names)``.
        """
        loaders = [DataLoader(self._val_ds, shuffle=False, **self._val_dl_kwargs)]
        for ds in self._test_ds.values():
            loaders.append(DataLoader(ds, shuffle=False, **self._test_dl_kwargs))
        return loaders

    def test_dataloader(self) -> list[DataLoader]:
        """Test loaders only — for ``cli test --ckpt_path <path>`` final eval.

        Returns:
            List of one DataLoader per entry in ``test_datasets``, in
            insertion order. Empty when no test sets are configured.
        """
        return [
            DataLoader(ds, shuffle=False, **self._test_dl_kwargs) for ds in self._test_ds.values()
        ]

    def predict_dataloader(self) -> DataLoader:
        """Build the DataLoader over the LR-only prediction dataset.

        Returns:
            DataLoader over the ``predict_dataset`` spec.

        Raises:
            RuntimeError: If ``predict_dataset`` was not configured — there
                is nothing to run ``cli predict`` against.
        """
        if self._predict_ds is None:
            raise RuntimeError(
                "SRDataModule.predict_dataloader() called but no predict_dataset was "
                "configured. Set data.predict_dataset (a {class_path, init_args} spec, "
                "e.g. sisr.datasets.predict.PredictDataset) pointing at a directory of "
                "LR images."
            )
        return DataLoader(self._predict_ds, shuffle=False, **self._predict_dl_kwargs)
