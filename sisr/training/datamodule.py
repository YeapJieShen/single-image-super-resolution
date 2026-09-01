"""Generic Lightning DataModule for SR training.

Dataset specs (``{class_path, init_args}``) are materialized lazily in
:meth:`SRDataModule.setup` so expensive constructors (LMDB cache builds)
only run for the stages that need them. Test sets are surfaced through
*both* :meth:`val_dataloader` (for ``cli fit`` monitoring) and
:meth:`test_dataloader` (for ``cli test --ckpt_path`` final eval).
"""

import importlib
import inspect
from typing import Any

import lightning
import torch.distributed as dist
from lightning.pytorch.cli import instantiate_class
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_SPEC_KEYS = frozenset({"class_path", "init_args"})


def _validate_spec(field_name: str, spec: dict[str, Any]) -> None:
    """Raise if ``spec`` isn't a well-formed ``{class_path, init_args}`` mapping.

    Every dataset field on this module is typed ``dict[str, Any]`` (not the
    real dataset class) so :meth:`SRDataModule.setup` can instantiate it lazily.
    The cost: jsonargparse treats the field as an opaque dict, so a dotted CLI
    override like ``--data.train_dataset.init_args.crops_per_image=8`` **cannot
    reach the nested key** — it lands as a stray sibling next to
    ``class_path``/``init_args``, invisible to ``instantiate_class``, which goes
    on building the dataset with the old value. This turns that silent no-op
    into an immediate error.

    Args:
        field_name: Dotted config path for the error message, e.g.
            ``'data.train_dataset'`` or ``'data.test_datasets.Set14'``.
        spec: The value to validate.

    Raises:
        ValueError: If ``spec`` isn't a dict, has no ``class_path``, has any
            key besides ``class_path``/``init_args``, or has a non-dict
            ``init_args``.
    """
    if not isinstance(spec, dict) or "class_path" not in spec:
        raise ValueError(f"{field_name} must be a {{class_path, init_args}} mapping; got {spec!r}.")
    extra = sorted(set(spec) - _SPEC_KEYS)
    if extra:
        raise ValueError(
            f"{field_name} has unexpected key(s) {extra} alongside class_path/init_args: "
            f"{spec!r}. This usually means a CLI override like "
            f"--{field_name}.init_args.<key>=<value> landed as a sibling key instead of "
            f"inside init_args — jsonargparse cannot merge a dotted override into this "
            f"dict[str, Any]-typed field. Edit the YAML directly instead."
        )
    init_args = spec.get("init_args", {})
    if not isinstance(init_args, dict):
        raise ValueError(f"{field_name}.init_args must be a mapping; got {init_args!r}.")


def _accepted_init_args(class_path: str) -> list[str] | None:
    """Returns *class_path*'s accepted keyword argument names, or ``None``.

    Only called on the error path, so importing here costs nothing on a
    healthy run -- and by then the class has already been imported anyway.

    Args:
        class_path: Dotted path from the spec.

    Returns:
        Sorted parameter names, or ``None`` if the class cannot be resolved
        or inspected (in which case the caller falls back to the raw error).
    """
    try:
        module_name, _, attr = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), attr)
        params = inspect.signature(cls).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    return sorted(
        name
        for name, p in params.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and name != "self"
    )


def _require(ds: Dataset | None, which: str, stage: str) -> Dataset:
    """Narrow a lazily-instantiated dataset, naming what should have created it.

    Args:
        ds: The dataset attribute, ``None`` until ``setup`` has run.
        which: Loader name, used to build the error message.
        stage: The ``setup`` stage that instantiates this dataset.

    Returns:
        ``ds``, narrowed to non-optional.

    Raises:
        RuntimeError: If ``setup`` has not instantiated it yet.
    """
    if ds is None:
        raise RuntimeError(
            f"SRDataModule.{which}_dataloader() called before setup({stage!r}) "
            f"instantiated the {which} dataset. Lightning calls setup() itself; a "
            "direct caller has to."
        )
    return ds


def _instantiate_spec(field_name: str, spec: dict[str, Any]) -> Any:
    """Materialises a dataset spec, naming the field when an init_arg is wrong.

    ``_validate_spec`` catches keys that landed *beside* ``init_args``. A
    misspelled key *inside* ``init_args`` gets all the way to the dataset
    constructor instead, and surfaces as a bare ``TypeError`` naming the class
    but not the config path that produced it -- so the reader is told
    ``TrainDataset`` is wrong without being told which of several dataset
    blocks to go and fix.

    Args:
        field_name: Dotted config path, e.g. ``'data.train_dataset'``.
        spec: A spec already checked by :func:`_validate_spec`.

    Returns:
        The instantiated dataset.

    Raises:
        ValueError: If ``init_args`` holds a key the target does not accept.
    """
    try:
        return instantiate_class((), spec)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        accepted = _accepted_init_args(spec["class_path"])
        given = sorted(spec.get("init_args", {}))
        unknown = [k for k in given if accepted is not None and k not in accepted]
        detail = f" Unexpected: {unknown}." if unknown else ""
        options = f" Accepted: {accepted}." if accepted is not None else ""
        raise ValueError(
            f"{field_name}.init_args holds a key {spec['class_path']} does not accept."
            f"{detail}{options} Original error: {exc}"
        ) from exc


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
      drives ``loss/val`` / ``psnr/val/*`` / ``ssim/val/*``.
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
        _validate_spec("data.train_dataset", train_dataset)
        _validate_spec("data.val_dataset", val_dataset)
        for name, spec in (test_datasets or {}).items():
            _validate_spec(f"data.test_datasets.{name}", spec)
        if predict_dataset is not None:
            _validate_spec("data.predict_dataset", predict_dataset)

        self._train_spec = train_dataset
        self._val_spec = val_dataset
        self._test_specs = test_datasets or {}
        self._predict_spec = predict_dataset
        self._train_dl_kwargs = train_dataloader_kwargs or {}
        self._val_dl_kwargs = val_dataloader_kwargs or {"batch_size": 1, "num_workers": 1}
        # dict(...), not the object itself: unset, this used to BE
        # _val_dl_kwargs, so the first thing to mutate either would silently
        # change both loaders.
        self._test_dl_kwargs = test_dataloader_kwargs or dict(self._val_dl_kwargs)
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

    @property
    def train_dataset(self) -> Dataset | None:
        """The instantiated train dataset, or ``None`` before ``setup('fit')`` runs.

        Public read accessor — lets consumers outside this module (e.g.
        :meth:`~sisr.training.SRLightning.setup`'s input-contract probe)
        inspect the real dataset without reaching into private state.
        """
        return self._train_ds

    @property
    def val_dataset(self) -> Dataset | None:
        """The instantiated primary val dataset, or ``None`` before setup runs."""
        return self._val_ds

    @property
    def test_datasets(self) -> dict[str, Dataset]:
        """Instantiated test datasets by name — empty before setup runs (or if none configured)."""
        return dict(self._test_ds)

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
            self._train_ds = _instantiate_spec("data.train_dataset", self._train_spec)
        if stage in ("fit", "validate", "test", None) and not self._test_ds and self._test_specs:
            self._test_ds = {
                name: _instantiate_spec(f"data.test_datasets.{name}", spec)
                for name, spec in self._test_specs.items()
            }
        if stage in ("fit", "validate", None) and self._val_ds is None:
            self._val_ds = _instantiate_spec("data.val_dataset", self._val_spec)
        if stage in ("predict", None) and self._predict_ds is None and self._predict_spec:
            self._predict_ds = _instantiate_spec("data.predict_dataset", self._predict_spec)

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader, sharded across processes when distributed.

        ``shuffle=True`` is forced — any ``shuffle`` entry the user supplies
        in ``train_dataloader_kwargs`` would be a TypeError (duplicate
        kwarg) and the YAML schema does not expose ``shuffle``.

        **Training is the only loader this module shards, and it shards it here
        rather than letting Lightning do it.** Lightning's injection is one
        trainer-wide switch that would also split the eval loaders, and
        ``DistributedSampler`` pads by *repeating* samples — a five-image
        benchmark across four processes becomes eight, three counted twice, and
        the figure is no longer that set's score. Every process therefore scores
        every benchmark image, and the reduction is a no-op that catches
        divergence.

        Returns:
            DataLoader over the train spec: shuffled, and split per process when
            a process group is running.

        Raises:
            RuntimeError: If ``setup('fit')`` has not run.
        """
        train_ds = _require(self._train_ds, "train", "fit")
        if dist.is_available() and dist.is_initialized():
            sampler: DistributedSampler[Any] = DistributedSampler(train_ds, shuffle=True)
            return DataLoader(train_ds, sampler=sampler, **self._train_dl_kwargs)
        return DataLoader(train_ds, shuffle=True, **self._train_dl_kwargs)

    def val_dataloader(self) -> list[DataLoader]:
        """Primary validation loader followed by every test loader.

        Index 0 is the held-out primary val set; indices 1+ are the test
        sets, which lets :class:`~sisr.training.callbacks.BenchmarkImageLogger`
        log Set5 / Set14 progress during every val cycle of ``cli fit``.

        Returns:
            List ``[primary_val_loader, test_loader_1, test_loader_2,
            ...]``. Length is ``1 + len(test_names)``.

        Raises:
            RuntimeError: If ``setup('fit')`` has not run.
        """
        val_ds = _require(self._val_ds, "val", "fit")
        loaders = [DataLoader(val_ds, shuffle=False, **self._val_dl_kwargs)]
        for ds in self._test_ds.values():
            loaders.append(DataLoader(ds, shuffle=False, **self._test_dl_kwargs))
        return loaders

    def test_dataloader(self) -> list[DataLoader]:
        """Test loaders only — for ``cli test --ckpt_path <path>`` final eval.

        Separates "not configured" from "not set up", the way
        :meth:`predict_dataloader` already does. Returning ``[]`` for both made
        them indistinguishable, and a ``test`` run that reached the second
        evaluated nothing and reported success. This is the final evaluation
        path, so the number that does not get produced is the one someone would
        publish.

        Returns:
            List of one DataLoader per entry in ``test_datasets``, in
            insertion order. Empty when no test sets are configured — that
            state is legitimate and stays silent.

        Raises:
            RuntimeError: If test sets are configured but ``setup`` has not
                instantiated them. Lightning calls ``setup()`` itself; a direct
                caller has to.
        """
        if self._test_specs and not self._test_ds:
            raise RuntimeError(
                f"SRDataModule.test_dataloader() called before setup('test') instantiated "
                f"the {len(self._test_specs)} configured test set(s) "
                f"({', '.join(self._test_specs)}). Lightning calls setup() itself; a direct "
                f"caller has to. Returning an empty list here would make a `test` run "
                f"evaluate nothing and report success."
            )
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
