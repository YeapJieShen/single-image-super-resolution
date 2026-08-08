r"""LightningCLI entrypoint for SR training.

Usage:
    sisr fit --config templates/config.srcnn.template.yaml
    sisr test --config templates/config.srcnn.template.yaml \
              --ckpt_path <best.ckpt>
    sisr export --config templates/config.srcnn.template.yaml \
                --output_path model.onnx --ckpt_path <best.ckpt>

The ``sisr`` console script is registered by ``pyproject.toml``; ``python -m
sisr.cli ...`` also works.

Top-level YAML keys ``optimizer:`` / ``lr_scheduler:`` are linked into
``model.init_args.optimizer`` / ``model.init_args.lr_scheduler`` by
``SRLightningCLI`` so :class:`~sisr.training.SRLightning` can build its
``configure_optimizers`` from them while keeping the YAML symmetric across
architectures.

Top-level ``matmul_precision:`` (``'highest' | 'high' | 'medium'``) calls
:func:`torch.set_float32_matmul_precision` once at startup. Set to ``'high'``
on Ampere+ GPUs to enable TF32 matmul kernels.

``sisr export`` is a thin CLI wrapper around
:func:`sisr.export.to_onnx` — see that module for what gets exported and its
SRCNN limitation. It requires the optional ``export`` extra
(``pip install '.[export]'``).

cuDNN autotuning is *not* exposed here — Lightning's own ``trainer.benchmark:``
already assigns ``torch.backends.cudnn.benchmark``, and it coordinates with
``trainer.deterministic``. A separate top-level key would silently bypass that.
"""

import sys
from pathlib import Path
from typing import Any, Literal

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from .export import to_onnx
from .training import SRDataModule, SRLightning

# Checkpoints saved before SRLightning.__init__ stopped flattening self.hparams for
# TensorBoard display (see its docstring) stored ONLY these two dataclasses' fields —
# under '/'-joined keys, e.g. 'training_config/example_input_shape/0' — plus incidental
# 'model/*'/'processor'/'criterion' entries that were never part of the loadable
# contract. _reconstruct_ckpt_hparams rebuilds exactly what current SRLightning saves.
_CKPT_HPARAM_KEYS = ("training_config", "eval_config")

# jsonargparse's own crash (see SRLightningCLI.parse_arguments) when a whole-dict
# CLI override collides with an already-set class_path/init_args value.
_JSONARGPARSE_INIT_ARGS_CRASH = "'dict' object has no attribute 'init_args'"


def _reconstruct_ckpt_hparams(hparams: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """Rebuild a checkpoint's training_config/eval_config overrides for ``--ckpt_path``.

    Tolerates both the current nested ``hyper_parameters`` format (each key already
    a plain dict, e.g. ``{"training_config": {...}}``) and the legacy '/'-flattened
    one older checkpoints carry (e.g. ``training_config/example_input_shape/0``).
    ``LightningCLI._parse_ckpt_path`` re-parses ``hyper_parameters`` verbatim as CLI
    options (``model.<key>``); a literal ``/`` in a key isn't a valid option-name
    character, so every legacy checkpoint failed to reload via ``--ckpt_path``
    (``fit`` resume, ``validate``, ``test``, ``export``) with a jsonargparse
    ``SystemExit``. Non-config entries (``model/*``, ``processor``, ``criterion``,
    ``_instantiator``) are dropped — they were TensorBoard-only in old checkpoints,
    and the ``--config`` file required alongside ``--ckpt_path`` already supplies
    those classes.

    Args:
        hparams: The checkpoint's raw ``hyper_parameters`` dict, in either format.
        sep: Separator ``SRLightning._flatten_hparams`` joins path segments with.

    Returns:
        A nested dict with at most ``training_config`` / ``eval_config`` keys,
        each a plain dict of field overrides, lists restored from their
        ``'0'``/``'1'``/... index encoding.
    """
    nested: dict[str, Any] = {}
    for compound_key, value in hparams.items():
        parts = compound_key.split(sep)
        if parts[0] not in _CKPT_HPARAM_KEYS:
            continue
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def _delistify(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = {k: _delistify(v) for k, v in node.items()}
        keys = list(node)
        if keys and keys == [str(i) for i in range(len(keys))]:
            return [node[str(i)] for i in range(len(keys))]
        return node

    return _delistify(nested)


class _ExportTrainer(Trainer):
    """``Trainer`` subclass adding an ``export`` method for the CLI subcommand.

    ``LightningCLI`` derives each subcommand's argument parser and help text
    from a method of this name on ``trainer_class`` (see
    ``_add_subcommands`` / ``_prepare_subcommand_parser`` in
    ``lightning.pytorch.cli``) — there is no supported way to register a
    subcommand that isn't a ``Trainer`` method. This subclass exists solely to
    give ``export`` that home; it changes no behavior for ``fit`` / ``validate``
    / ``test`` / ``predict``.
    """

    def export(
        self,
        model: SRLightning,
        datamodule: SRDataModule | None = None,
        output_path: str = "model.onnx",
        ckpt_path: str | None = None,
        opset_version: int = 17,
    ) -> None:
        """Export ``model`` to ONNX — thin wrapper around ``sisr.export.to_onnx``.

        Args:
            model: Model to export.
            datamodule: Unused — see the comment below.
            output_path: Destination ``.onnx`` file path.
            ckpt_path: Optional checkpoint whose weights are loaded into
                ``model`` before export.
            opset_version: ONNX opset to target.
        """
        # datamodule is unused: export needs no data, only the model's
        # architecture + weights, but LightningCLI always injects it (matching
        # every other subcommand's signature contract).
        to_onnx(model, output_path, ckpt_path=ckpt_path, opset_version=opset_version)


class SRLightningCLI(LightningCLI):
    """LightningCLI subclass that wires top-level optimizer/lr_scheduler config.

    With ``auto_configure_optimizers=False``, LightningCLI does *not* inject
    a default ``configure_optimizers`` into the model.  Instead we explicitly
    add the top-level args via ``add_optimizer_args(link_to=...)`` so they
    populate ``model.init_args.optimizer`` / ``model.init_args.lr_scheduler``
    as ``OptimizerCallable`` / ``LRSchedulerCallable``.  ``SRLightning``'s
    own ``configure_optimizers`` then constructs the optimizer (uniform or
    per-Conv2d ``param_groups`` depending on ``training_config.layer_lrs``).

    Also exposes a top-level ``matmul_precision`` YAML key that calls
    :func:`torch.set_float32_matmul_precision` in
    :meth:`before_instantiate_classes`. ``Trainer`` has no equivalent, unlike
    cuDNN autotuning which ``trainer.benchmark:`` already covers.

    Defaults ``trainer_class`` to :class:`_ExportTrainer`: :meth:`subcommands`
    unconditionally registers ``export``, and ``LightningCLI`` resolves every
    subcommand's parser via ``getattr(trainer_class, subcommand)`` regardless
    of which subcommand is actually invoked — so any caller constructing this
    class needs a ``trainer_class`` that has an ``export`` method, not just
    ``main()``. Callers may still override it explicitly (e.g. a project-specific
    ``Trainer`` subclass), as long as it also provides ``export``.
    """

    def __init__(self, *args, trainer_class: type[Trainer] = _ExportTrainer, **kwargs):
        # LightningCLI resolves a parser for *every* registered subcommand during
        # __init__, so a trainer_class without `export` fails as a bare AttributeError
        # from inside Lightning. Say what is actually wrong instead.
        if not hasattr(trainer_class, "export"):
            raise TypeError(
                f"{trainer_class.__name__} has no `export` method. SRLightningCLI "
                f"registers an `export` subcommand, and LightningCLI resolves a parser "
                f"for every subcommand at construction. Subclass _ExportTrainer, or add "
                f"an `export` method with the same signature."
            )
        super().__init__(*args, trainer_class=trainer_class, **kwargs)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parse CLI arguments, turning a known jsonargparse crash into an actionable error.

        A whole-dict/JSON CLI override of a ``data.*_dataset`` field (e.g.
        ``--data.train_dataset='{...}'`` or
        ``--data.train_dataset.init_args='{...}'``) only crashes when it
        collides with an *existing* ``class_path``/``init_args`` value for
        that field — typically one set by ``--config`` (both shipped
        templates set ``train_dataset``/``val_dataset``/``test_datasets``,
        so overriding those this way always collides). jsonargparse's merge
        logic then reaches for ``prev_val.init_args`` assuming ``prev_val``
        is a ``Namespace``, but every ``data.*_dataset`` field on
        :class:`~sisr.training.SRDataModule` is typed ``dict[str, Any]``
        (not the real dataset class, so datasets can be instantiated lazily —
        see that module's docstring), so ``prev_val`` is a plain ``dict`` and
        the attribute access raises. When there is no prior value (e.g.
        ``predict_dataset``, which neither template sets — the documented
        ``sisr predict`` workflow overrides it this exact way), the same
        whole-dict override parses fine.

        A prior version of this guard pre-scanned the raw CLI tokens and
        rejected *any* whole-dict override of these fields, which broke that
        working ``predict_dataset`` case. Catching the actual jsonargparse
        failure here instead only intercepts the forms that really crash.

        Args:
            parser: The top-level parser (unchanged, passed through).
            args: CLI arguments as given to ``LightningCLI(args=...)``.

        Raises:
            SystemExit: If parsing hits jsonargparse's
                ``prev_val.init_args`` crash on an already-``dict`` value.
        """
        try:
            super().parse_arguments(parser, args)
        except AttributeError as e:
            if _JSONARGPARSE_INIT_ARGS_CRASH not in str(e):
                raise
            raise SystemExit(
                "A CLI override passed a whole JSON/dict value for a data.*_dataset "
                "field that already has a class_path/init_args value (typically set by "
                "--config). jsonargparse cannot merge a whole dict into this "
                "dict[str, Any]-typed field in that case and fails with an internal "
                "AttributeError instead of a clean one. Edit the dataset spec directly "
                "in your YAML config instead of overriding an existing one from the CLI."
            ) from e

    def add_arguments_to_parser(self, parser):
        """Wire top-level ``optimizer:`` / ``lr_scheduler:`` / ``matmul_precision:`` keys.

        Subclass mode (``subclass_mode_model=True``) means the module's init args
        live under ``model.init_args.<arg>``, which is what these links must
        target. Subclass mode exists so a YAML can name its Lightning module by
        ``class_path`` — without it ``model_class`` is fixed and no config can
        select :class:`~sisr.training.SRGANLightning`.
        """
        parser.add_optimizer_args(link_to="model.init_args.optimizer")
        parser.add_lr_scheduler_args(link_to="model.init_args.lr_scheduler")
        parser.add_argument(
            "--matmul_precision",
            type=Literal["highest", "high", "medium"] | None,
            default=None,
            help=(
                "If set, calls torch.set_float32_matmul_precision(<value>) "
                "before instantiating classes. Use 'high' on Ampere+ GPUs to "
                "enable TF32 matmul kernels."
            ),
        )

    def before_instantiate_classes(self):
        """Apply process-global flags (``matmul_precision``) before class instantiation."""
        if self.subcommand is None:
            return
        precision = self.config[self.subcommand].matmul_precision
        if precision is not None:
            torch.set_float32_matmul_precision(precision)

    def _parse_ckpt_path(self) -> None:
        """Reload ``--ckpt_path`` hyperparameters via ``_reconstruct_ckpt_hparams`` first.

        Duplicates ``LightningCLI._parse_ckpt_path`` (``lightning/pytorch/cli.py``)
        with two substitutions. First, the raw ``hyper_parameters`` dict is rebuilt
        through :func:`_reconstruct_ckpt_hparams` before being handed to
        ``parse_object``, so both the current nested format and checkpoints saved
        before that fix (which reload with a jsonargparse ``SystemExit`` otherwise —
        see that function's docstring) work through every subcommand that accepts
        ``--ckpt_path`` (``fit`` resume, ``validate``, ``test``, ``export``).

        Second, under ``subclass_mode_model=True`` the module's fields are options
        at ``model.init_args.<key>``, not ``model.<key>``, so the reconstructed
        hparams are nested one level deeper — and that nested override MUST be
        parsed through the *subcommand's own* parser/config branch
        (``self._parser(subcommand)``, ``self.config[subcommand]``), never through
        the top-level subcommand-dispatching one (``self.parser``, ``self.config``),
        even though both expose the same ``model.init_args.<key>`` option path.
        The top-level route parses without a syntax error, but per-field the
        outcome is either a hard ``SystemExit`` (``model``, ``processor``: no
        default, required) or a silent revert to the bare-annotation default
        (any nested subclass field with a fallback, e.g. ``eval_config``) — see
        below for which and why.

        The reason: jsonargparse's subclass-merge machinery
        (``ActionTypeHint._check_type``) looks up the field's *previous* value as
        ``cfg.get(self.dest)`` — ``self.dest`` is the action's bare name (``"model"``),
        not a subcommand-qualified one. Reached through the subcommand's own
        parser, ``cfg`` is already scoped to that subcommand, so ``cfg.get("model")``
        finds the real, already-resolved value and merges our override's keys onto
        it, preserving every sibling key we don't mention (``model``, ``processor``,
        nested ``class_path`` identities) and any subclass-only default our override
        never names. Reached through the top-level parser, ``cfg`` is the
        *unscoped* full config, so the same ``cfg.get("model")`` misses the value
        (it actually lives at ``cfg["validate"]["model"]``) and silently gets
        "no previous value" instead of a lookup error — jsonargparse then rebuilds
        the field from schema defaults, which are ``None`` for ``model``/
        ``processor`` (no default; required) and the bare annotation type for any
        nested subclass field (``eval_config`` reverts to base ``SREvalConfig``).
        A resumed run would then either hit that "arguments are required"
        ``SystemExit`` outright, or — for a field with a valid bare-default
        fallback — silently drop back to the config file's/annotation's values with
        nothing in the logs to say so. Re-inlining this as ``self.parser`` "for
        simplicity" reintroduces exactly that.
        """
        if not self.config.get("subcommand"):
            return
        subcommand = self.config.subcommand
        ckpt_path = self.config[subcommand].get("ckpt_path")
        if not (ckpt_path and Path(ckpt_path).is_file()):
            return
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        hparams = _reconstruct_ckpt_hparams(ckpt.get("hyper_parameters", {}))
        if not hparams:
            return
        hparams = {"model": {"init_args": hparams}}
        try:
            self.config[subcommand] = self._parser(subcommand).parse_object(
                hparams, self.config[subcommand]
            )
        except SystemExit:
            sys.stderr.write("Parsing of ckpt_path hyperparameters failed!\n")
            raise

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Register ``export`` alongside the stock ``fit``/``validate``/``test``/``predict``.

        ``{"model", "datamodule"}`` is skipped from ``export``'s CLI-parsed
        arguments the same way ``fit`` skips them — both are already wired
        from the top-level ``model:`` / ``data:`` YAML keys, not re-specified
        per-subcommand.
        """
        subcommands = LightningCLI.subcommands()
        subcommands["export"] = {"model", "datamodule"}
        return subcommands


def main() -> None:
    """Console-script entrypoint.

    Invoked as ``sisr ...`` via the entry point registered in
    ``pyproject.toml``, and also as ``python -m sisr.cli ...``.
    ``freeze_support`` lives inside ``main`` (not the ``__main__`` block)
    so it also runs under the console-script path, which bypasses
    ``__main__``.
    """
    if sys.platform == "win32":
        from multiprocessing import freeze_support

        freeze_support()
    SRLightningCLI(
        model_class=SRLightning,
        datamodule_class=SRDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
