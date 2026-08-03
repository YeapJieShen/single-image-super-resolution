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

import re
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

# Matches a --data.<dataset field> flag, with an optional trailing .init_args,
# and (for the --flag=value form) captures the value. SRDataModule's dataset
# fields (train_dataset, val_dataset, predict_dataset, each test_datasets.<name>
# entry) are all typed dict[str, Any] for lazy per-stage instantiation — see
# _find_whole_dict_dataset_override's docstring for why that makes a whole-JSON
# CLI override crash inside jsonargparse instead of raising cleanly.
_DATASET_FIELD_FLAG = re.compile(
    r"^--data\.(train_dataset|val_dataset|predict_dataset|test_datasets\.[^.=]+)"
    r"(\.init_args)?(?:=(.*))?$",
    re.DOTALL,
)


def _find_whole_dict_dataset_override(args: list[str]) -> str | None:
    """Return the offending ``--data.*_dataset`` flag if ``args`` has a whole dict/JSON override.

    jsonargparse crashes with an unactionable ``AttributeError`` (raised deep
    inside its own ``_typehints.py``, from code that assumes the field's
    previous value is a ``Namespace`` with an ``.init_args`` attribute) when a
    CLI override's value looks like a subclass spec (starts with ``{``) and
    gets merged against an already-parsed value that is a plain ``dict`` —
    exactly what every ``data.*_dataset`` field on :class:`~sisr.training.SRDataModule`
    is, since each is typed ``dict[str, Any]`` rather than the real dataset
    class (so datasets can be instantiated lazily, only for the stages that
    need them). Both ``--data.train_dataset='{...}'`` (whole spec) and
    ``--data.train_dataset.init_args='{...}'`` (whole ``init_args``) hit this.

    Scanning here lets :meth:`SRLightningCLI.parse_arguments` raise an
    actionable error *before* handing off to jsonargparse's own parser,
    instead of letting the bare ``AttributeError`` surface.

    Args:
        args: Raw CLI argument tokens (as passed to ``LightningCLI(args=...)``,
            or ``sys.argv[1:]``). Handles both the ``--flag=value`` and the
            ``--flag value`` (separate token) forms.

    Returns:
        The matched flag (e.g. ``'--data.train_dataset.init_args'``), or
        ``None`` if no argument matches the crashing shape.
    """
    for i, arg in enumerate(args):
        match = _DATASET_FIELD_FLAG.match(arg)
        if match is None:
            continue
        value = match.group(3)
        if value is None:  # split "--flag value" form: value is the next token
            value = args[i + 1] if i + 1 < len(args) else None
        if value is not None and value.strip().startswith("{"):
            return arg.split("=", 1)[0]
    return None


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

        ``LightningCLI.__init__`` calls this before ``before_instantiate_classes``/
        ``instantiate_classes`` even run, so it is the earliest seam available —
        the crash :func:`_find_whole_dict_dataset_override` guards against
        happens *during* ``parser.parse_args``, deep inside jsonargparse
        itself, not at class-instantiation time.

        Args:
            parser: The top-level parser (unchanged, passed through).
            args: CLI arguments as given to ``LightningCLI(args=...)`` — a
                list of tokens, a dict/``Namespace`` (structured config, not
                scanned here since it never takes the crashing ``parse_args``
                code path), or ``None`` (read from ``sys.argv[1:]``).

        Raises:
            SystemExit: If ``args`` contains a whole-dict/JSON override of a
                ``data.*_dataset`` field.
        """
        arg_list = args if isinstance(args, list) else (sys.argv[1:] if args is None else None)
        if arg_list is not None:
            offending = _find_whole_dict_dataset_override(arg_list)
            if offending is not None:
                raise SystemExit(
                    f"{offending} passes a whole JSON/dict value. jsonargparse cannot "
                    f"merge that into the class_path/init_args shape of this "
                    f"dict[str, Any]-typed field — it fails with an internal "
                    f"AttributeError instead of a clean error. Edit data.train_dataset / "
                    f"data.val_dataset / data.test_datasets.<name> / data.predict_dataset "
                    f"directly in your YAML config instead of overriding it from the CLI."
                )
        super().parse_arguments(parser, args)

    def add_arguments_to_parser(self, parser):
        """Wire top-level ``optimizer:`` / ``lr_scheduler:`` / ``matmul_precision:`` keys.

        Non-subclass mode (``model_class=SRLightning`` fixed) means
        ``SRLightning``'s init args land at ``model.<arg>``, not
        ``model.init_args.<arg>`` — that's why the link targets omit
        ``init_args``.
        """
        parser.add_optimizer_args(link_to="model.optimizer")
        parser.add_lr_scheduler_args(link_to="model.lr_scheduler")
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
        with one substitution: the raw ``hyper_parameters`` dict is rebuilt through
        :func:`_reconstruct_ckpt_hparams` before being handed to
        ``parser.parse_object``, so both the current nested format and checkpoints
        saved before that fix (which reload with a jsonargparse ``SystemExit``
        otherwise — see that function's docstring) work through every subcommand
        that accepts ``--ckpt_path`` (``fit`` resume, ``validate``, ``test``, ``export``).
        """
        if not self.config.get("subcommand"):
            return
        ckpt_path = self.config[self.config.subcommand].get("ckpt_path")
        if not (ckpt_path and Path(ckpt_path).is_file()):
            return
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        hparams = _reconstruct_ckpt_hparams(ckpt.get("hyper_parameters", {}))
        if not hparams:
            return
        hparams = {self.config.subcommand: {"model": hparams}}
        try:
            self.config = self.parser.parse_object(hparams, self.config)
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
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
