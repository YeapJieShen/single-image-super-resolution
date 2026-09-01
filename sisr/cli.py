r"""LightningCLI entrypoint for SR training.

Usage:
    sisr fit --config templates/config.srcnn.template.yaml
    sisr test --config templates/config.srcnn.template.yaml \
              --ckpt_path <best.ckpt>
    sisr export --config templates/config.srcnn.template.yaml \
                --output_path model.onnx --ckpt_path <best.ckpt>

The ``sisr`` console script is registered by ``pyproject.toml``; ``python -m
sisr.cli ...`` also works.

Top-level ``optimizer:`` / ``lr_scheduler:`` are linked into
``model.init_args.*`` so ``SRLightning.configure_optimizers`` can build from
them and the YAML stays symmetric across architectures.

Top-level ``matmul_precision:`` calls
:func:`torch.set_float32_matmul_precision` once at startup (``'high'`` enables
TF32 matmul kernels on Ampere+).

``sisr export`` wraps :func:`sisr.export.to_onnx` -- see that module for its
SRCNN limitation. Needs the ``export`` extra.

**cuDNN autotuning is deliberately not exposed.** ``trainer.benchmark:``
already assigns it and coordinates with ``trainer.deterministic``; a separate
top-level key would silently bypass that.
"""

import sys
from pathlib import Path
from typing import Any, Literal

import torch
from jsonargparse import set_parsing_settings
from lightning.pytorch import Trainer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from .export import to_onnx
from .training import SRDataModule, SREvalConfig, SRLightning, SRTrainingConfig

# The only keys a checkpoint's hyper_parameters can restore. Legacy checkpoints also
# carry 'model/*'/'processor'/'criterion', which were TensorBoard-only and never part
# of the loadable contract.
_CKPT_HPARAM_KEYS = ("training_config", "eval_config")

# jsonargparse's own crash (see SRLightningCLI.parse_arguments) when a whole-dict
# CLI override collides with an already-set class_path/init_args value.
_JSONARGPARSE_INIT_ARGS_CRASH = "'dict' object has no attribute 'init_args'"


def _reconstruct_ckpt_hparams(hparams: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """Rebuild a checkpoint's training_config/eval_config overrides for ``--ckpt_path``.

    Accepts both the current nested format and the legacy ``'/'``-flattened one
    (``training_config/example_input_shape/0``). ``LightningCLI._parse_ckpt_path``
    re-parses ``hyper_parameters`` verbatim as ``model.<key>`` CLI options, and
    ``/`` is not a valid option-name character -- so every legacy checkpoint died
    with a jsonargparse ``SystemExit`` on any ``--ckpt_path`` subcommand.
    Non-config entries are dropped; the ``--config`` required alongside
    ``--ckpt_path`` already supplies those classes.

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
    populate ``model.init_args.optimizer`` / ``.lr_scheduler``, which
    ``SRLightning.configure_optimizers`` then builds from. Also exposes
    top-level ``matmul_precision``, which ``Trainer`` has no equivalent for.

    **``trainer_class`` must have an ``export`` method.** :meth:`subcommands`
    registers ``export`` unconditionally and ``LightningCLI`` resolves *every*
    subcommand's parser at construction, whichever one is invoked. Hence the
    :class:`_ExportTrainer` default; override it with anything that also
    provides ``export``.
    """

    def __init__(
        self, *args: Any, trainer_class: type[Trainer] = _ExportTrainer, **kwargs: Any
    ) -> None:
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
        # Must precede super().__init__(), which builds every subcommand's parser.
        # jsonargparse treats a pure dataclass as a *closed* type, so a dotted
        # override like --model.init_args.eval_config.crop_border=0 rebuilds the
        # field from the bare annotation and discards the architecture's subclass
        # with every subclass-only default. Naming the two bases re-enables
        # subclasses for all descendants; disabling the selector instead would
        # change parsing for every dataclass field in Lightning too.
        set_parsing_settings(subclasses_enabled=[SREvalConfig, SRTrainingConfig])
        # mypy cannot rule out an *args long enough to reach the parent's own
        # trainer_class. No caller does that, and forbidding it would mean restating
        # LightningCLI's whole signature here.
        super().__init__(*args, trainer_class=trainer_class, **kwargs)  # type: ignore[misc]

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parse CLI arguments, turning a known jsonargparse crash into an actionable error.

        A whole-dict override of a ``data.*_dataset`` field crashes **only**
        when it collides with an existing ``class_path``/``init_args`` value,
        which ``--config`` always supplies for ``train``/``val``/``test``.
        jsonargparse's merge then reads ``prev_val.init_args`` assuming a
        ``Namespace``, but those fields are typed ``dict[str, Any]`` (so
        datasets stay lazily instantiable), and the attribute access raises.
        With no prior value it parses fine -- which is exactly the documented
        ``sisr predict`` workflow overriding ``predict_dataset``.

        **Catch the failure, do not pre-scan.** An earlier guard rejected every
        whole-dict override of these fields and broke that working case.

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

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Wire top-level ``optimizer:`` / ``lr_scheduler:`` / ``matmul_precision:`` keys.

        Subclass mode puts init args under ``model.init_args.<arg>``, which is
        what these links target. It exists so a YAML can name its Lightning
        module by ``class_path``; without it ``model_class`` is fixed and no
        config could select ``SRGANLightning``.
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

    def before_instantiate_classes(self) -> None:
        """Apply process-global flags (``matmul_precision``) before class instantiation."""
        if self.subcommand is None:
            return
        precision = self.config[self.subcommand].matmul_precision
        if precision is not None:
            torch.set_float32_matmul_precision(precision)

    def _parse_ckpt_path(self) -> None:
        """Reload ``--ckpt_path`` hyperparameters via ``_reconstruct_ckpt_hparams`` first.

        Duplicates ``LightningCLI._parse_ckpt_path`` with two substitutions.
        First, ``hyper_parameters`` is rebuilt through
        :func:`_reconstruct_ckpt_hparams`, so legacy checkpoints reload rather
        than exiting.

        Second -- **and this is the trap** -- under ``subclass_mode_model=True``
        the reconstructed override sits at ``model.init_args.<key>`` and MUST be
        parsed through the *subcommand's own* parser and config branch
        (``self._parser(subcommand)``, ``self.config[subcommand]``), never the
        top-level ``self.parser`` / ``self.config``, though both expose the same
        option path.

        jsonargparse's subclass merge (``ActionTypeHint._check_type``) looks up
        the previous value as ``cfg.get(self.dest)``, and ``self.dest`` is the
        bare name, never subcommand-qualified. Scoped, the lookup hits and merges
        onto the resolved value, keeping sibling keys and subclass-only defaults.
        Unscoped, the value really lives at ``cfg["validate"]["model"]`` and the
        miss reads as *no previous value* rather than an error -- so jsonargparse
        rebuilds from schema defaults.

        **A resumed run then either dies on "arguments are required"
        (``model``/``processor``, which have no default) or silently reverts a
        nested subclass field to its bare annotation, with nothing in the logs.**
        Re-inlining this as ``self.parser`` "for simplicity" reintroduces it.
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

        ``export`` skips ``{"model", "datamodule"}`` as ``fit`` does: both come
        from the top-level ``model:`` / ``data:`` keys, not per-subcommand.
        """
        subcommands = LightningCLI.subcommands()
        subcommands["export"] = {"model", "datamodule"}
        return subcommands


def main() -> None:
    """Console-script entrypoint.

    ``freeze_support`` sits inside ``main``, not the ``__main__`` block, so it
    also runs under the console-script path, which bypasses ``__main__``.
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
