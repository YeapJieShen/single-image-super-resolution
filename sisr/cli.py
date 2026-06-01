"""LightningCLI entrypoint for SR training.

Usage:
    sisr fit --config templates/config.srcnn.template.yaml
    sisr test --config templates/config.srcnn.template.yaml \\
              --ckpt_path <best.ckpt>

The ``sisr`` console script is registered by ``pyproject.toml``; ``python -m
sisr.cli ...`` also works.

Top-level YAML keys ``optimizer:`` / ``lr_scheduler:`` are linked into
``model.init_args.optimizer`` / ``model.init_args.lr_scheduler`` by
``SRLightningCLI`` so :class:`~sisr.training.SRLightning` can build its
``configure_optimizers`` from them while keeping the YAML symmetric across
architectures.
"""
import sys

from lightning.pytorch.cli import LightningCLI

from .training import SRDataModule, SRLightning


class SRLightningCLI(LightningCLI):
    """LightningCLI subclass that wires top-level optimizer/lr_scheduler config.

    With ``auto_configure_optimizers=False``, LightningCLI does *not* inject
    a default ``configure_optimizers`` into the model.  Instead we explicitly
    add the top-level args via ``add_optimizer_args(link_to=...)`` so they
    populate ``model.init_args.optimizer`` / ``model.init_args.lr_scheduler``
    as ``OptimizerCallable`` / ``LRSchedulerCallable``.  ``SRLightning``'s
    own ``configure_optimizers`` then constructs the optimizer (uniform or
    per-Conv2d ``param_groups`` depending on ``training_config.layer_lrs``).
    """

    def add_arguments_to_parser(self, parser):
        # Non-subclass mode (model_class=SRLightning fixed): SRLightning's
        # init args land at model.<arg>, not model.init_args.<arg>.
        parser.add_optimizer_args(link_to="model.optimizer")
        parser.add_lr_scheduler_args(link_to="model.lr_scheduler")


def main() -> None:
    # `freeze_support` lives inside `main` (not the __main__ block) so it
    # also runs when invoked via the `sisr` console script declared in
    # pyproject.toml — that entry point bypasses __main__.
    if sys.platform == 'win32':
        from multiprocessing import freeze_support
        freeze_support()
    SRLightningCLI(
        model_class=SRLightning,
        datamodule_class=SRDataModule,
        save_config_kwargs={'overwrite': True},
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    main()
