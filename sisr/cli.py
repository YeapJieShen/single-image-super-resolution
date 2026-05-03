"""LightningCLI entrypoint for SR training.

Usage:
    python -m sisr.cli fit --config templates/config.srcnn.template.yaml

LightningCLI auto-generates the rest of the surface: ``--print_config``,
``--model.lr=1e-3``, ``--trainer.max_steps=500000``, etc.
"""
import sys

from lightning.pytorch.cli import LightningCLI

from .training import SRDataModule, SRLightning


def main() -> None:
    LightningCLI(
        model_class=SRLightning,
        datamodule_class=SRDataModule,
        save_config_kwargs={'overwrite': True},
    )


if __name__ == '__main__':
    if sys.platform == 'win32':
        from multiprocessing import freeze_support
        freeze_support()
    main()
