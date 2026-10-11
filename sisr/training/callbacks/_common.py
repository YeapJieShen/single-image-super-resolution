"""Helpers every callback module shares: the typed module narrowing and the logger step axis."""

from typing import TYPE_CHECKING, cast

import lightning

if TYPE_CHECKING:  # `SRLightning` is referenced in annotations only, and a
    # runtime import would be circular -- it imports this module's package.
    from ..lightning_module import SRLightning


def _sr(pl_module: lightning.LightningModule) -> "SRLightning":
    """Narrow Lightning's hook parameter to the module these callbacks require.

    Every hook here is typed by Lightning as taking a bare ``LightningModule``,
    while everything they read -- ``eval_config``, ``scorer``, ``predict_rgb``,
    the provenance builders -- belongs to :class:`SRLightning`. Overriding a hook
    with a narrower parameter type is unsound and rejected, so the narrowing
    happens at the boundary instead, once, with a name.

    A cast, not an ``isinstance`` check: a module without those attributes
    already fails at the same call today, and turning that into a different
    failure is a behaviour change this seam has no business making.
    """
    return cast("SRLightning", pl_module)


def _logger_step(trainer: lightning.Trainer) -> int:
    """Return the step axis Lightning writes ``self.log`` metrics on.

    Deliberately not ``trainer.global_step``, which counts *optimizer* steps: a
    module under manual optimization with two optimizers (SRGAN steps D and G
    per batch) advances it twice per batch, so anything logged against it lands
    at 2x the step of every ``self.log`` scalar. Lightning maintains
    ``_batches_that_stepped`` for precisely this reason — "increased once per
    batch disregarding multiple optimizers on purpose for loggers" — and reads
    it back in ``logger_connector``; its own ``LearningRateMonitor`` and
    ``DeviceStatsMonitor`` reach for the same private attribute, which makes
    matching them here correct rather than a shortcut.
    """
    return trainer.fit_loop.epoch_loop._batches_that_stepped
