"""The progress bar callback."""

import lightning
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
)

from ...metrics.perceptual import PERCEPTUAL_METRICS

#: Metric families the progress bar can be narrowed to. ``loss`` covers every
#: ``loss/...`` tag (train, val, and the discriminator's on an adversarial run);
#: the rest are the scored families, taken from the metric layer rather than
#: restated, so a new perceptual metric is selectable the day it exists.
PROG_BAR_FAMILIES: frozenset[str] = frozenset({"loss", "psnr", "ssim"}) | frozenset(
    PERCEPTUAL_METRICS
)


class SRProgressBar(TQDMProgressBar):
    """Progress bar showing the whole run name, and only the metrics asked for.

    Exists for two reasons, both of which need a callback this project owns --
    there was none, so neither was reachable from a config.

    **The run name.** Lightning's ``get_standard_metrics`` truncates
    unconditionally::

        if isinstance(version, str):
            version = version[-4:]      # "vgg54-scaled" -> "aled"

    That rule is written for integer-ish versions (``version_10`` -> ``_10``).
    This project names runs after what they *are*, so two runs differing
    anywhere but their last four characters are indistinguishable in the bar --
    exactly when you most want to know which one you are watching.

    **The metric set.** Every entry in the bar comes from a ``prog_bar=True``
    at a log call, spread over five sites, with no way to turn any of them off.
    On an adversarial run that is five metrics plus ``v_num``, which wraps the
    line and pushes PSNR/SSIM to the far end. Placement was thought about --
    ``prog_bar`` is a display choice and stays with the caller -- but
    configurability was not.

    **Nothing about what is logged changes.** ``prog_bar`` controls display
    only; every metric still reaches TensorBoard and ``metrics.csv`` exactly as
    before. This changes what a human reads mid-run, never what a result is
    computed from.

    Args:
        metrics: Families to display, from :data:`PROG_BAR_FAMILIES`. ``None``
            (default) shows everything logged with ``prog_bar=True``, so no
            shipped config changes what it displays. ``['psnr', 'ssim']`` gives
            the scored view; ``['loss']`` suits an adversarial run, where
            PSNR/SSIM get worse by design. ``v_num`` is not a family and always
            survives.
        refresh_rate: Batches between bar refreshes, as ``TQDMProgressBar``.
        process_position: Row offset when several bars are shown.

    Raises:
        ValueError: If any entry of *metrics* is not a known family.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        refresh_rate: int = 1,
        process_position: int = 0,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        if metrics is not None:
            unknown = [m for m in metrics if m not in PROG_BAR_FAMILIES]
            if unknown:
                raise ValueError(
                    f"SRProgressBar.metrics entries must be one of "
                    f"{sorted(PROG_BAR_FAMILIES)}; got unsupported {unknown}. A typo would "
                    f"otherwise show an empty bar, which reads as 'nothing is being logged' "
                    f"rather than 'that family does not exist'. Fix the callback's "
                    f"init_args.metrics in your YAML."
                )
        self.metrics = metrics

    def get_metrics(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        """The bar's items: the full version, plus the selected metric families.

        Args:
            trainer: The running trainer.
            pl_module: The module being trained. Unused; part of the hook.

        Returns:
            What to render, in the order Lightning would have rendered it.
        """
        items = super().get_metrics(trainer, pl_module)

        # Undo get_standard_metrics' truncation. Lightning derives v_num from a
        # private helper that reports a version only when every logger agrees on
        # one; that rule is reproduced here from the public `.version`
        # attribute rather than by importing the private symbol, which is the
        # same cross-boundary fragility this package avoids in its own modules
        # -- and worse here, since it can vanish in a Lightning minor release.
        # If v_num is absent, or the loggers disagree, whatever Lightning
        # decided stands.
        if "v_num" in items:
            versions = {logger.version for logger in trainer.loggers}
            if len(versions) == 1 and (version := versions.pop()) not in ("", None):
                items["v_num"] = version

        if self.metrics is None:
            return items
        allowed = set(self.metrics)
        # A key with no "/" is not one of our tags (v_num, and anything
        # Lightning adds later); those are never filtered out.
        return {k: v for k, v in items.items() if "/" not in k or k.split("/")[0] in allowed}
