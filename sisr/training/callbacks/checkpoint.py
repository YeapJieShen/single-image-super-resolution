"""Checkpoint callbacks: rolling-window saves, full ``.ckpt`` and bare-weights presets."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from weakref import proxy

import lightning
import torch
import torch.nn.functional
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from ... import artifacts
from ...metrics.perceptual import PERCEPTUAL_METRICS
from ...metrics.scoring import expected_tags, perceptual_tag
from ..metadata import build_component_metadata, build_metadata
from ._common import _logger_step, _sr

if TYPE_CHECKING:
    # The mixin reads `monitor`, `dirpath`, `FILE_EXTENSION` and `_remove_checkpoint`
    # off the ModelCheckpoint it is combined with, and chains `on_train_start` through
    # it. Naming that base for the checker alone keeps the runtime MRO exactly as the
    # docstring below describes -- a mixin over `object`, so `super()` still lands on
    # ModelCheckpoint rather than ahead of it.
    _RollingBase = ModelCheckpoint
else:
    _RollingBase = object


class _RollingSaveMixin(_RollingBase):
    """Oldest-first deletion for the no-monitor (rolling) case.

    Lightning refuses ``ModelCheckpoint(monitor=None, save_top_k=N>1)``
    ("No quantity for top_k to track"), so a rolling window cannot be
    expressed through ``save_top_k`` at all. The legal construction is
    ``monitor=None, save_top_k=-1`` (keep everything) with the window
    enforced here. Verified on Lightning 2.6.5.

    Shared by :class:`SRCheckpoint` and :class:`SRWeightsCheckpoint`, which are
    siblings rather than parent/child, so one ``super()`` chain cannot serve
    both. **The bookkeeping lives in :meth:`_enforce_rolling_window`, not in a
    ``_save_checkpoint`` of this mixin**: ``SRWeightsCheckpoint`` overrides
    ``_save_checkpoint`` without calling ``super()``, so a mixin version would
    be shadowed and silently never run. Each subclass calls
    :meth:`_enforce_rolling_window` explicitly.

    The window itself is never persisted in ``state_dict`` — it is rebuilt from
    the files already on disk when a run resumes (:meth:`on_train_start`), which
    is what keeps ``fit --ckpt_path`` from orphaning the pre-resume saves.
    """

    def _init_rolling(self, keep_last: int, filename_prefix: str) -> None:
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1; got {keep_last}")
        self.keep_last = keep_last
        self._rolling: list[str] = []
        # Rolling filenames are exactly f"{filename_prefix}-{step}{FILE_EXTENSION}",
        # so this matches every file THIS callback writes and nothing else: a
        # sibling callback in the same dirpath differs in prefix or extension, and
        # `last.ckpt`/a monitored callback's metric-bearing name never match.
        self._rolling_pattern = re.compile(
            rf"{re.escape(filename_prefix)}_s(\d+){re.escape(self.FILE_EXTENSION)}"
        )

    def on_train_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Seed the rolling window from disk when this run is resuming.

        Nothing persists the window: ``ModelCheckpoint.state_dict`` carries
        ``monitor``/``best_*``/``kth_*``/``dirpath``/``last_model_path`` only. So
        without this, a ``fit --ckpt_path`` resume starts with an empty window and
        every pre-resume file is orphaned — never counted, never deleted, up to
        ``keep_last`` per callback per resume (three rolling callbacks run in the
        shipped adversarial template).

        Only a resume seeds. A fresh run into a populated ``dirpath`` leaves those
        files alone rather than adopting — and eventually deleting — checkpoints it
        did not write.

        Runs here rather than in ``setup`` because ``trainer.ckpt_path`` is not
        assigned until the checkpoint connector restores state, which is after
        every ``setup`` hook.

        A ``dirpath`` that does not exist yet is a no-op, not an error: resuming
        into a **fresh** run directory (branching a run, or restarting a crashed
        one so the old directory survives as evidence) reaches this hook before
        Lightning has created it.

        Args:
            trainer: The active trainer — supplies ``ckpt_path``.
            pl_module: Unused; part of the hook signature.
        """
        super().on_train_start(trainer, pl_module)
        if self.monitor is not None or not trainer.ckpt_path or not self.dirpath:
            return
        # Lightning creates dirpath lazily, in its IO plugin, at the first save --
        # so resuming into a run directory that has not saved yet finds nothing to
        # list. Missing and empty mean the same thing here: no window to adopt.
        directory = Path(self.dirpath)
        if not directory.is_dir():
            return
        found = sorted(
            (int(match.group(1)), str(path))
            for path in directory.iterdir()
            if (match := self._rolling_pattern.fullmatch(path.name))
        )
        self._rolling = [path for _, path in found]

    def _enforce_rolling_window(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Track *filepath* and drop the oldest save(s) beyond ``keep_last``.

        No-op when ``monitor`` is set — Lightning's own top-k selection owns
        retention then, unmodified.

        Args:
            trainer: The active trainer, forwarded to ``_remove_checkpoint``.
            filepath: Path just written by the caller's ``_save_checkpoint``.
        """
        if self.monitor is not None:
            return
        self._rolling.append(filepath)
        while len(self._rolling) > self.keep_last:
            self._remove_checkpoint(trainer, self._rolling.pop(0))


class _SRCheckpointBase(_RollingSaveMixin, ModelCheckpoint):
    """Everything :class:`SRCheckpoint` and :class:`SRWeightsCheckpoint` share.

    They were siblings under ``ModelCheckpoint`` rather than parent and child,
    which forced the filename construction, the rolling-mode ``save_top_k``
    rule and the monitor validation to be written twice or reached through a
    free function taking the caller's identity as an argument. They are two
    adapters over one module, so that module now exists.

    Subclasses supply defaults and, where they differ, a ``_save_checkpoint``
    payload. Nothing here decides what a checkpoint *contains*.
    """

    def _monitor_candidates(self, trainer: lightning.Trainer) -> dict[str, torch.Tensor]:
        """Substitute the batch-counted axis for ``{step}`` in every checkpoint name.

        Lightning fills ``{step}`` from ``trainer.global_step``, which counts
        *optimizer* steps. Under manual optimization with two optimizers (an
        adversarial run steps D and G per batch) that is **2x** the batch count,
        while every metric the checkpoint would be read against lands on
        ``_batches_that_stepped``. The result is an artifact whose name cannot be
        located on any curve it should be comparable with (a golden SRGAN run held
        ``sr-400000`` against a ``metrics.csv`` maxing at 199,999), while every
        automatic-optimization run was 1:1.

        Overriding here rather than at each call site is deliberate: Lightning
        routes filename formatting, top-k bookkeeping and the ``last``/``every_n``
        paths through this one method, so a single substitution keeps them
        consistent with each other. It is also invisible where the counters
        already agree, which is every SRResNet/SRCNN run.

        The name is only half of it — see :meth:`_save_checkpoint`, which records
        *both* counters in the artifact's own metadata so the axis survives the
        file being copied away from its run directory.

        Args:
            trainer: The active trainer.

        Returns:
            Lightning's candidates dict with ``step`` on the batch axis.
        """
        candidates = super()._monitor_candidates(trainer)
        candidates["step"] = torch.tensor(_logger_step(trainer))
        return candidates

    def __init__(
        self,
        monitor_metric: str | None = "psnr/val/RGB",
        save_top_k: int = 3,
        dirpath: str | None = None,
        filename_prefix: str | None = None,
        mode: str = "max",
        keep_last: int = 3,
        **kwargs: Any,
    ):
        if monitor_metric is None:
            # Lightning refuses save_top_k > 1 with nothing to rank by; the
            # window is enforced by _RollingSaveMixin instead.
            save_top_k = -1
        super().__init__(
            monitor=monitor_metric,
            mode=mode,
            save_top_k=save_top_k,
            dirpath=dirpath,
            auto_insert_metric_name=False,
            **kwargs,
        )
        self._keep_last = keep_last
        self._explicit_prefix = filename_prefix
        # The *raw* dirpath, because setup() resolves self.dirpath against the
        # trainer -- so when one callback probes another, one side may be resolved
        # and the other not, and comparing those is comparing different things.
        self._explicit_dirpath = None if dirpath is None else str(Path(dirpath).resolve())
        # A usable name before setup() can derive the real one -- direct
        # construction in a test never reaches setup.
        self._apply_naming(filename_prefix or self.DEFAULT_PREFIX)

    #: Placeholder identity used until setup() can derive the real one. Distinct
    #: per subclass so two sinks sharing a dirpath cannot collide before then.
    DEFAULT_PREFIX: str = "sr"

    def _apply_naming(self, prefix: str) -> None:
        """Set the filename template and the rolling window's matching pattern together.

        They must agree: the window finds this callback's own files by matching
        exactly what it writes, so a prefix change that touched only one of them
        would either orphan every save or adopt a sibling callback's.

        Args:
            prefix: Identity portion of the filename, without the step.
        """
        self.filename = f"{prefix}_s{{step}}"
        self._init_rolling(self._keep_last, prefix)

    def _resolved_prefix(self, pl_module: lightning.LightningModule) -> str:
        """The identity this callback will write under, whether or not setup() has run.

        Siblings are compared before Lightning has necessarily set them all up, so
        this cannot be read off ``self.filename`` -- that still holds
        ``DEFAULT_PREFIX`` on anything yet to reach setup. Returns the explicit
        ``filename_prefix`` if given, else the stem derived from what this
        callback saves.
        """
        if self._explicit_prefix is not None:
            return self._explicit_prefix
        return artifacts.stem(self._describe(pl_module))

    def _reject_filename_collision(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Raise if a sibling callback would write this callback's exact filename.

        Distinct default prefixes keep the two *classes* apart, and their docstrings
        say so -- but that reasoning never covered two instances of one class, which
        is what both shipped templates configured: one on ``psnr/val/RGB`` and one on
        ``ssim/val/RGB``, deriving the same model-provenance stem. Lightning does not
        refuse it. It keeps one and appends ``-v1`` to the other, **assigned by save
        order**, so which file wins the bare name is stable across neither runs nor
        Lightning versions, and neither name says which metric it is best on.

        The comparison is on the whole filename rather than the prefix: ``.ckpt``
        beside ``.safetensors`` under one prefix is the intended, shipped pair.

        Refusing is deliberately preferred to disambiguating automatically. Any
        automatic rule has to decide which sibling keeps the bare name, and every
        answer to that is either order-dependent -- the defect itself -- or silently
        renames published artifacts when a monitor changes.

        Args:
            trainer: The active trainer, whose ``checkpoint_callbacks`` are the
                siblings this could collide with.
            pl_module: The model being trained.

        Raises:
            MisconfigurationException: If another :class:`_SRCheckpointBase` in the
                callback list would write the same name into the same directory.
        """
        prefix = self._resolved_prefix(pl_module)
        mine = (self._explicit_dirpath, prefix, self.FILE_EXTENSION)
        for other in trainer.checkpoint_callbacks:
            if other is self or not isinstance(other, _SRCheckpointBase):
                continue
            try:
                theirs = (
                    other._explicit_dirpath,
                    other._resolved_prefix(pl_module),
                    other.FILE_EXTENSION,
                )
            except Exception:
                # A sibling whose own configuration is broken cannot be probed. Its
                # setup() reports that precisely; masking it with a collision error
                # raised from over here would be strictly worse than skipping it.
                continue
            if theirs != mine:
                continue
            raise MisconfigurationException(
                f"`{type(self).__name__}(monitor_metric={self.monitor!r})` and "
                f"`{type(other).__name__}(monitor_metric={other.monitor!r})` would both "
                f"write `{prefix}_s{{step}}{self.FILE_EXTENSION}` into the same directory. "
                "Lightning keeps one and suffixes the other `-v1`, assigned by save "
                "order, so which artifact is which is neither stable across runs nor "
                "readable from its name. HINT: give one of them a distinct "
                "`filename_prefix`."
            )

    def _describe(self, pl_module: lightning.LightningModule) -> dict[str, Any]:
        """Provenance for whatever this callback saves. Overridden per component."""
        return build_metadata(_sr(pl_module))

    def _validate_monitor(self, pl_module: lightning.LightningModule) -> None:
        """Raise if this callback's ``monitor``/``mode`` pair is unloggable or inverted.

        The direction half exists because both subclasses default to ``mode='max'``,
        which is right for PSNR/SSIM and exactly inverted for LPIPS/DISTS — the
        failure is silent, keeping the worst checkpoint of the run with nothing in
        the logs or filenames to indicate it. Which perceptual metrics are
        lower-is-better comes from ``PERCEPTUAL_METRICS`` rather than being
        restated here, so a future higher-is-better perceptual metric would still
        be checked in the right direction.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, or if ``mode`` disagrees
                with that metric's direction. ``monitor=None`` is rolling mode,
                which monitors nothing and is always valid.
        """
        if self.monitor is None:
            return

        eval_config = _sr(pl_module).eval_config
        # "Which tags will this run log?" is exactly expected_tags' question, and
        # it is the grammar's owner. Re-deriving it here is what let the emitted
        # set and the accepted set drift apart, silently.
        valid_metrics = set(expected_tags(eval_config, "val"))
        lower_better = {
            perceptual_tag(name, "val")
            for name in eval_config.perceptual_keys
            if PERCEPTUAL_METRICS[name]
        }
        label = type(self).__name__
        if self.monitor not in valid_metrics:
            raise MisconfigurationException(
                f"`{label}(monitor_metric={self.monitor!r})` does not match any "
                f"metric `SRLightning` will log: {sorted(valid_metrics)}. HINT: check "
                f"`eval_config.psnr_channels` / `eval_config.separate_psnr`, "
                f"`eval_config.ssim_channels`, or `eval_config.perceptual_metrics`."
            )

        wanted = "min" if self.monitor in lower_better else "max"
        if self.mode != wanted:
            direction = "lower-is-better" if wanted == "min" else "higher-is-better"
            raise MisconfigurationException(
                f"`{label}(monitor_metric={self.monitor!r}, mode={self.mode!r})` monitors a "
                f"{direction} metric in the wrong direction — it would keep the worst "
                f"checkpoint of the run, silently. Set mode={wanted!r}."
            )

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Validate ``monitor``, derive the filename, and refuse a colliding sibling.

        Lightning itself only raises ``MisconfigurationException`` for a
        missing monitor once the val loop has already run at least once
        (``ModelCheckpoint._save_topk_checkpoint``, gated on
        ``val_loop._has_run``) — with a long ``val_check_interval`` that is a
        late, expensive-to-reach crash. This moves the same failure to startup.

        Args:
            trainer: The active trainer.
            pl_module: The model being trained; must expose ``eval_config``.
            stage: Lightning trainer stage. Only ``"fit"`` is checked — the
                monitored tags are val-loop metrics, so they are never logged
                under ``validate``/``test``/``predict`` and demanding them
                there would reject configs valid for the stage being run.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, if ``mode`` disagrees with
                that metric's direction, or if a sibling callback would write this
                callback's exact filename.
        """
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        self._validate(pl_module)
        # Derived here rather than in __init__ because it comes from the module,
        # which does not exist when a callback is built from YAML -- and after
        # _validate, because it reads whatever that just checked.
        if self._explicit_prefix is None:
            self._apply_naming(artifacts.stem(self._describe(pl_module)))
        self._reject_filename_collision(trainer, pl_module)

    def _validate(self, pl_module: lightning.LightningModule) -> None:
        """Everything that must hold before this callback can name or write anything."""
        self._validate_monitor(pl_module)


class SRCheckpoint(_SRCheckpointBase):
    """Model checkpoint that monitors a super-resolution quality metric.

    Wraps :class:`~lightning.pytorch.callbacks.ModelCheckpoint` with
    ``mode='max'`` (PSNR and SSIM are both higher-is-better) and a descriptive
    filename pattern.

    **The metric label is sanitised (``/`` -> ``_``) and
    ``auto_insert_metric_name`` is disabled**, so a ``/``-bearing
    ``monitor_metric`` cannot leak into the filename template. Lightning does
    no sanitisation itself, and a raw ``/`` there makes ``TorchCheckpointIO``
    silently create a nested directory tree per save instead of a flat file.
    The ``metrics`` dict lookup supplying the *value* is unaffected.

    Args:
        monitor_metric: The validation metric to monitor. Any ``psnr/val/{key}``
            or ``ssim/val/{key}`` logged by the lightning module (e.g.
            ``"psnr/val/Y"``, ``"psnr/val/YCbCr"``, ``"ssim/val/RGB"``).
            ``None`` selects rolling mode instead: no metric is tracked at
            all, and the last ``keep_last`` checkpoints (by step) are kept.
            Needed because under an adversarial objective (e.g. SRGAN) PSNR
            and SSIM get worse by design, so top-k selection on either would
            keep the LEAST adversarial checkpoint of the run — typically one
            from the first few thousand steps.
        save_top_k: Number of best checkpoints to keep. Ignored (forced to
            ``-1``, i.e. keep-everything) when ``monitor_metric`` is ``None``
            — Lightning refuses ``save_top_k > 1`` with no monitor outright,
            so ``keep_last`` enforces the window instead.
        dirpath: Directory to save checkpoints.
        filename_prefix: Prefix for checkpoint filenames.
        keep_last: In rolling mode, number of most recent checkpoints to
            retain; oldest is deleted first. Ignored when ``monitor_metric``
            is set. Defaults to ``3``.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Save via :class:`~lightning.pytorch.callbacks.ModelCheckpoint`, then enforce rolling.

        Args:
            trainer: The active trainer.
            filepath: Destination path, already formatted by ``ModelCheckpoint``.
        """
        super()._save_checkpoint(trainer, filepath)
        # Any edit here must preserve this call -- see _RollingSaveMixin's
        # docstring for why a rolling window can't be expressed via save_top_k alone.
        self._enforce_rolling_window(trainer, filepath)


class SRWeightsCheckpoint(_SRCheckpointBase):
    """Model checkpoint that saves bare, optimizer-free weights as ``.safetensors``.

    The distributable sibling to :class:`SRCheckpoint`: that writes resumable
    ``.ckpt`` files (optimizer moments, scheduler, callback state); this writes
    only ``getattr(pl_module, attribute).state_dict()`` plus provenance.
    ``attribute='model'`` (default) saves the bare ``SRModel`` — so keys carry
    no ``model.`` prefix — described by
    :func:`~sisr.training.metadata.build_metadata`. Any other attribute uses
    :func:`~sisr.training.metadata.build_component_metadata` instead, because
    ``build_metadata``'s generator-scoped fields (``io.scale``, ``criterion``,
    ``eval_config``) would describe something the file does not contain.

    Everything else — top-k selection, monitor validation, filename formatting
    — is inherited unchanged; **only :meth:`_save_checkpoint` is overridden**,
    to swap the payload.

    That method is private Lightning API, unlike the rest of this class's
    surface. ``test_sr_weights_checkpoint_writes_bare_payload_via_real_fit``
    guards it: if a future Lightning routes saves elsewhere, that test fails
    loudly rather than letting saves fall back to full, optimizer-bearing
    checkpoints under a misleadingly bare ``.safetensors`` extension.

    ``FILE_EXTENSION`` plus a distinct default ``filename_prefix`` keep this
    callback's deletion passes off :class:`SRCheckpoint`'s files when both
    share a ``dirpath`` — each only names and deletes files matching its own
    ``filename``/``FILE_EXTENSION`` pair. That separates the two *classes*, not
    two instances of one: a pair differing only in ``monitor_metric`` derives
    the same stem and collides, which
    :meth:`_SRCheckpointBase._reject_filename_collision` refuses at startup.

    Args:
        monitor_metric: The validation metric to monitor — same contract as
            :class:`SRCheckpoint`, including the ``None`` rolling-mode meaning.
        save_top_k: Number of best weight files to keep. Ignored (forced to ``-1``)
            when ``monitor_metric`` is ``None`` — same contract as :class:`SRCheckpoint`.
        dirpath: Directory to save weight files.
        filename_prefix: Prefix for weight filenames. Defaults to ``'sr-weights'``,
            distinct from ``SRCheckpoint``'s ``'sr'`` default.
        mode: ``'max'`` or ``'min'`` — same contract as :class:`SRCheckpoint`.
        keep_last: In rolling mode, number of most recent weight files to retain.
            Ignored when ``monitor_metric`` is set. Defaults to ``3``.
        attribute: Name of the ``pl_module`` attribute whose ``state_dict()`` is saved.
            Defaults to ``'model'`` (the SR model). Any other value saves that named
            component instead (e.g. ``'discriminator'`` for an adversarial run's
            discriminator), with metadata scoped to that component rather than the
            generator — see the class docstring.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    FILE_EXTENSION = artifacts.SUFFIX
    DEFAULT_PREFIX = "sr-weights"

    def __init__(
        self,
        monitor_metric: str | None = "psnr/val/RGB",
        save_top_k: int = 3,
        dirpath: str | None = None,
        filename_prefix: str | None = None,
        mode: str = "max",
        keep_last: int = 3,
        attribute: str = "model",
        **kwargs: Any,
    ):
        super().__init__(
            monitor_metric=monitor_metric,
            save_top_k=save_top_k,
            dirpath=dirpath,
            filename_prefix=filename_prefix,
            mode=mode,
            keep_last=keep_last,
            **kwargs,
        )
        self.attribute = attribute

    def _describe(self, pl_module: lightning.LightningModule) -> dict[str, Any]:
        """Component-scoped when this callback saves anything but the generator."""
        if self.attribute == "model":
            return build_metadata(_sr(pl_module))
        return build_component_metadata(_sr(pl_module), self.attribute)

    @property
    def state_key(self) -> str:
        """Lightning's callback-state key, widened by ``attribute``.

        ``ModelCheckpoint``'s own key is built from ``monitor``/``mode`` and the
        cadence fields only — not from ``attribute``, ``dirpath`` or
        ``filename``. Lightning refuses two stateful callbacks that share a key
        (``_validate_callbacks_list``), so without this, one generator and one
        discriminator weights callback on the same monitor and cadence — the
        configuration ``attribute`` exists for, and the one the SRGAN template
        ships — cannot be constructed at all.

        Appended to ``super()``'s key rather than rebuilt, so a Lightning
        release that adds or renames a key field carries through unchanged.

        **Resuming a ``.ckpt`` written before this key gained its ``attribute``
        suffix finds no state under the new key**, so monitored bookkeeping
        (``best_model_score``/``best_k_models``) restarts empty for the rest of
        that run. Rolling mode loses nothing: its window was never persisted,
        and :meth:`_RollingSaveMixin.on_train_start` rebuilds it from disk.

        Returns:
            A key unique per ``(monitor, mode, cadence, attribute)``.
        """
        return f"{super().state_key}[attribute={self.attribute}]"

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Validate ``monitor`` against the metrics ``SRLightning`` will log, and ``attribute``.

        The ``monitor`` check is inherited unchanged from
        :meth:`_SRCheckpointBase.setup`; only ``attribute`` is added here.

        ``attribute`` is otherwise only read when a checkpoint is written, so a
        typo — or ``attribute='discriminator'`` on a plain
        :class:`~sisr.training.SRLightning` — would cost a whole checkpoint
        interval before dying with a bare ``AttributeError`` from deep inside the
        save path. Same reasoning as moving the monitor check to startup.

        Args:
            trainer: The active trainer.
            pl_module: The model being trained; must expose ``eval_config`` and
                the component named by ``attribute``.
            stage: Lightning trainer stage. Only ``"fit"`` is checked — see
                ``SRCheckpoint.setup``.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, if ``mode``
                disagrees with that metric's direction (PSNR/SSIM are
                higher-is-better; LPIPS/DISTS are lower-is-better), or if
                ``pl_module`` has no attribute named ``attribute``.
        """
        super().setup(trainer, pl_module, stage)

    def _validate(self, pl_module: lightning.LightningModule) -> None:
        """The base's monitor check, plus that ``attribute`` names something real.

        Runs before the filename is derived, which reads the very component this
        confirms exists -- otherwise a typo surfaces as a bare ``AttributeError``
        from inside the naming code rather than as this message.
        """
        super()._validate(pl_module)
        if not hasattr(pl_module, self.attribute):
            raise MisconfigurationException(
                f"`SRWeightsCheckpoint(attribute={self.attribute!r})` has nothing to save: "
                f"{type(pl_module).__name__} has no attribute {self.attribute!r}. Name a "
                f"component the module actually defines — 'model' (the generator, the "
                f"default) on any SRLightning, 'discriminator' only on an adversarial "
                f"module. Fix the callback's `attribute` in your YAML."
            )

    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Write one component's bare weights + matching provenance metadata.

        Overrides :meth:`ModelCheckpoint._save_checkpoint` — the private hook every
        public save path (``_save_topk_checkpoint``, ``_save_none_monitor_checkpoint``,
        ``_save_last_checkpoint``) funnels through — so ``save_top_k``/filename/removal
        bookkeeping all keep working unmodified; only the on-disk payload changes.
        Mirrors the base's post-save bookkeeping (``_last_global_step_saved``,
        ``_last_checkpoint_saved``, logger notification) so this stays a drop-in
        peer of :class:`SRCheckpoint` to the trainer.

        ``self.attribute`` selects the saved component *and* its metadata
        builder: ``'model'`` uses ``build_metadata`` (generator-scoped),
        anything else ``build_component_metadata`` — so a discriminator's
        artifact never carries metadata for a network it does not contain.

        **Rolling deletion runs last and is called explicitly**, because this
        override never calls ``ModelCheckpoint._save_checkpoint`` (that would
        write the optimizer-bearing payload this class exists to avoid).
        """
        pl_module = trainer.lightning_module
        component = getattr(pl_module, self.attribute)
        monitor_value = float(self.current_score) if self.current_score is not None else None
        if self.attribute == "model":
            meta = build_metadata(
                _sr(pl_module),
                global_step=trainer.global_step,
                batch_step=_logger_step(trainer),
                epoch=trainer.current_epoch,
                monitor=self.monitor,
                monitor_value=monitor_value,
            )
        else:
            meta = build_component_metadata(
                _sr(pl_module),
                self.attribute,
                global_step=trainer.global_step,
                batch_step=_logger_step(trainer),
                epoch=trainer.current_epoch,
                monitor=self.monitor,
                monitor_value=monitor_value,
            )
        # Rank-gated to match what ModelCheckpoint's own save path gets for free:
        # it writes through Strategy.save_checkpoint, which gates on is_global_zero,
        # and this override replaces exactly that method. Without the gate every
        # process writes the same path at once. The bookkeeping below stays
        # unconditional, as it is in the base implementation, and the rolling
        # window's deletes are gated inside Strategy.remove_checkpoint already.
        if trainer.is_global_zero:
            artifacts.save(filepath, component.state_dict(), meta)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

        # Any edit here must preserve this call -- see _RollingSaveMixin's
        # docstring for why a rolling window can't be expressed via save_top_k alone.
        self._enforce_rolling_window(trainer, filepath)
