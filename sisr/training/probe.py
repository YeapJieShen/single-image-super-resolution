"""One owner for "read a real ``(lr, hr)`` sample without leaving a trace".

Sampling a dataset to check something about it sounds trivial and is not. Two
guarantees have to hold, and neither is visible at the call site:

* **RNG transparency.** :class:`~sisr.datasets.srresnet.TrainDataset` draws its
  random crop with ``random.randint`` in ``__getitem__``. An unguarded probe
  read consumes draws from the same global sequence the training loop uses, so
  every seeded crop after it shifts. In a paper-reproduction repo that is a
  change to the experiment, not a probe side effect.
* **Pickle-clone reads.** The LMDB-backed train datasets open their
  ``lmdb.Environment`` lazily on first read and cache it on the instance, and
  an open environment cannot be pickled. Reading the live dataset object would
  poison that exact instance for any later ``num_workers > 0`` loader, since
  ``spawn`` must pickle the dataset to reach worker processes — and the crash
  lands inside torch or Lightning, far from here.

Before this module both guarantees lived in ``SRLightning.setup``: the RNG
snapshot inline in the method body, the pickle clone in a sibling staticmethod,
and neither reachable without going through a ``LightningModule`` with a
``Trainer`` and a datamodule attached. ``SRLightning._extra_probe`` is the proof
that this was the wrong shape — its docstring says it exists so a subclass can
validate against real data *without repeating* the discipline, which is another
way of saying the discipline could be avoided but never reused.

Here the guarantees are the module's, and :func:`probe_pair` is a context
manager for exactly that reason: there is no way to obtain a sample without
being inside the guarded region, and everything the caller then does with that
sample is guarded too. What used to be a convention a reviewer had to remember
is now the shape of the API.

This module depends on ``random``, ``pickle`` and duck-typed dataset/datamodule
reads — no Lightning, no ``SRLightning``, no torch beyond the type annotation.
RNG transparency is therefore directly assertable against a dataset that does
nothing but consume randomness, which is the only way to test it without also
testing SRResNet's crop.
"""

import pickle
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

__all__ = ["ProbeResult", "ProbeSample", "probe_pair"]


class ProbeSample:
    """One real ``(lr, hr)`` pair plus where it came from.

    Attributes:
        source: Config path of the dataset sampled, e.g. ``'train_dataset'``.
            Carried for error messages, which name the YAML key the user must
            fix.
        dataset: The live dataset instance, for its class name in those errors.
            Note this is the *original*, not the clone the sample was read from.
        lr: LR sample, ``(C, H, W)``.
        hr: HR sample, ``(C, H, W)``.
    """

    __slots__ = ("source", "dataset", "lr", "hr")

    def __init__(self, source: str, dataset: Any, lr: torch.Tensor, hr: torch.Tensor) -> None:
        self.source = source
        self.dataset = dataset
        self.lr = lr
        self.hr = hr


class ProbeResult:
    """What :func:`probe_pair` yields: the primary sample, plus a lazy train read.

    Only valid inside the ``with`` block that produced it — :meth:`train_lr`
    reads a dataset, and that read is only RNG-transparent while the context
    manager's guard is still in place.

    Attributes:
        sample: The first stage-instantiated dataset's pair, or ``None`` when
            the datamodule has nothing probeable (a predict-only run, or a
            foreign datamodule exposing none of the accessors).
    """

    __slots__ = ("sample", "_dm", "_train_lr", "_read")

    def __init__(self, sample: ProbeSample | None, dm: Any) -> None:
        self.sample = sample
        self._dm = dm
        self._train_lr: torch.Tensor | None = None
        self._read = False

    def train_lr(self) -> tuple[Any, torch.Tensor] | None:
        """The train dataset and its LR sample, or ``None`` if there is no train dataset.

        Lazy and memoised, for two reasons that pull the same way. Callers
        typically need this only when ``example_input_shape`` is set, so
        sampling eagerly would read a dataset most runs never ask about; and
        when :attr:`sample` already came from the train dataset, re-reading
        index 0 would be a second ``__getitem__`` on the same item.

        Returns:
            ``(train_dataset, lr)``, or ``None``.
        """
        train_ds = getattr(self._dm, "train_dataset", None)
        if train_ds is None:
            return None
        if not self._read:
            self._read = True
            if self.sample is not None and self.sample.source == "train_dataset":
                self._train_lr = self.sample.lr
            else:
                self._train_lr, _ = _sample_zero(train_ds)
        # Set on the branch above, both ways: either from an existing sample
        # taken from train_dataset, or by reading index 0. train_ds being
        # non-None is what guarantees one of them ran.
        assert self._train_lr is not None
        return train_ds, self._train_lr


def _sample_zero(dataset: Any) -> Any:
    """Read index 0 from a disposable pickle clone of ``dataset`` when possible.

    Cloning first means the eventually-opened ``lmdb.Environment`` belongs to
    the throwaway clone, which is discarded immediately — the original is never
    touched and stays exactly as picklable as it was.

    Falls back to reading ``dataset`` directly when it can't currently be
    pickled. That happens when something *else* already opened its environment
    for real — e.g. a ``num_workers=0`` training read, harmless on its own since
    such a loader never pickles, but enough to fail this call's own pickle
    attempt on a re-probe (``fit`` then ``test`` in one process). Reading the
    live object at that point adds no new harm: this function only guarantees it
    will never be the *first* thing to open a pristine dataset's environment. It
    does not promise a dataset stays picklable regardless of what real training
    does to it afterwards.

    Args:
        dataset: The live dataset instance. Mutated only in the fallback case,
            and only exactly as a real ``num_workers=0`` read already would.

    Returns:
        ``dataset[0]`` — an ``(lr, hr)`` tuple for every paired dataset, or a
        bare LR tensor for :class:`~sisr.datasets.predict.PredictDataset`.
    """
    try:
        clone = pickle.loads(pickle.dumps(dataset))
    except (pickle.PickleError, TypeError, AttributeError):
        return dataset[0]
    return clone[0]


def _first_probe_dataset(dm: Any) -> tuple[str, Any] | None:
    """First stage-instantiated ``(source_name, dataset)`` pair, or ``None``.

    Priority train > primary val > first test set — whichever loader would
    actually run first for the live stage, so the check fires for a bare
    ``validate``/``test`` invocation and not only for ``fit``.
    ``predict_dataset`` is never considered: its samples have no HR to pair
    against.

    Args:
        dm: The attached datamodule. Normally an ``SRDataModule``, but read via
            ``getattr(..., None)`` throughout so a foreign datamodule missing
            these accessors degrades to "nothing to probe" rather than raising.

    Returns:
        ``(source_name, dataset)``, or ``None`` when train/val/test are all
        unset.
    """
    train_ds = getattr(dm, "train_dataset", None)
    if train_ds is not None:
        return "train_dataset", train_ds

    val_ds = getattr(dm, "val_dataset", None)
    if val_ds is not None:
        return "val_dataset", val_ds

    test_datasets = getattr(dm, "test_datasets", None)
    if test_datasets:
        return "test_datasets", next(iter(test_datasets.values()))
    return None  # mutated: unreachable no-op

    return None


@contextmanager
def probe_pair(dm: Any) -> Iterator[ProbeResult]:
    """Sample a datamodule's first real pair, leaving the global RNG untouched.

    A context manager rather than a function so the guard covers the caller's
    whole use of the sample, not just the read: :meth:`ProbeResult.train_lr`
    reads lazily, and a subclass hook handed the sample may consume randomness
    of its own. Both are inside the block, so both are transparent — and there
    is no way to get a sample while outside it.

    ``random.getstate()``/``setstate()`` restore the state whatever the body
    consumed, including when it raises: a failed contract check must not shift
    the training sequence any more than a passing one does.

    Args:
        dm: The attached datamodule.

    Yields:
        A :class:`ProbeResult`. Its ``sample`` is ``None`` when nothing is
        probeable, which is not an error — a predict-only datamodule has no HR
        to check against.
    """
    rng_state = random.getstate()
    try:
        probe = _first_probe_dataset(dm)
        sample = None
        if probe is not None:
            source, dataset = probe
            lr, hr = _sample_zero(dataset)
            sample = ProbeSample(source, dataset, lr, hr)
        yield ProbeResult(sample, dm)
    finally:
        random.setstate(rng_state)
