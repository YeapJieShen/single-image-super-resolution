"""Direct tests for the sampling discipline itself, not for a dataset that happens to use it.

``test_lightning_module.py`` already covers this path end-to-end, but only
*through* a real SRResNet ``TrainDataset``. Those tests are load-bearing and stay
where they are; what they cannot do is discriminate. ``TrainDataset.__getitem__``
draws its crop with ``random.randint`` today, and the RNG-transparency test is
only meaningful for as long as that stays true — move the crop to a private
``numpy.Generator`` and the assertion still passes while asserting nothing.

Since :mod:`sisr.training.probe` depends on nothing but ``random``, ``pickle``
and duck-typed reads, the guarantees can be stated against datasets built to
exercise them: one that consumes randomness unconditionally, one that refuses to
pickle, one that counts its reads. That is the point of the seam.
"""

import pickle
import random

import pytest
import torch

from sisr.training.probe import probe_pair


class _RandomConsumingDataset:
    """Draws from the global ``random`` sequence on every read, like the real crop does."""

    def __init__(self) -> None:
        self.reads: list[int] = []

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.reads.append(idx)
        random.random()
        random.randint(0, 10_000)
        return torch.zeros(3, 4, 4), torch.zeros(3, 8, 8)


class _UnpicklableDataset(_RandomConsumingDataset):
    """Stands in for a dataset whose ``lmdb.Environment`` is already open."""

    def __reduce__(self):
        raise TypeError("cannot pickle 'Environment' object")


class _DM:
    """Minimal datamodule surface: whichever accessors the test sets."""

    def __init__(self, **datasets) -> None:
        for name, ds in datasets.items():
            setattr(self, name, ds)


def _state_after(fn) -> tuple:
    random.seed(20260819)
    before = random.getstate()
    fn()
    return before, random.getstate()


def test_probe_pair_restores_the_global_random_state():
    """The guarantee, asserted against a dataset that definitely consumes draws.

    Stated directly rather than via SRResNet's crop, so it keeps discriminating
    even if every real dataset stops using the global ``random`` sequence.
    """
    dm = _DM(train_dataset=_RandomConsumingDataset())

    def body():
        with probe_pair(dm) as probe:
            assert probe.sample is not None

    before, after = _state_after(body)
    assert after == before


def test_the_fixture_would_actually_move_the_rng_unguarded():
    """Mutation guard: proves the assertion above can fail.

    Without this, a probe that never touched the dataset at all would satisfy
    the test — the failure mode that makes an RNG assertion look green while
    proving nothing.
    """
    ds = _RandomConsumingDataset()

    def body():
        ds[0]

    before, after = _state_after(body)
    assert after != before


def test_probe_pair_restores_the_rng_even_when_the_body_raises():
    """A failed check must not shift the training sequence either.

    ``setup``'s contract violations raise from inside the ``with`` block, and a
    run that dies at startup is often re-run with the same seed — so the guard
    has to hold on the exception path, not just the happy one.
    """
    dm = _DM(train_dataset=_RandomConsumingDataset())

    def body():
        with pytest.raises(ValueError, match="from the body"):
            with probe_pair(dm) as probe:
                assert probe.sample is not None
                raise ValueError("raised from the body")

    before, after = _state_after(body)
    assert after == before


def test_lazy_train_read_is_also_inside_the_guard():
    """``train_lr()`` reads on demand; that read must be transparent too.

    The laziness is what makes this worth pinning — the read happens after
    ``probe_pair`` has already yielded, which is exactly where a guard written
    as a plain function wrapper would have stopped covering it.
    """
    train, val = _RandomConsumingDataset(), _RandomConsumingDataset()
    dm = _DM(train_dataset=None, val_dataset=val)
    dm.train_dataset = train

    def body():
        with probe_pair(dm) as probe:
            assert probe.train_lr() is not None

    before, after = _state_after(body)
    assert after == before


def test_probe_reads_a_clone_leaving_the_original_untouched():
    """The original dataset must never be the object index 0 was read from.

    This is the picklability guarantee stated positively: the live instance
    handed to a ``DataLoader`` stays exactly as it was, so a later spawned
    worker can still pickle it.
    """
    ds = _RandomConsumingDataset()
    dm = _DM(train_dataset=ds)

    with probe_pair(dm) as probe:
        assert probe.sample is not None

    assert ds.reads == [], "index 0 was read from the live object, not a clone"
    pickle.dumps(ds)  # must still round-trip


def test_probe_falls_back_to_the_live_object_when_it_cannot_be_pickled():
    """An already-poisoned dataset degrades to a direct read rather than raising.

    A ``num_workers=0`` training read opens the environment for real, which is
    harmless on its own but makes a later re-probe's pickle attempt fail. Reading
    live at that point adds no new harm — something else already made it so.
    """
    ds = _UnpicklableDataset()
    dm = _DM(train_dataset=ds)

    with probe_pair(dm) as probe:
        assert probe.sample is not None

    assert ds.reads == [0], "expected the fallback direct read"


def test_train_dataset_index_zero_is_read_at_most_once():
    """When train is both the probe source and the train-shape source, reuse it."""
    ds = _RandomConsumingDataset()
    dm = _DM(train_dataset=ds)

    with probe_pair(dm) as probe:
        assert probe.sample is not None
        first = probe.train_lr()
        second = probe.train_lr()

    assert first is not None and second is not None
    assert first[1] is probe.sample.lr, "train_lr re-sampled instead of reusing the probe sample"
    assert first[1] is second[1], "train_lr is not memoised"


@pytest.mark.parametrize(
    "datasets, expected",
    [
        ({"train_dataset": "T", "val_dataset": "V"}, "train_dataset"),
        ({"val_dataset": "V"}, "val_dataset"),
        ({"test_datasets": {"Set5": "S"}}, "test_datasets"),
    ],
)
def test_dataset_priority_is_train_then_val_then_test(datasets, expected):
    """Whichever loader would actually run first for the live stage."""
    made = {
        k: (
            _RandomConsumingDataset()
            if not isinstance(v, dict)
            else {n: _RandomConsumingDataset() for n in v}
        )
        for k, v in datasets.items()
    }
    with probe_pair(_DM(**made)) as probe:
        assert probe.sample is not None
        assert probe.sample.source == expected


def test_a_datamodule_with_no_probeable_dataset_yields_no_sample():
    """A predict-only run, or a foreign datamodule, is 'nothing to probe' — not an error."""
    with probe_pair(_DM()) as probe:
        assert probe.sample is None
        assert probe.train_lr() is None
