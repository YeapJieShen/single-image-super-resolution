import functools

import pytest
import torch

from sisr.losses import SRLoss, TotalVariationLoss, VGG19FeatureLoss, WeightedSumLoss
from sisr.models.srcnn import SRCNN
from sisr.processors import RGBProcessor, RGBSignedOutputProcessor, SRProcessor
from sisr.training import SREvalConfig, SRLightning


class _BindSpy(SRLoss):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.bound: list[SRProcessor] = []

    def bind(self, processor: SRProcessor) -> None:
        self.bound.append(processor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.value, requires_grad=True)


def test_weighted_sum_matches_the_hand_computed_total():
    loss = WeightedSumLoss(
        terms={"a": _BindSpy(2.0), "b": _BindSpy(5.0)}, weights={"a": 1.0, "b": 0.1}
    )

    got = loss(torch.zeros(1), torch.zeros(1))

    assert got.item() == pytest.approx(2.0 * 1.0 + 5.0 * 0.1)


def test_a_missing_weight_defaults_to_one():
    loss = WeightedSumLoss(terms={"a": _BindSpy(2.0), "b": _BindSpy(5.0)}, weights={"b": 0.1})

    assert loss(torch.zeros(1), torch.zeros(1)).item() == pytest.approx(2.5)


def test_an_unknown_weight_key_raises_rather_than_being_ignored():
    """A typo in a weight name would otherwise silently train at weight 1.0."""
    with pytest.raises(ValueError, match="vgg2"):
        WeightedSumLoss(terms={"vgg22": _BindSpy(1.0)}, weights={"vgg2": 0.5})


def test_empty_terms_raises():
    with pytest.raises(ValueError, match="at least one"):
        WeightedSumLoss(terms={})


@pytest.mark.parametrize("name", ["a/b", "a.b"])
def test_a_term_name_that_would_break_a_metric_tag_raises(name: str):
    """Names become TensorBoard tag segments under loss/train/, so a '/' would
    silently invent a nesting level. '.' is rejected by ModuleDict anyway."""
    with pytest.raises(ValueError, match="term name"):
        WeightedSumLoss(terms={name: _BindSpy(1.0)})


def test_bind_reaches_nested_srloss_terms_and_skips_plain_modules():
    spy = _BindSpy(1.0)
    loss = WeightedSumLoss(terms={"spy": spy, "plain": torch.nn.MSELoss()})
    processor = RGBSignedOutputProcessor()

    loss.bind(processor)

    assert spy.bound == [processor]


def test_nested_vgg_loss_actually_receives_the_bound_range():
    """Forwarding bind must take EFFECT, not merely be called. An unbound
    VGGFeatureLoss silently assumes [0, 1], so a composite that failed to
    forward would train on mis-normalised features with no error at all."""
    with pytest.warns(UserWarning, match="randomly initialised"):
        nested = VGG19FeatureLoss(layer="vgg22", weights=None)
    with pytest.warns(UserWarning, match="randomly initialised"):
        direct = VGG19FeatureLoss(layer="vgg22", weights=None)
    direct._vgg.load_state_dict(nested._vgg.state_dict())

    composite = WeightedSumLoss(terms={"vgg22": nested})
    composite.bind(RGBSignedOutputProcessor())
    direct.bind(RGBSignedOutputProcessor())
    pred = torch.rand(1, 3, 32, 32) * 2 - 1
    target = torch.rand(1, 3, 32, 32) * 2 - 1

    assert composite(pred, target).item() == pytest.approx(direct(pred, target).item(), rel=1e-6)


def test_last_terms_holds_detached_weighted_contributions_that_sum_to_the_total():
    """Weighted, not raw: the point of the breakdown is 'which term dominates',
    and summing to the total is the property that answers it."""
    loss = WeightedSumLoss(
        terms={"a": _BindSpy(2.0), "b": _BindSpy(5.0)}, weights={"a": 1.0, "b": 0.1}
    )

    total = loss(torch.zeros(1), torch.zeros(1))

    assert set(loss.last_terms) == {"a", "b"}
    assert loss.last_terms["b"].item() == pytest.approx(0.5)
    assert sum(v.item() for v in loss.last_terms.values()) == pytest.approx(total.item())
    assert not any(v.requires_grad for v in loss.last_terms.values())


def test_last_terms_are_stable_buffers_written_in_place():
    """A CUDA-graph replay re-runs only recorded kernels, never the Python line
    that creates a tensor — so the entry a replay updates is the one present at
    capture. Rebinding it on an eager forward (mid-training validation, or an
    epoch's partial last batch) would strand every per-term tag on that eager
    value for the rest of the run, silently, while loss/train kept moving."""
    loss = WeightedSumLoss(terms={"a": _BindSpy(2.0)}, weights={"a": 1.0})

    loss(torch.zeros(1), torch.zeros(1))
    first = loss.last_terms["a"]
    loss.terms["a"].value = 7.0
    loss(torch.zeros(1), torch.zeros(1))

    assert loss.last_terms["a"] is first, "rebound instead of writing in place"
    assert first.item() == pytest.approx(7.0), "buffer was not updated"


def test_describe_renders_the_recipe():
    loss = WeightedSumLoss(
        terms={"vgg22": _BindSpy(1.0), "tv": TotalVariationLoss()},
        weights={"vgg22": 1.0, "tv": 2e-8},
    )

    assert loss.describe() == "1*vgg22 + 2e-08*tv"


def test_the_faithful_srresnet_vgg22_recipe_binds_and_backpropagates():
    """Ledig et al. §3.4: VGG22 content loss plus total variation at 2e-8, and
    no MSE term. This is the recipe the composite exists for."""
    with pytest.warns(UserWarning, match="randomly initialised"):
        vgg = VGG19FeatureLoss(layer="vgg22", weights=None)
    criterion = WeightedSumLoss(
        terms={"vgg22": vgg, "tv": TotalVariationLoss()},
        weights={"vgg22": 1.0, "tv": 2.0e-8},
    )
    lit = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBSignedOutputProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        criterion=criterion,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    loss, *_ = lit._step((torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)))
    loss.backward()

    assert lit.criterion_description == "1*vgg22 + 2e-08*tv"
    assert any(p.grad is not None for p in lit.model.parameters())


def test_composite_criterion_survives_a_module_state_dict_roundtrip():
    """A nested VGG must not reintroduce itself into the checkpoint."""
    with pytest.warns(UserWarning):
        vgg = VGG19FeatureLoss(layer="vgg22", weights=None)
    lit = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        criterion=WeightedSumLoss(terms={"vgg22": vgg, "tv": TotalVariationLoss()}),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    assert all(k.startswith("model.") for k in lit.state_dict()), list(lit.state_dict())
