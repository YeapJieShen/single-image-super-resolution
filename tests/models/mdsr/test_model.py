import pytest
import torch

from sisr.models.mdsr import MDSR


def _net(**kw):
    torch.manual_seed(0)
    return MDSR(**{"features": 8, "num_blocks": 2} | kw)


def _n(module):
    return sum(p.numel() for p in module.parameters())


def test_one_model_serves_every_scale_with_the_matching_output_size():
    net = _net().eval()
    x = torch.rand(1, 3, 5, 6)

    for s in (2, 3, 4):
        assert net(x, scale=s).shape == (1, 3, 5 * s, 6 * s)
    assert MDSR.multi_scale is True and MDSR.input_contract == "native_lr"


def test_scale_none_is_the_first_listed_scale():
    net = _net(scale=(3, 2)).eval()
    x = torch.rand(1, 3, 4, 4)

    assert torch.equal(net(x), net(x, scale=3))


def test_an_unconfigured_scale_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        _net(scale=(2, 4))(torch.rand(1, 3, 4, 4), scale=3)


def test_each_scale_runs_only_its_own_preprocessing_and_upsampler():
    """Zeroing scale 3's modules must not move the x2 output; zeroing x2's must."""
    net = _net().eval()
    x = torch.rand(1, 3, 5, 5)
    with torch.no_grad():
        before = net(x, scale=2)
        for p in [*net.pre_process[1].parameters(), *net.up[1].parameters()]:
            p.zero_()
        assert torch.equal(net(x, scale=2), before)
        for p in [*net.pre_process[0].parameters(), *net.up[0].parameters()]:
            p.zero_()
        assert not torch.equal(net(x, scale=2), before)


def test_the_trunk_is_shared_and_the_rest_is_per_scale():
    net = _net()

    assert _n(net) == (
        _n(net.head) + _n(net.blocks) + _n(net.body_conv) + _n(net.tail)
        + _n(net.pre_process) + _n(net.up)
    )  # fmt: skip
    assert len(net.pre_process) == 3 and len(net.up) == 3 and len(net.blocks) == 2
    assert [len(u) for u in net.up] == [1, 1, 2]


def test_preprocessing_blocks_use_5x5_convs_and_the_trunk_3x3():
    net = _net()

    assert all(
        c.kernel_size == (5, 5)
        for c in net.pre_process[0][0].modules()
        if hasattr(c, "kernel_size")
    )
    assert net.blocks[0].conv1.kernel_size == (3, 3)


def test_there_is_no_batch_norm_and_no_saved_mean():
    net = _net()

    assert not [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert not [k for k in net.state_dict() if "mean" in k]


def test_the_mean_shift_matches_edsrs():
    """Zeroed convs: the output is the tail bias over the range plus the mean."""
    net = _net(rgb_mean=(0.5, 0.25, 0.125))
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        net.tail.bias.copy_(torch.tensor([51.0, 25.5, 12.75]))

    out = net(torch.rand(1, 3, 4, 4), scale=2)

    expect = torch.tensor([0.7, 0.35, 0.175]).view(1, 3, 1, 1)
    assert torch.allclose(out, expect.expand_as(out), atol=1e-6)


def test_hparams_hold_the_scale_list_and_the_variant_tag():
    net = _net(scale=(2, 3, 4), features=64, num_blocks=16)

    assert net.hparams["scale"] == [2, 3, 4] and net.variant_tag == "16B64F"
    assert _net(features=64, num_blocks=80).variant_tag == "80B64F"


@pytest.mark.parametrize(
    "kwargs",
    [{"scale": ()}, {"scale": (2, 2)}, {"scale": (2, 5)}, {"num_blocks": 0}, {"rgb_mean": (0.5,)}],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        MDSR(**kwargs)
