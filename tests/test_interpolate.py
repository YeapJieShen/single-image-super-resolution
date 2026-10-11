import pytest
import torch

from sisr import artifacts
from sisr.interpolate import interpolate_weights


def _meta(**over):
    meta = {
        "kind": "sr_model",
        "model": {"class_path": "sisr.models.esrgan.RRDBNet", "init_args": {"scale": 4}},
        "processor": {"class_path": "sisr.processors.RGBProcessor"},
        "io": {"scale": 4, "output_range": [0.0, 1.0]},
    }
    for dotted, value in over.items():
        section, key = dotted.split(".")
        meta[section] = {**meta[section], key: value}
    return meta


def _write(path, w, b, **over):
    artifacts.save(path, {"w": w, "b": b}, _meta(**over))
    return path


@pytest.fixture
def ab(tmp_path):
    torch.manual_seed(0)
    a = _write(tmp_path / "a.safetensors", torch.rand(3, 3), torch.rand(3))
    b = _write(tmp_path / "b.safetensors", torch.rand(3, 3), torch.rand(3))
    return a, b, tmp_path / "out.safetensors"


def test_alpha_zero_is_a_and_alpha_one_is_b(ab):
    a, b, out = ab
    ta, tb = artifacts.load(a)[0], artifacts.load(b)[0]

    interpolate_weights(a, b, 0.0, out)
    assert all(torch.equal(artifacts.load(out)[0][k], ta[k]) for k in ta)
    interpolate_weights(a, b, 1.0, out)
    assert all(torch.equal(artifacts.load(out)[0][k], tb[k]) for k in tb)


def test_the_midpoint_is_the_mean_and_alpha_weights_b(ab):
    a, b, out = ab
    ta, tb = artifacts.load(a)[0], artifacts.load(b)[0]

    interpolate_weights(a, b, 0.25, out)

    got = artifacts.load(out)[0]
    for k in ta:
        torch.testing.assert_close(got[k], 0.75 * ta[k] + 0.25 * tb[k])


def test_the_output_carries_a_header_plus_the_interpolation_record(ab):
    a, b, out = ab

    interpolate_weights(a, b, 0.8, out)

    meta = artifacts.load(out)[1]
    assert meta["model"] == _meta()["model"] and meta["io"] == _meta()["io"]
    assert meta["interpolation"] == {"a": "a.safetensors", "b": "b.safetensors", "alpha": 0.8}


def test_dtype_is_preserved(tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    artifacts.save(a, {"w": torch.ones(2, dtype=torch.float16)}, _meta())
    artifacts.save(b, {"w": torch.zeros(2, dtype=torch.float16)}, _meta())

    interpolate_weights(a, b, 0.5, tmp_path / "o.safetensors")

    out = artifacts.load(tmp_path / "o.safetensors")[0]["w"]
    assert out.dtype == torch.float16 and torch.equal(
        out, torch.full((2,), 0.5, dtype=torch.float16)
    )


@pytest.mark.parametrize(
    "over, key",
    [
        ({"model.class_path": "sisr.models.srresnet.SRResNet"}, "model.class_path"),
        ({"model.init_args": {"scale": 2}}, "model.init_args"),
        ({"io.scale": 2}, "io.scale"),
        ({"processor.class_path": "sisr.processors.YChannelProcessor"}, "processor.class_path"),
        ({"io.output_range": [-1.0, 1.0]}, "io.output_range"),
    ],
)
def test_networks_that_differ_are_refused_naming_the_key(tmp_path, over, key):
    a = _write(tmp_path / "a.safetensors", torch.rand(2), torch.rand(2))
    b = _write(tmp_path / "b.safetensors", torch.rand(2), torch.rand(2), **over)

    with pytest.raises(ValueError, match=key):
        interpolate_weights(a, b, 0.5, tmp_path / "o.safetensors")
    assert not (tmp_path / "o.safetensors").exists()


def test_mismatched_tensors_are_refused(tmp_path):
    a = _write(tmp_path / "a.safetensors", torch.rand(2), torch.rand(2))
    artifacts.save(tmp_path / "b.safetensors", {"w": torch.rand(3), "b": torch.rand(2)}, _meta())

    with pytest.raises(ValueError, match="w:"):
        interpolate_weights(a, tmp_path / "b.safetensors", 0.5, tmp_path / "o.safetensors")


def test_a_component_artifact_is_refused(tmp_path):
    a = _write(tmp_path / "a.safetensors", torch.rand(2), torch.rand(2))
    artifacts.save(
        tmp_path / "d.safetensors",
        {"w": torch.rand(2), "b": torch.rand(2)},
        {"kind": "component", "component": {"name": "discriminator"}},
    )

    with pytest.raises(ValueError, match="component"):
        interpolate_weights(a, tmp_path / "d.safetensors", 0.5, tmp_path / "o.safetensors")
