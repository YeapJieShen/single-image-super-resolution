"""The distributable artifact's own contract: header encoding, refusal, drift."""

import json

import pytest
import safetensors.torch
import torch

from sisr import artifacts

_META = {
    "format": "sisr-meta-v2",
    "kind": "sr_model",
    "created": "2026-08-28T00:00:00+00:00",
    "versions": {"sisr": "0.1.0", "torch": "2.13.0", "lightning": "2.6.5"},
    "io": {"scale": 4, "output_range": [-1.0, 1.0], "output_colorspace": "RGB"},
    "processor": {"class_path": "sisr.processors.rgb.RGBSignedOutputProcessor"},
}


def test_header_is_one_entry_per_field_not_one_blob():
    """The whole reason for this encoding: anything that can open the file gets a
    readable table, not a single opaque string it has to know to parse."""
    header = artifacts.encode_metadata(_META)

    assert set(header) == set(_META)
    # A string field stays a string, so it is legible without decoding.
    assert header["format"] == "sisr-meta-v2"
    assert json.loads(header["io"])["scale"] == 4


def test_metadata_round_trips_through_the_flat_header():
    assert artifacts.decode_metadata(artifacts.encode_metadata(_META)) == _META


def test_save_and_load_round_trip_tensors_and_provenance(tmp_path):
    tensors = {"a": torch.randn(2, 3), "b": torch.zeros(4)}
    path = tmp_path / f"m{artifacts.SUFFIX}"

    artifacts.save(path, tensors, _META)
    loaded, meta = artifacts.load(path)

    assert meta == _META
    assert torch.equal(loaded["a"], tensors["a"])
    assert torch.equal(loaded["b"], tensors["b"])


def test_save_accepts_a_non_contiguous_tensor(tmp_path):
    """A state_dict may hand back views, which safetensors refuses outright."""
    view = torch.randn(4, 4).t()
    assert not view.is_contiguous()
    path = tmp_path / f"m{artifacts.SUFFIX}"

    artifacts.save(path, {"w": view}, _META)

    assert torch.equal(artifacts.load(path)[0]["w"], view)


def test_load_refuses_a_file_carrying_no_provenance(tmp_path):
    """Weights whose provenance is unknown are exactly what every downstream
    check exists to prevent, so an unlabelled file is refused, not guessed at."""
    path = tmp_path / f"foreign{artifacts.SUFFIX}"
    safetensors.torch.save_file({"w": torch.zeros(2)}, str(path))

    with pytest.raises(ValueError, match="no sisr provenance header"):
        artifacts.load(path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("io", "output_range", [0.0, 1.0]), ("processor", "class_path", "sisr.processors.rgb.RGB")],
)
def test_require_compatible_refuses_a_meaning_changing_mismatch(section, key, value):
    """Both defaults are fields where being wrong yields a plausible image and no
    error -- apply the wrong output range and every pixel is silently misscaled."""
    found = {**_META, section: {**_META[section], key: value}}

    with pytest.raises(ValueError, match=f"{section}.{key}"):
        artifacts.require_compatible(found, _META)


def test_require_compatible_accepts_a_match():
    artifacts.require_compatible(dict(_META), _META)


def test_require_compatible_warns_about_version_drift_rather_than_refusing():
    """Refusing here would expire every artifact on the next dependency bump,
    and a library version cannot change what the tensors mean."""
    found = {**_META, "versions": {**_META["versions"], "torch": "2.9.0"}}

    with pytest.warns(UserWarning, match="different library versions"):
        artifacts.require_compatible(found, _META)


def test_require_compatible_takes_a_stricter_field_list_when_the_caller_has_more_to_lose():
    """Initialising a generator from the wrong weights silently mistrains, so that
    caller refuses on more than the default set."""
    found = {**_META, "io": {**_META["io"], "scale": 2}}

    artifacts.require_compatible(found, _META)  # scale is not a default field

    with pytest.raises(ValueError, match="io.scale"):
        artifacts.require_compatible(found, _META, fields=(("io", "scale"),))


def test_stem_names_a_model_by_what_distinguishes_it():
    """`ls` should answer "which model is this" without opening anything."""
    meta = {
        "kind": "sr_model",
        "model": {"class_path": "sisr.models.srresnet.model.SRResNet", "variant": "16B64F"},
        "io": {"scale": 4, "output_colorspace": "RGB"},
    }
    assert artifacts.stem(meta) == "SRResNet_x4_RGB_16B64F"


def test_stem_drops_the_parts_that_do_not_describe_a_critic():
    """Scale and colourspace describe a generator's output; a discriminator has
    neither, so naming it with them would assert something untrue."""
    meta = {
        "kind": "component",
        "component": {
            "class_path": "sisr.models.srgan.discriminator.SRDiscriminator",
            "variant": "96",
        },
    }
    assert artifacts.stem(meta) == "SRDiscriminator_96"


def test_stem_omits_an_absent_part_rather_than_rendering_it():
    """A missing variant must not produce a literal 'None' in a filename."""
    meta = {
        "kind": "sr_model",
        "model": {"class_path": "a.b.Thing", "variant": None},
        "io": {"scale": 2, "output_colorspace": "Y"},
    }
    assert artifacts.stem(meta) == "Thing_x2_Y"
