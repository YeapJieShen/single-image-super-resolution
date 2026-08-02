from dataclasses import fields

import pytest

from sisr.models.srcnn import SRCNN
from sisr.models.srresnet import SRResNet
from sisr.processors import RGBProcessor
from sisr.training import SREvalConfig, SRTrainingConfig


def test_sr_training_config_defaults():
    cfg = SRTrainingConfig()
    assert cfg.layer_lrs is None
    assert cfg.example_input_shape is None
    assert cfg.init_strategy == "default"
    assert cfg.init_mean == 0.0
    assert cfg.init_std == 0.01
    assert cfg.scale is None


def test_sr_eval_config_defaults():
    cfg = SREvalConfig()
    assert cfg.crop_border == 0
    assert cfg.psnr_channels == ["RGB"]
    assert cfg.separate_psnr is False
    # Unlike psnr_channels, the base already defaults to ['RGB', 'Y'] — SSIM
    # has no post-hoc PSNR-style dB correction, so the paper-comparable
    # Y-SSIM ships without waiting for an architecture subclass to add it.
    assert cfg.ssim_channels == ["RGB", "Y"]


def test_sr_eval_config_psnr_channels_isolated_per_instance():
    """`field(default_factory=...)` guards against shared mutable defaults."""
    a = SREvalConfig()
    b = SREvalConfig()
    a.psnr_channels.append("YCbCr")
    assert b.psnr_channels == ["RGB"], "default_factory must produce a fresh list per instance"


def test_sr_eval_config_ssim_channels_isolated_per_instance():
    a = SREvalConfig()
    b = SREvalConfig()
    a.ssim_channels.append("YCbCr")
    assert b.ssim_channels == ["RGB", "Y"], "default_factory must produce a fresh list per instance"


def test_sr_training_config_field_names():
    """Reduced surface check — guards against accidental field renames."""
    names = {f.name for f in fields(SRTrainingConfig)}
    assert names == {
        "layer_lrs",
        "example_input_shape",
        "init_strategy",
        "init_mean",
        "init_std",
        "scale",
    }


def test_sr_eval_config_field_names():
    names = {f.name for f in fields(SREvalConfig)}
    assert names == {"crop_border", "psnr_channels", "separate_psnr", "ssim_channels"}


def test_eval_config_rejects_unknown_psnr_channel():
    """Regression (P2.3): a colorspace outside {RGB, YCbCr} fails fast at
    construction with an actionable message — not an opaque KeyError deep in
    SRLightning."""
    SREvalConfig(psnr_channels=["RGB", "YCbCr"])  # valid — must not raise
    with pytest.raises(ValueError, match="psnr_channels"):
        SREvalConfig(psnr_channels=["RGB", "HSV"])


def test_eval_config_rejects_unknown_psnr_channel_still_raises_alongside_bare_y():
    """Regression (P2.9): adding bare single-channel entries to the allowlist
    must not widen it into accepting arbitrary strings — 'HSV' stays invalid
    even though 'Y' is now first-class."""
    SREvalConfig(psnr_channels=["RGB", "Y"])  # valid — must not raise
    with pytest.raises(ValueError, match="psnr_channels"):
        SREvalConfig(psnr_channels=["RGB", "HSV"])


def test_eval_config_rejects_unknown_ssim_channel():
    """(P3.8): ssim_channels reuses the same allowlist as psnr_channels, and
    the error names the offending field so it's actionable."""
    SREvalConfig(ssim_channels=["RGB", "Y", "YCbCr"])  # valid — must not raise
    with pytest.raises(ValueError, match="ssim_channels"):
        SREvalConfig(ssim_channels=["RGB", "HSV"])


def test_eval_config_rejects_unknown_channel_in_either_field_independently():
    """psnr_channels and ssim_channels are validated independently — an
    invalid entry in one must not be masked by the other being valid."""
    with pytest.raises(ValueError, match="psnr_channels"):
        SREvalConfig(psnr_channels=["HSV"], ssim_channels=["RGB"])
    with pytest.raises(ValueError, match="ssim_channels"):
        SREvalConfig(psnr_channels=["RGB"], ssim_channels=["HSV"])


# ---------------------------------------------------------------------------
# psnr_keys (INIT.16) — the public seam B1/C1 depend on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "psnr_channels,separate_psnr",
    [
        (["RGB"], False),
        (["RGB"], True),
        (["YCbCr"], False),
        (["YCbCr"], True),
        (["RGB", "YCbCr"], False),
        (["RGB", "YCbCr"], True),
        (["Y"], False),
        (["Y"], True),
        (["RGB", "Y"], False),
        (["RGB", "Y"], True),
    ],
)
def test_psnr_keys_matches_legacy_loop_derivation(psnr_channels, separate_psnr):
    """psnr_keys must reproduce exactly the loop SRLightning.__init__ used to
    inline (self._psnr_keys) before this property existed. Downstream PRs
    (benchmark PSNR-key selection, TensorBoard tag rename) depend on this
    exact ordering. Bare single-channel entries (P2.9) map to () — nothing to
    expand — so they always contribute exactly one key."""
    channel_names = {
        "RGB": ("R", "G", "B"),
        "YCbCr": ("Y", "Cb", "Cr"),
        "R": (),
        "G": (),
        "B": (),
        "Y": (),
        "Cb": (),
        "Cr": (),
    }
    legacy_keys: list[str] = []
    for cs in psnr_channels:
        if separate_psnr:
            legacy_keys.extend(channel_names[cs])
        legacy_keys.append(cs)

    cfg = SREvalConfig(psnr_channels=psnr_channels, separate_psnr=separate_psnr)
    assert cfg.psnr_keys == legacy_keys


def test_psnr_keys_bare_y_entry_yields_exactly_one_key():
    """Regression (P2.9): a bare 'Y' entry must not decompose into anything
    else — it is already a single channel, unlike 'YCbCr'."""
    cfg = SREvalConfig(psnr_channels=["RGB", "Y"])
    assert cfg.psnr_keys == ["RGB", "Y"]


def test_psnr_keys_is_not_a_dataclass_field():
    """psnr_keys is a derived property, not a stored field — guards against
    it silently becoming a constructor argument / dataclasses.asdict() entry."""
    names = {f.name for f in fields(SREvalConfig)}
    assert "psnr_keys" not in names


# ---------------------------------------------------------------------------
# ssim_keys (P3.8) — mirrors psnr_keys, minus the separate_psnr expansion
# ---------------------------------------------------------------------------


def test_ssim_keys_default_matches_ssim_channels_default():
    cfg = SREvalConfig()
    assert cfg.ssim_keys == ["RGB", "Y"]


@pytest.mark.parametrize(
    "ssim_channels",
    [["RGB"], ["Y"], ["RGB", "Y"], ["YCbCr"], ["RGB", "YCbCr", "Cb"]],
)
def test_ssim_keys_equals_ssim_channels_verbatim(ssim_channels):
    """Unlike psnr_keys, ssim_keys never expands a colorspace into
    sub-channels — there is no separate_ssim flag — so it always equals
    ssim_channels itself, in order."""
    cfg = SREvalConfig(ssim_channels=ssim_channels)
    assert cfg.ssim_keys == ssim_channels


def test_ssim_keys_is_not_a_dataclass_field():
    names = {f.name for f in fields(SREvalConfig)}
    assert "ssim_keys" not in names


# ---------------------------------------------------------------------------
# SRTrainingConfig.validate_against (INIT.16) — the construction-time seam
# ---------------------------------------------------------------------------


def test_validate_against_noop_when_example_input_shape_unset():
    """example_input_shape is optional; validate_against must not probe the
    model (or raise) when it's left at the default None."""
    cfg = SRTrainingConfig()
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    cfg.validate_against(model, RGBProcessor())  # must not raise


def test_validate_against_rejects_channel_mismatch():
    cfg = SRTrainingConfig(example_input_shape=(1, 33, 33))
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(ValueError, match="example_input_shape"):
        cfg.validate_against(model, RGBProcessor())  # RGBProcessor.model_channels == 3


def test_validate_against_accepts_matching_shape_and_runs_forward_probe():
    cfg = SRTrainingConfig(example_input_shape=(3, 33, 33))
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    cfg.validate_against(model, RGBProcessor())  # must not raise


def test_validate_against_forward_probe_catches_architecture_mismatch_not_just_channels():
    """The forward probe validates the real nn.Module, so it also catches
    defects a bare channel-count check would miss — here, a 'valid'-padding
    9x9 kernel run over a 5x5 spatial input (channels match; PyTorch itself
    rejects the spatial size)."""
    cfg = SRTrainingConfig(example_input_shape=(3, 5, 5))
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(RuntimeError):
        cfg.validate_against(model, RGBProcessor())


# ---------------------------------------------------------------------------
# SRTrainingConfig.scale / validate_against's scale correlation check
# ---------------------------------------------------------------------------


def test_validate_against_noop_when_scale_unset_on_config():
    """scale=None (the default) skips the check even when the model declares one."""
    cfg = SRTrainingConfig()
    model = SRResNet(scale=4, num_residual_blocks=1)
    cfg.validate_against(model, RGBProcessor())  # must not raise


def test_validate_against_noop_when_model_has_no_scale_hparam():
    """SRCNN's hparams carry no 'scale' key; a configured scale goes unchecked
    rather than raising on a key that was never meant to correlate."""
    cfg = SRTrainingConfig(scale=4)
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    cfg.validate_against(model, RGBProcessor())  # must not raise


def test_validate_against_accepts_matching_scale():
    cfg = SRTrainingConfig(scale=4)
    model = SRResNet(scale=4, num_residual_blocks=1)
    cfg.validate_against(model, RGBProcessor())  # must not raise


def test_validate_against_rejects_scale_mismatch():
    cfg = SRTrainingConfig(scale=2)
    model = SRResNet(scale=4, num_residual_blocks=1)
    with pytest.raises(ValueError, match="scale"):
        cfg.validate_against(model, RGBProcessor())
