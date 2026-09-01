import dataclasses
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
    assert cfg.compile_backend is None


def test_sr_eval_config_defaults():
    cfg = SREvalConfig()
    assert cfg.crop_border == 0
    assert cfg.psnr_channels == ["RGB"]
    assert cfg.separate_psnr is False
    # Unlike psnr_channels, the base already defaults to ['RGB', 'Y'] — SSIM
    # has no post-hoc PSNR-style dB correction, so the paper-comparable
    # Y-SSIM ships without waiting for an architecture subclass to add it.
    assert cfg.ssim_channels == ["RGB", "Y"]
    assert cfg.ssim_impl == "wang"


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
        "compile_backend",
        "compile_mode",
    }


def test_sr_eval_config_field_names():
    names = {f.name for f in fields(SREvalConfig)}
    assert names == {
        "crop_border",
        "psnr_channels",
        "separate_psnr",
        "ssim_channels",
        "ssim_impl",
        "perceptual_metrics",
        "lpips_net",
    }


def test_eval_config_rejects_unknown_psnr_channel():
    """Regression: a colorspace outside {RGB, YCbCr} fails fast at
    construction with an actionable message — not an opaque KeyError deep in
    SRLightning."""
    SREvalConfig(psnr_channels=["RGB", "YCbCr"])  # valid — must not raise
    with pytest.raises(ValueError, match="psnr_channels"):
        SREvalConfig(psnr_channels=["RGB", "HSV"])


def test_eval_config_rejects_unknown_psnr_channel_still_raises_alongside_bare_y():
    """Regression: adding bare single-channel entries to the allowlist
    must not widen it into accepting arbitrary strings — 'HSV' stays invalid
    even though 'Y' is now first-class."""
    SREvalConfig(psnr_channels=["RGB", "Y"])  # valid — must not raise
    with pytest.raises(ValueError, match="psnr_channels"):
        SREvalConfig(psnr_channels=["RGB", "HSV"])


def test_eval_config_rejects_unknown_ssim_channel():
    """ssim_channels reuses the same allowlist as psnr_channels, and
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


def test_eval_config_rejects_unknown_ssim_impl():
    """An unknown implementation fails at construction with an actionable
    message, not at the first validation batch hours into a run."""
    SREvalConfig(ssim_impl="daala")  # valid -- must not raise
    with pytest.raises(ValueError, match="ssim_impl"):
        SREvalConfig(ssim_impl="wangg")


# ---------------------------------------------------------------------------
# psnr_keys — the public seam SRLightning and BenchmarkImageLogger depend on
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
    exact ordering. Bare single-channel entries map to () — nothing to
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
    """Regression: a bare 'Y' entry must not decompose into anything
    else — it is already a single channel, unlike 'YCbCr'."""
    cfg = SREvalConfig(psnr_channels=["RGB", "Y"])
    assert cfg.psnr_keys == ["RGB", "Y"]


def test_psnr_keys_is_not_a_dataclass_field():
    """psnr_keys is a derived property, not a stored field — guards against
    it silently becoming a constructor argument / dataclasses.asdict() entry."""
    names = {f.name for f in fields(SREvalConfig)}
    assert "psnr_keys" not in names


# ---------------------------------------------------------------------------
# ssim_keys — mirrors psnr_keys, minus the separate_psnr expansion
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
# perceptual_metrics / lpips_net / perceptual_keys
# ---------------------------------------------------------------------------


def test_perceptual_metrics_default_off():
    """Existing architectures must log exactly the tags they logged before."""
    assert SREvalConfig().perceptual_keys == []


def test_perceptual_metrics_validated_at_construction():
    with pytest.raises(ValueError, match="perceptual_metrics"):
        SREvalConfig(perceptual_metrics=["lpips", "psnr"])


def test_lpips_net_validated_at_construction():
    with pytest.raises(ValueError, match="lpips_net"):
        SREvalConfig(lpips_net="resnet")


def test_perceptual_fields_live_on_the_base_class():
    """A subclass-only *field* breaks --ckpt_path outright, not just silently.

    dataclasses.asdict dumps every field into the checkpoint's hyper_parameters;
    on reload that dict is handed to the *annotation* type, which is base
    SREvalConfig. A field only the subclass declares is then an unexpected
    keyword argument — a hard failure, strictly worse than reverting to a base
    default. ssim_impl is the precedent: field on the base, default on the
    subclass.
    """
    field_names = {f.name for f in dataclasses.fields(SREvalConfig)}
    assert {"perceptual_metrics", "lpips_net"} <= field_names


# ---------------------------------------------------------------------------
# SRTrainingConfig.validate_against — the construction-time seam
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


# --- Field validation: a nonsense value must not reach a published number ---


def test_eval_config_rejects_negative_crop_border():
    """A negative border silently scores UNCROPPED: the slice guard is `n <= 0`,
    so -4 and 0 produce byte-identical numbers and neither errors. `None` is the
    derive-from-scale sentinel, so -1 is exactly what someone reaches for when
    they mean "auto" — it must not be a valid way to ask for that."""
    SREvalConfig(crop_border=0)  # valid -- must not raise
    SREvalConfig(crop_border=4)  # valid -- must not raise
    SREvalConfig(crop_border=None)  # the derive-from-scale sentinel
    with pytest.raises(ValueError, match="crop_border"):
        SREvalConfig(crop_border=-1)
    with pytest.raises(ValueError, match="crop_border"):
        SREvalConfig(crop_border=-4)


def test_training_config_rejects_non_positive_scale():
    """`scale` is unvalidated, and for an architecture with no `scale` hparam
    (SRCNN, deliberately) `validate_against` checks nothing either. It then
    resolves `crop_border`, so a negative scale becomes a negative border and
    the score is silently uncropped."""
    SRTrainingConfig(scale=4)  # valid -- must not raise
    SRTrainingConfig(scale=None)  # unset is legitimate
    for bad in (-3, 0):
        with pytest.raises(ValueError, match="scale"):
            SRTrainingConfig(scale=bad)


def test_training_config_rejects_non_positive_layer_lrs():
    """Length is checked when the optimizer is built; the values never are.
    A negative LR trains every layer in the wrong direction."""
    SRTrainingConfig(layer_lrs=[1e-4, [1e-4, 1e-5]])  # valid -- must not raise
    with pytest.raises(ValueError, match="layer_lrs"):
        SRTrainingConfig(layer_lrs=[-1e-4, -1e-4, -1e-4])
    with pytest.raises(ValueError, match="layer_lrs"):
        SRTrainingConfig(layer_lrs=[1e-4, [1e-4, -1e-5]])
    with pytest.raises(ValueError, match="layer_lrs"):
        SRTrainingConfig(layer_lrs=[0.0])


def test_training_config_rejects_non_positive_init_std():
    """Reaches torch.nn.init.normal_ as-is. init_mean stays unconstrained."""
    SRTrainingConfig(init_std=0.001, init_mean=-1.0)  # valid -- must not raise
    for bad in (-0.001, 0.0):
        with pytest.raises(ValueError, match="init_std"):
            SRTrainingConfig(init_std=bad)


def test_training_config_rejects_unknown_init_strategy():
    """The dispatch is `== "paper"`, so ANY other string means "default".
    'Paper' therefore produces a run that looks configured for the paper's
    initialisation and is not, with nothing saying so. The Literal annotation
    is enforced by jsonargparse at the CLI and by nothing in Python, so a
    directly-constructed config — every test, every script — has no guard."""
    SRTrainingConfig(init_strategy="paper")  # valid -- must not raise
    SRTrainingConfig(init_strategy="default")  # valid -- must not raise
    with pytest.raises(ValueError, match="init_strategy"):
        SRTrainingConfig(init_strategy="Paper")
    with pytest.raises(ValueError, match="init_strategy"):
        SRTrainingConfig(init_strategy="xavier")
