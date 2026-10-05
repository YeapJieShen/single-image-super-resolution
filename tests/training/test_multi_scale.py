"""Multi-scale training: config, dispatch to a ``multi_scale`` model, per-scale evaluation."""

import functools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sisr import artifacts
from sisr.metrics.scoring import SRScorer, metric_tag
from sisr.models.base import SRModel
from sisr.models.srgan import SRDiscriminator
from sisr.processors import RGBProcessor, RGBSignedOutputProcessor
from sisr.training import SRDataModule, SREvalConfig, SRLightning, SRTrainingConfig
from sisr.training.callbacks import BenchmarkImageLogger
from sisr.training.config import AdversarialTrainingConfig
from sisr.training.gan_module import SRGANLightning
from sisr.training.metadata import build_metadata


class _MultiScaleNet(SRModel):
    """Nearest-neighbour upsampler that serves several scales and records what it was handed."""

    input_contract = "native_lr"
    multi_scale = True

    def __init__(self, scales=(2, 3, 4)):
        super().__init__()
        self._hparams = {"scale": list(scales)}
        self.w = torch.nn.Parameter(torch.ones(1))
        self.seen: list = []

    @property
    def variant_tag(self) -> str:
        return "ms"

    def forward(self, x, scale=None):
        self.seen.append(scale)
        s = scale if scale is not None else self._hparams["scale"][0]
        return torch.nn.functional.interpolate(x, scale_factor=s, mode="nearest") * self.w


class _PlainNet(SRModel):
    """A single-scale model whose ``forward`` would raise on any extra argument."""

    input_contract = "native_lr"

    def __init__(self, scale=2):
        super().__init__()
        self._hparams = {"scale": scale}
        self.s = scale
        self.w = torch.nn.Parameter(torch.ones(1))

    @property
    def variant_tag(self) -> str:
        return "plain"

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.s, mode="nearest") * self.w


def _lit(model=None, scale=(2, 3, 4), eval_config=None, **kw) -> SRLightning:
    return SRLightning(
        model=model or _MultiScaleNet(),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=list(scale) if scale is not None else None),
        eval_config=eval_config or SREvalConfig(crop_border=None, ssim_channels=[]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-3),
        **kw,
    )


def _pair(scale: int, hw: int = 16):
    lr = torch.rand(1, 3, hw // scale, hw // scale) * 0.4 + 0.2
    return lr, torch.nn.functional.interpolate(lr, scale_factor=scale, mode="nearest")


# --- config ---------------------------------------------------------------------------


def test_a_one_element_scale_list_is_the_int():
    cfg = SRTrainingConfig(scale=[4])

    assert cfg.scale == 4
    assert SRTrainingConfig(scale=[2, 4]).scale == [2, 4]


def test_validate_against_compares_scales_as_sets():
    lit = _lit(scale=(4, 2, 3))  # order is free; the model's hparam is [2, 3, 4]
    assert lit.resolved_scales() == [4, 2, 3]

    with pytest.raises(ValueError, match="does not match"):
        _lit(model=_MultiScaleNet(scales=(2, 3)), scale=(2, 3, 4))
    with pytest.raises(ValueError, match="does not match"):
        _lit(model=_PlainNet(scale=2), scale=(2, 3))


def test_predict_scale_must_be_a_positive_int():
    assert SREvalConfig(predict_scale=3).predict_scale == 3
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="predict_scale"):
            SREvalConfig(predict_scale=bad)  # type: ignore[arg-type]


def test_resolved_scale_refuses_when_there_are_several():
    lit = _lit()

    assert lit.resolved_scales() == [2, 3, 4]
    with pytest.raises(ValueError, match="several scales"):
        lit.resolved_scale("an artifact needs one")


def test_a_multi_scale_model_must_be_native_lr():
    class _Bad(_MultiScaleNet):
        input_contract = "pre_upsampled"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="native_lr"):
        _lit(model=_Bad())


def test_srgan_rejects_a_scale_list():
    from dataclasses import dataclass

    @dataclass
    class _Cfg(AdversarialTrainingConfig):
        pass

    with pytest.raises(ValueError, match="one scale"):
        SRGANLightning(
            model=_MultiScaleNet(scales=(2, 4)),
            processor=RGBSignedOutputProcessor(),
            discriminator=SRDiscriminator(),
            training_config=_Cfg(scale=[2, 4]),
        )


# --- metadata / artifact names ---------------------------------------------------------


def test_metadata_records_a_scale_list_and_the_filename_token_sorts_it():
    meta = build_metadata(_lit(scale=(4, 2, 3)))

    assert meta["io"]["scale"] == [4, 2, 3]
    assert artifacts.stem(meta).endswith("x2-3-4_RGB_ms")


def test_metadata_keeps_an_int_for_a_single_scale():
    lit = _lit(model=_PlainNet(scale=4), scale=[4])

    assert build_metadata(lit)["io"]["scale"] == 4


# --- dispatch --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_training_step_hands_the_batch_scale_to_a_multi_scale_model(scale):
    lit = _lit()
    lr, hr = _pair(scale)

    lit._step((lr, hr), need_sr_rgb=False)

    assert lit.model.seen == [scale]  # type: ignore[attr-defined]


def test_a_plain_model_is_never_handed_a_scale():
    lit = _lit(model=_PlainNet(scale=2), scale=(2,))
    lr, hr = _pair(2)

    lit._step((lr, hr))
    lit.predict_rgb(lr, hr, self_ensemble=True)
    lit.predict_step(lr, 0)  # forward(x) would raise TypeError on a stray scale=


def test_self_ensemble_passes_the_scale_on_every_pass():
    lit = _lit()
    lr, hr = _pair(4)

    lit.predict_rgb(lr, hr, self_ensemble=True)

    assert lit.model.seen == [4] * 8  # type: ignore[attr-defined]


def test_the_scale_is_read_off_the_pair_and_a_ragged_pair_is_refused():
    lit = _lit()

    with pytest.raises(ValueError, match="whole multiple"):
        lit._step((torch.rand(1, 3, 5, 5), torch.rand(1, 3, 16, 16)))


def test_predict_needs_predict_scale_when_the_model_has_several():
    lit = _lit()
    lr = torch.rand(1, 3, 8, 8)

    with pytest.raises(ValueError, match="predict_scale"):
        lit.predict_step(lr, 0)

    lit3 = _lit(eval_config=SREvalConfig(crop_border=None, ssim_channels=[], predict_scale=3))
    assert lit3.predict_step(lr, 0).shape[-2:] == (24, 24)
    assert lit3.model.seen == [3]  # type: ignore[attr-defined]


def test_predict_uses_the_only_scale_when_there_is_one():
    lit = _lit(model=_MultiScaleNet(scales=(4,)), scale=(4,))

    assert lit.predict_step(torch.rand(1, 3, 8, 8), 0).shape[-2:] == (32, 32)
    assert lit.model.seen == [4]  # type: ignore[attr-defined]


# --- evaluation ------------------------------------------------------------------------


def test_the_border_follows_each_samples_scale_when_crop_border_is_none():
    """A stripe of error at row 3 survives a 2px border (scale 2) but not a 4px one."""
    sr = torch.full((1, 3, 16, 16), 0.5)
    hr = sr.clone()
    hr[..., 3, :] += 0.25
    scorer = SRScorer(SREvalConfig(crop_border=None, ssim_channels=[]))

    at2 = scorer.score(sr, hr, scale=2).psnr["RGB"]
    at4 = scorer.score(sr, hr, scale=4).psnr["RGB"]

    assert math.isfinite(float(at2)) and math.isinf(float(at4))
    with pytest.raises(ValueError, match="still None at scoring time"):
        scorer.score(sr, hr)


def test_an_explicit_crop_border_wins_over_the_scale():
    sr = torch.full((1, 3, 16, 16), 0.5)
    hr = sr.clone()
    hr[..., 3, :] += 0.25
    scorer = SRScorer(SREvalConfig(crop_border=2, ssim_channels=[]))

    assert scorer.score(sr, hr, scale=2).psnr["RGB"] == scorer.score(sr, hr, scale=4).psnr["RGB"]


def test_a_multi_scale_module_keeps_crop_border_none_and_a_single_one_resolves_it():
    assert _lit().eval_config.crop_border is None
    assert _lit(model=_PlainNet(scale=4), scale=[4]).eval_config.crop_border == 4


def _logged(lit):
    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    lit.log = fake_log  # type: ignore[method-assign]
    return logged


def _val_module():
    lit = _lit(model=_MultiScaleNet(scales=(2, 4)), scale=(2, 4))
    lit._trainer = SimpleNamespace(  # type: ignore[assignment]
        datamodule=SimpleNamespace(n_primary=2, primary_val_scales=[2, 4])
    )
    return lit


def test_multi_scale_validation_logs_each_scale_and_their_mean():
    """Constant error d per scale -> MSE d^2, PSNR 20*log10(1/d): hand-computable."""
    lit = _val_module()
    logged = _logged(lit)
    d = {0: 0.1, 1: 0.2}
    for idx, scale in enumerate((2, 4)):
        lr, up = _pair(scale)
        lit.validation_step((lr, up + d[idx]), 0, dataloader_idx=idx)
    assert logged == {}, "nothing is logged until both scales have run"

    lit.on_validation_epoch_end()

    p2, p4 = 20 * math.log10(1 / 0.1), 20 * math.log10(1 / 0.2)
    assert logged[metric_tag("psnr", "val", "RGB", 2)] == pytest.approx(p2, rel=1e-4)
    assert logged[metric_tag("psnr", "val", "RGB", 4)] == pytest.approx(p4, rel=1e-4)
    assert logged["psnr/val/RGB"] == pytest.approx((p2 + p4) / 2, rel=1e-4)
    assert logged["loss/val/x2"] == pytest.approx(0.01, rel=1e-4)
    assert logged["loss/val/x4"] == pytest.approx(0.04, rel=1e-4)
    assert logged["loss/val"] == pytest.approx(0.025, rel=1e-4)
    assert lit.model.seen == [2, 4]  # type: ignore[attr-defined]


def test_multi_scale_validation_ignores_the_test_loaders_and_resets_between_epochs():
    lit = _val_module()
    logged = _logged(lit)
    lr, up = _pair(2)

    lit.validation_step((lr, up + 0.1), 0, dataloader_idx=2)  # a test loader: not ours
    lit.on_validation_epoch_end()
    assert logged == {}

    lit.validation_step((lr, up + 0.1), 0, dataloader_idx=0)
    lr4, up4 = _pair(4)
    lit.validation_step((lr4, up4 + 0.1), 0, dataloader_idx=1)
    lit.on_validation_epoch_end()
    first = dict(logged)
    logged.clear()
    lit.on_validation_epoch_end()  # nothing left over to log twice
    assert logged == {} and first["loss/val"] == pytest.approx(0.01, rel=1e-4)


def test_single_scale_validation_still_logs_the_plain_tags_in_step():
    lit = _lit(model=_PlainNet(scale=2), scale=[2])
    logged = _logged(lit)
    lr, up = _pair(2)

    lit.validation_step((lr, up + 0.1), 0)

    assert "loss/val" in logged and "psnr/val/RGB" in logged
    assert not any(k.endswith("/x2") for k in logged)


# --- callback ----------------------------------------------------------------------------


def test_benchmark_logger_skips_every_primary_loader_when_mapping_test_sets():
    cb = BenchmarkImageLogger()
    dm = SimpleNamespace(test_names=["Set5_x2", "Set5_x4"], n_primary=3)

    cb.setup(SimpleNamespace(datamodule=dm, loggers=[]), None, "fit")  # type: ignore[arg-type]

    assert cb._val_mapping == {3: "Set5_x2", 4: "Set5_x4"}
    assert cb._test_mapping == {0: "Set5_x2", 1: "Set5_x4"}


def test_benchmark_logger_defaults_to_one_primary_loader():
    cb = BenchmarkImageLogger()
    dm = SimpleNamespace(test_names=["Set5"])

    cb.setup(SimpleNamespace(datamodule=dm, loggers=[]), None, "fit")  # type: ignore[arg-type]

    assert cb._val_mapping == {1: "Set5"}


# --- datamodule --------------------------------------------------------------------------


def _spec(path: Path, scale):
    return {
        "class_path": "sisr.datasets.pre_upsampled.ValidationDataset",
        "init_args": {"img_dir": str(path), "scale": scale},
    }


def _dm(path: Path, **kw) -> SRDataModule:
    train = {
        "class_path": "sisr.datasets.native_lr.TrainDataset",
        "init_args": {
            "img_dir": str(path),
            "scale": [2, 4],
            "hr_crop_size": 16,
            "crops_per_image": 4,
            "cache_dir": str(path / ".c"),
            "build_num_workers": 1,
        },
    }
    kwargs = dict(
        train_dataset=train,
        val_dataset=_spec(path, [2, 4]),
        test_datasets={"Set5": _spec(path, [2, 4]), "Set14": _spec(path, 2)},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )
    return SRDataModule(**(kwargs | kw))  # type: ignore[arg-type]


def test_eval_specs_with_a_scale_list_become_one_loader_per_scale(tiny_rgb_image_dir: Path):
    dm = _dm(tiny_rgb_image_dir)
    dm.setup("fit")

    assert dm.n_primary == 2 and dm.primary_val_scales == [2, 4]
    assert dm.test_names == ["Set5_x2", "Set5_x4", "Set14"]
    loaders = dm.val_dataloader()
    assert [ld.dataset.scale for ld in loaders] == [2, 4, 2, 4, 2]
    assert len(dm.test_dataloader()) == 3
    assert dm.val_dataset is loaders[0].dataset
    assert set(dm.test_datasets) == {"Set5_x2", "Set5_x4", "Set14"}


def test_single_scale_specs_keep_their_names_and_one_primary_loader(tiny_rgb_image_dir: Path):
    dm = _dm(
        tiny_rgb_image_dir,
        val_dataset=_spec(tiny_rgb_image_dir, [2]),
        test_datasets={"Set5": _spec(tiny_rgb_image_dir, 2)},
    )

    assert dm.n_primary == 1 and dm.primary_val_scales == [2]
    assert dm.test_names == ["Set5"]


def test_train_loader_draws_one_scale_per_batch(tiny_rgb_image_dir: Path):
    from sisr.datasets.sampling import ScaleBatchSampler

    dm = _dm(tiny_rgb_image_dir)
    dm.setup("fit")
    loader = dm.train_dataloader()
    torch.manual_seed(0)

    ratios = {hr.shape[-1] // lr.shape[-1] for _ in range(3) for lr, hr in loader}

    assert isinstance(loader.batch_sampler, ScaleBatchSampler)
    assert ratios == {2, 4}


def test_train_loader_for_a_scale_list_needs_a_batch_size(tiny_rgb_image_dir: Path):
    dm = _dm(tiny_rgb_image_dir, train_dataloader_kwargs={"num_workers": 0})
    dm.setup("fit")

    with pytest.raises(ValueError, match="batch_size is required"):
        dm.train_dataloader()


def test_train_loader_honours_drop_last_for_a_scale_list(tiny_rgb_image_dir: Path):
    dm = _dm(tiny_rgb_image_dir, train_dataloader_kwargs={"batch_size": 5, "drop_last": True})
    dm.setup("fit")

    assert len(dm.train_dataloader()) == 12 // 5


def test_the_example_input_shape_check_skips_a_per_batch_scale_dataset(tiny_rgb_image_dir: Path):
    """The LR patch is crop // scale, so one example shape cannot match every batch."""
    dm = _dm(tiny_rgb_image_dir)
    dm.setup("fit")
    lit = _lit(model=_MultiScaleNet(scales=(2, 4)), scale=(2, 4))

    lit._check_example_input_shape((3, 5, 5), torch.zeros(3, 8, 8), dm.train_dataset)  # no raise


# --- export ------------------------------------------------------------------------------


def test_onnx_export_of_a_multi_scale_model_uses_the_first_listed_scale(tmp_path: Path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from sisr.export import to_onnx

    class _NeedsScale(_MultiScaleNet):
        def forward(self, x, scale):  # type: ignore[override]  # no default to fall back on
            return super().forward(x, scale)

    lit = _lit(model=_NeedsScale(scales=(3, 2)), scale=(3, 2))
    path = tmp_path / "m.onnx"

    to_onnx(lit, path, input_sample=torch.zeros(1, 3, 8, 8))

    out = ort.InferenceSession(str(path)).run(None, {"input": torch.rand(1, 3, 6, 6).numpy()})[0]
    assert out.shape == (1, 3, 18, 18)
