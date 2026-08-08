import functools
import json

import pytest
import torch

from sisr.losses import TotalVariationLoss, VGG19FeatureLoss, WeightedSumLoss
from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig
from sisr.models.srresnet import SRResNet, SRResNetEvalConfig, SRResNetTrainingConfig
from sisr.processors import RGBProcessor, RGBSignedOutputProcessor, YChannelProcessor
from sisr.training import SRLightning, SRTrainingConfig
from sisr.training.metadata import build_metadata


def _make_srcnn_lit() -> SRLightning:
    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(5, 1, 3), padding=0)
    return SRLightning(
        model=model,
        processor=YChannelProcessor(),
        training_config=SRCNNTrainingConfig(),
        eval_config=SRCNNEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _make_srresnet_lit(*, scale: int = 4) -> SRLightning:
    model = SRResNet(scale=scale, hidden_channel=8, num_residual_blocks=1)
    return SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRResNetTrainingConfig(),
        eval_config=SRResNetEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_build_metadata_top_level_shape():
    lit = _make_srcnn_lit()
    meta = build_metadata(lit)
    assert set(meta.keys()) == {
        "format",
        "created",
        "versions",
        "model",
        "processor",
        "criterion",
        "io",
        "eval_config",
        "training",
    }
    assert meta["format"] == "sisr-meta-v1"


def test_build_metadata_versions_are_plain_strings():
    meta = build_metadata(_make_srcnn_lit())
    for value in meta["versions"].values():
        assert type(value) is str  # not e.g. torch's TorchVersion (a str subclass)


def test_build_metadata_model_class_path_and_init_args():
    meta = build_metadata(_make_srcnn_lit())
    assert meta["model"]["class_path"] == "sisr.models.srcnn.model.SRCNN"
    assert meta["model"]["init_args"] == {
        "num_channels": 1,
        "num_filters": [8, 4],
        "kernel_sizes": [5, 1, 3],
        "padding": 0,
    }


def test_build_metadata_model_init_args_has_no_tuples():
    """SRCNN/SRResNet hparams store tuples (num_filters, kernel_sizes); the
    metadata dict must convert them to lists — JSON-safe and unambiguous
    under torch.load(weights_only=True) (which pickles tuple/list differently)."""
    meta = build_metadata(_make_srresnet_lit())
    init_args = meta["model"]["init_args"]
    assert isinstance(init_args["kernel_sizes"], list)
    assert not isinstance(init_args["kernel_sizes"], tuple)


def test_build_metadata_processor_class_path():
    meta = build_metadata(_make_srresnet_lit())
    assert meta["processor"]["class_path"] == "sisr.processors.rgb.RGBSignedOutputProcessor"


def test_build_metadata_io_reflects_model_input_contract_and_processor():
    srcnn_meta = build_metadata(_make_srcnn_lit())
    assert srcnn_meta["io"]["input"] == "pre_upsampled"
    assert srcnn_meta["io"]["input_channels"] == 1  # YChannelProcessor.model_channels
    assert srcnn_meta["io"]["output_range"] == [0.0, 1.0]
    assert srcnn_meta["io"]["output_colorspace"] == "Y"

    srresnet_meta = build_metadata(_make_srresnet_lit())
    assert srresnet_meta["io"]["input"] == "native_lr"
    assert srresnet_meta["io"]["input_channels"] == 3  # RGBSignedOutputProcessor.model_channels
    assert srresnet_meta["io"]["output_range"] == [-1.0, 1.0]  # the asymmetric paper range
    assert srresnet_meta["io"]["output_colorspace"] == "RGB"


def test_build_metadata_scale_prefers_training_config_scale():
    """SRResNetTrainingConfig defaults scale=4, matching the model's own scale
    hparam (SRLightning.__init__ validates the two agree)."""
    lit = _make_srresnet_lit(scale=4)
    assert lit.training_config.scale == 4
    meta = build_metadata(lit)
    assert meta["io"]["scale"] == 4


def test_build_metadata_scale_falls_back_to_model_hparams():
    """The generic SRTrainingConfig defaults scale=None -> falls back to the
    model's own 'scale' hparam instead of leaving it unrecorded."""
    model = SRResNet(scale=2, hidden_channel=8, num_residual_blocks=1)
    lit = SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRTrainingConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit.training_config.scale is None
    meta = build_metadata(lit)
    assert meta["io"]["scale"] == 2


def test_build_metadata_scale_is_none_when_neither_source_has_it():
    """SRCNN has no 'scale' hparam and training_config.scale is unset -> None,
    not a guessed value."""
    meta = build_metadata(_make_srcnn_lit())
    assert meta["io"]["scale"] is None


def test_metadata_records_ssim_impl():
    """Which SSIM produced a checkpoint's filename must be answerable from the
    artifact alone. eval_config is serialised wholesale, so this rides along —
    the assertion exists so a future refactor can't quietly drop it."""
    meta = build_metadata(_make_srresnet_lit())
    assert meta["eval_config"]["ssim_impl"] == "daala"


def test_build_metadata_eval_config_matches_dataclass_asdict():
    import dataclasses

    lit = _make_srcnn_lit()
    meta = build_metadata(lit)
    assert meta["eval_config"] == dataclasses.asdict(lit.eval_config)


def test_build_metadata_training_fields_default_to_none():
    meta = build_metadata(_make_srcnn_lit())
    assert meta["training"] == {
        "global_step": None,
        "epoch": None,
        "monitor": None,
        "monitor_value": None,
    }


def test_build_metadata_training_fields_forwarded():
    meta = build_metadata(
        _make_srcnn_lit(),
        global_step=1000,
        epoch=5,
        monitor="psnr/val/RGB",
        monitor_value=30.5,
    )
    assert meta["training"] == {
        "global_step": 1000,
        "epoch": 5,
        "monitor": "psnr/val/RGB",
        "monitor_value": 30.5,
    }


def test_build_metadata_omits_dataset_paths():
    """Deliberate: no dataset directory/path ever appears in the metadata tree,
    unlike e.g. Ultralytics' train_args (which leaks local filesystem layout)."""
    meta = build_metadata(_make_srcnn_lit())
    dumped = json.dumps(meta)
    for leaky_token in ("img_dir", "cache_dir", "data/", "C:\\", "/home/"):
        assert leaky_token not in dumped


def test_build_metadata_is_weights_only_safe(tmp_path):
    """The whole tree must round-trip through torch.save/load(weights_only=True) —
    the contract every checkpoint sink depends on."""
    meta = build_metadata(
        _make_srresnet_lit(),
        global_step=42,
        epoch=1,
        monitor="ssim/val/RGB",
        monitor_value=0.87,
    )
    path = tmp_path / "meta.pt"
    torch.save({"meta": meta}, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["meta"] == meta


def test_build_metadata_every_field_json_encodable_individually():
    """Mirrors sisr.export's per-field ONNX metadata_props encoding: every
    top-level value must either be a str already, or survive json.dumps."""
    meta = build_metadata(_make_srresnet_lit(), global_step=1, epoch=0)
    for value in meta.values():
        encoded = value if isinstance(value, str) else json.dumps(value)
        assert isinstance(encoded, str)
        if not isinstance(value, str):
            assert json.loads(encoded) == value


def test_metadata_records_the_criterion_identity():
    """Provenance for 'which loss produced these weights'. A VGG-trained model
    and an MSE-trained one are indistinguishable from the tensors alone."""
    with pytest.warns(UserWarning, match="randomly initialised"):
        vgg = VGG19FeatureLoss(layer="vgg22", weights=None)
    module = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBSignedOutputProcessor(),
        criterion=WeightedSumLoss(
            terms={"vgg22": vgg, "tv": TotalVariationLoss()},
            weights={"vgg22": 1.0, "tv": 2.0e-8},
        ),
    )

    meta = build_metadata(module)

    assert meta["criterion"]["class_path"] == "sisr.losses.composite.WeightedSumLoss"
    assert meta["criterion"]["description"] == "1*vgg22 + 2e-08*tv"


def test_metadata_criterion_defaults_to_the_class_name_for_a_plain_loss():
    module = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
    )

    meta = build_metadata(module)

    assert meta["criterion"] == {
        "class_path": "torch.nn.modules.loss.MSELoss",
        "description": "MSELoss",
    }


def test_metadata_stays_weights_only_loadable_with_a_criterion_block(tmp_path):
    """Every metadata value must remain a plain type — the checkpoint contract."""
    module = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
    )
    path = tmp_path / "meta.pt"
    torch.save({"meta": build_metadata(module)}, path)

    loaded = torch.load(path, weights_only=True)

    assert loaded["meta"]["criterion"]["description"] == "MSELoss"
