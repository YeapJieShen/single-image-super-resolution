from pathlib import Path
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from sisr.training import SRDataModule
from sisr.training.datamodule import _accepted_init_args


def _make_dm(image_dir: Path, *, with_train: bool = True) -> SRDataModule:
    """SRDataModule pointing at a single tiny image dir for train/val/test."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(image_dir / ".lmdb_cache_train"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    test_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": test_spec, "Set14": test_spec},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        test_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


def test_test_names_property_in_order(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    assert dm.test_names == ["Set5", "Set14"]


def test_setup_fit_instantiates_train_val_and_tests(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="fit")
    assert dm._train_ds is not None
    assert dm._val_ds is not None
    assert set(dm._test_ds) == {"Set5", "Set14"}


def test_setup_validate_skips_train(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="validate")
    assert dm._train_ds is None, "validate stage must not build train dataset"
    assert dm._val_ds is not None
    assert set(dm._test_ds) == {"Set5", "Set14"}


def test_setup_test_only_builds_tests(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="test")
    assert dm._train_ds is None
    assert dm._val_ds is None, "test stage doesn't need primary val"
    assert set(dm._test_ds) == {"Set5", "Set14"}


def test_train_dataloader_shuffles(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    assert isinstance(loader, DataLoader)
    # DataLoader.sampler is RandomSampler when shuffle=True.
    from torch.utils.data import RandomSampler

    assert isinstance(loader.sampler, RandomSampler)


def test_val_dataloader_returns_primary_then_tests(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="fit")
    loaders = dm.val_dataloader()
    assert len(loaders) == 1 + len(dm.test_names)
    # idx 0 is primary val; idx 1+ are test sets in test_names order.
    assert loaders[0].dataset is dm._val_ds
    for i, name in enumerate(dm.test_names):
        assert loaders[i + 1].dataset is dm._test_ds[name]


def test_test_dataloader_returns_test_only(tiny_rgb_image_dir: Path):
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="test")
    loaders = dm.test_dataloader()
    assert len(loaders) == len(dm.test_names)
    for i, name in enumerate(dm.test_names):
        assert loaders[i].dataset is dm._test_ds[name]


def test_test_dataloader_kwargs_falls_back_to_val(tiny_rgb_image_dir: Path):
    """When test_dataloader_kwargs is omitted, the datamodule reuses
    val_dataloader_kwargs."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_train_fb"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": val_spec},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 7, "num_workers": 0},
        test_dataloader_kwargs=None,
    )
    assert dm._test_dl_kwargs == dm._val_dl_kwargs
    assert dm._test_dl_kwargs["batch_size"] == 7


def test_no_test_datasets_val_dataloader_returns_only_primary(tiny_rgb_image_dir: Path):
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_only_primary"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets=None,
        train_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )
    dm.setup(stage="fit")
    loaders = dm.val_dataloader()
    assert len(loaders) == 1


def test_old_class_params_rejected(tiny_rgb_image_dir: Path):
    """The legacy `train_dataset_class` etc. params no longer exist."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_train_legacy"),
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    with pytest.raises(TypeError):
        SRDataModule(
            train_dataset=train_spec,
            val_dataset=val_spec,
            train_dataset_class=object,  # legacy param — must not exist
        )


def _predict_spec(image_dir: Path) -> dict:
    return {
        "class_path": "sisr.datasets.predict.PredictDataset",
        "init_args": {"img_dir": str(image_dir)},
    }


def test_setup_predict_only_builds_predict_dataset(tiny_rgb_image_dir: Path):
    """stage='predict' must build only the predict dataset, not train/val/test —
    mirroring the other stages' selective-instantiation contract."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_predict_stage"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": val_spec},
        predict_dataset=_predict_spec(tiny_rgb_image_dir),
    )
    dm.setup(stage="predict")
    assert dm._train_ds is None
    assert dm._val_ds is None
    assert dm._test_ds == {}
    assert dm._predict_ds is not None


def test_predict_dataloader_returns_loader_over_predict_dataset(tiny_rgb_image_dir: Path):
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_predict_loader"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        predict_dataset=_predict_spec(tiny_rgb_image_dir),
        predict_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.dataset is dm._predict_ds
    from torch.utils.data import SequentialSampler

    assert isinstance(loader.sampler, SequentialSampler)  # shuffle=False


def test_predict_dataloader_default_kwargs(tiny_rgb_image_dir: Path):
    """batch_size=1/num_workers=0 default when predict_dataloader_kwargs is omitted."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_predict_default"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        predict_dataset=_predict_spec(tiny_rgb_image_dir),
    )
    assert dm._predict_dl_kwargs == {"batch_size": 1, "num_workers": 0}


@pytest.mark.parametrize("loader", ["train", "val"])
def test_dataloader_before_setup_names_the_stage_that_was_missed(
    tiny_rgb_image_dir: Path, loader: str
):
    """Loader called before setup() -> a RuntimeError naming setup, not a NoneType crash.

    Lightning calls setup() itself, so this only fires for a direct caller — a test,
    a notebook, or a script driving the module by hand. Before the guard the call
    **did not raise at all**: DataLoader wraps ``None`` without complaint because its
    sampler only takes ``len()`` on iteration, so the failure surfaced later and
    somewhere else, naming neither the module nor the call that was skipped.
    """
    dm = _make_dm(tiny_rgb_image_dir)
    with pytest.raises(RuntimeError, match=r"setup\('fit'\)"):
        getattr(dm, f"{loader}_dataloader")()


def test_predict_dataloader_without_predict_dataset_raises(tiny_rgb_image_dir: Path):
    """No predict_dataset configured -> a clear RuntimeError, not a silent None loader."""
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="predict")
    with pytest.raises(RuntimeError, match="predict_dataset"):
        dm.predict_dataloader()


# ---------------------------------------------------------------------------
# Dataset spec validation — catches malformed {class_path, init_args} dicts,
# including the shape a CLI dotted override produces when it can't reach a
# nested init_args key on this dict[str, Any]-typed field (see test_cli.py's
# test_dataset_cli_override_fails_loudly_instead_of_silently_ignored).
# ---------------------------------------------------------------------------


def test_train_dataset_spec_rejects_stray_sibling_key(tiny_rgb_image_dir: Path):
    """A key alongside class_path/init_args — exactly what a dotted CLI
    override produces when it can't reach nested init_args — must raise
    instead of silently building the wrong dataset."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
        },
        "crops_per_image": 8,  # stray — never reaches init_args
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    with pytest.raises(ValueError, match="crops_per_image"):
        SRDataModule(train_dataset=train_spec, val_dataset=val_spec)


def test_val_dataset_spec_rejects_stray_sibling_key(tiny_rgb_image_dir: Path):
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
        "scale": 3,
    }
    with pytest.raises(ValueError, match="val_dataset"):
        SRDataModule(train_dataset=train_spec, val_dataset=val_spec)


def test_test_datasets_spec_rejects_stray_sibling_key_names_the_entry(tiny_rgb_image_dir: Path):
    """The error must name which test_datasets entry is malformed — 'Set14',
    not just 'test_datasets' — since there can be several."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    bad_test_spec = {**val_spec, "scale": 4}
    with pytest.raises(ValueError, match="Set14"):
        SRDataModule(
            train_dataset=train_spec,
            val_dataset=val_spec,
            test_datasets={"Set5": val_spec, "Set14": bad_test_spec},
        )


def test_predict_dataset_spec_rejects_stray_sibling_key(tiny_rgb_image_dir: Path):
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    bad_predict_spec = {
        "class_path": "sisr.datasets.predict.PredictDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir)},
        "extra": 1,
    }
    with pytest.raises(ValueError, match="predict_dataset"):
        SRDataModule(
            train_dataset=train_spec, val_dataset=val_spec, predict_dataset=bad_predict_spec
        )


def test_predict_dataset_none_skips_validation(tiny_rgb_image_dir: Path):
    """predict_dataset=None (the default, unconfigured predict) must not be
    validated as if it were a malformed spec."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    SRDataModule(train_dataset=train_spec, val_dataset=val_spec)  # must not raise


def test_dataset_spec_missing_class_path_rejected(tiny_rgb_image_dir: Path):
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    with pytest.raises(ValueError, match="class_path"):
        SRDataModule(train_dataset={"init_args": {"scale": 2}}, val_dataset=val_spec)


def test_train_dataset_built_from_class_path_spec(tiny_rgb_image_dir: Path):
    """train_dataset accepts {class_path, init_args} and setup() instantiates it."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(tiny_rgb_image_dir / ".lmdb_cache_train_cp"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    test_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": test_spec},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        test_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )
    dm.setup(stage="fit")
    from sisr.datasets.srcnn import TrainDataset, ValidationDataset

    assert isinstance(dm._train_ds, TrainDataset)
    assert isinstance(dm._val_ds, ValidationDataset)
    assert isinstance(dm._test_ds["Set5"], ValidationDataset)


def test_unknown_init_args_key_names_the_config_field_and_the_valid_keys(tmp_path):
    """_validate_spec catches keys that land *beside* init_args. A misspelled
    key *inside* it reaches the dataset constructor instead and surfaces as a
    bare TypeError naming the class but not the config path — which of several
    dataset blocks is wrong is exactly what the reader needs."""
    dm = SRDataModule(
        train_dataset={
            "class_path": "sisr.datasets.srresnet.ValidationDataset",
            "init_args": {"img_dir": str(tmp_path), "scale": 2, "crops_per_imag": 8},
        },
        val_dataset={
            "class_path": "sisr.datasets.srresnet.ValidationDataset",
            "init_args": {"img_dir": str(tmp_path), "scale": 2},
        },
    )

    with pytest.raises(ValueError) as excinfo:
        dm.setup(stage="fit")

    message = str(excinfo.value)
    assert "data.train_dataset.init_args" in message, "must name the offending config field"
    assert "crops_per_imag" in message, "must name the key that was not accepted"
    assert "img_dir" in message, "must list what the target does accept"
    assert isinstance(excinfo.value.__cause__, TypeError), "the original error must be preserved"


def test_an_unrelated_type_error_from_a_dataset_constructor_is_not_reworded(tmp_path):
    """Only the unexpected-keyword case is translated. A TypeError raised by
    the constructor's own logic must propagate untouched, or a real bug gets
    reported as a config mistake."""

    def _boom(*args, **kwargs):
        raise TypeError("something else entirely")

    with patch("sisr.training.datamodule.instantiate_class", side_effect=_boom):
        dm = SRDataModule(
            train_dataset={"class_path": "sisr.datasets.srresnet.ValidationDataset"},
            val_dataset={"class_path": "sisr.datasets.srresnet.ValidationDataset"},
        )
        with pytest.raises(TypeError, match="something else entirely"):
            dm.setup(stage="fit")


def test_non_mapping_init_args_is_refused_by_name(tmp_path):
    """init_args typed as anything but a mapping cannot be splatted into the
    constructor, so it is caught at validation rather than at instantiation."""
    with pytest.raises(ValueError, match=r"data\.train_dataset\.init_args must be a mapping"):
        SRDataModule(
            train_dataset={
                "class_path": "sisr.datasets.srresnet.ValidationDataset",
                "init_args": ["img_dir", str(tmp_path)],
            },
            val_dataset={"class_path": "sisr.datasets.srresnet.ValidationDataset"},
        )


@pytest.mark.parametrize(
    "class_path",
    ["sisr.datasets.srresnet.NoSuchDataset", "no_such_module.Thing", "not-a-dotted-path"],
)
def test_accepted_init_args_gives_up_quietly_on_an_unresolvable_class(class_path):
    """The accepted-keys lookup is a courtesy on the error path. If the class
    cannot be resolved or inspected it must return None so the caller still
    reports the original constructor error, rather than replacing a real
    failure with an import error raised while composing the message."""
    assert _accepted_init_args(class_path) is None


# ---------------------------------------------------------------------------
# test_dataloader must not fail silently
# ---------------------------------------------------------------------------


def _make_dm_no_test_sets(image_dir: Path) -> SRDataModule:
    """Same shape as _make_dm, with test_datasets left unset."""
    return SRDataModule(
        train_dataset={
            "class_path": "sisr.datasets.srcnn.TrainDataset",
            "init_args": {
                "img_dir": str(image_dir),
                "subimg_size": 33,
                "stride": 14,
                "scale": 2,
                "use_tqdm": False,
                "cache_dir": str(image_dir / ".lmdb_cache_nt"),
                "build_num_workers": 1,
            },
        },
        val_dataset={
            "class_path": "sisr.datasets.srcnn.ValidationDataset",
            "init_args": {"img_dir": str(image_dir), "scale": 2},
        },
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        # test_datasets and test_dataloader_kwargs both deliberately unset
    )


def test_test_dataloader_before_setup_raises_when_test_sets_are_configured(
    tiny_rgb_image_dir: Path,
):
    """Its two siblings raise; this one returned [] silently, and it is the one
    whose failure matters most.

    The two states were indistinguishable: "configured with Set5, setup() never
    run" and "configured with no test sets at all" both gave []. A `test` run in
    the first state evaluates NOTHING and reports success -- and that is the
    final evaluation path, so the number that never gets produced is the one
    someone would publish."""
    dm = _make_dm(tiny_rgb_image_dir)
    with pytest.raises(RuntimeError, match="setup"):
        dm.test_dataloader()


def test_test_dataloader_still_returns_empty_when_no_test_sets_configured(
    tiny_rgb_image_dir: Path,
):
    """That state is legitimate and must stay silent -- a fit run with no
    benchmark sets is a normal thing to configure. It must not become an error
    just because the other state now is one."""
    dm = _make_dm_no_test_sets(tiny_rgb_image_dir)
    assert dm.test_dataloader() == []
    dm.setup(stage="test")
    assert dm.test_dataloader() == []


def test_test_dataloader_after_setup_is_unchanged(tiny_rgb_image_dir: Path):
    """The working path must keep working: one loader per entry, in order."""
    dm = _make_dm(tiny_rgb_image_dir)
    dm.setup(stage="test")
    loaders = dm.test_dataloader()
    assert len(loaders) == 2
    assert all(isinstance(dl, DataLoader) for dl in loaders)


def test_val_and_test_dataloader_kwargs_are_not_the_same_object(tiny_rgb_image_dir: Path):
    """`self._test_dl_kwargs is self._val_dl_kwargs` was True whenever
    test_dataloader_kwargs was unset -- the same dict object, not a copy.
    Nothing mutates them today, so it costs nothing now and couples the two
    loaders silently the first time anything does."""
    dm = _make_dm_no_test_sets(tiny_rgb_image_dir)
    assert dm._test_dl_kwargs == dm._val_dl_kwargs
    assert dm._test_dl_kwargs is not dm._val_dl_kwargs
