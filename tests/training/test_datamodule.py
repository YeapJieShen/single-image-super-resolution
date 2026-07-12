from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from sisr.training import SRDataModule


def _make_dm(image_dir: Path, *, with_train: bool = True) -> SRDataModule:
    """SRDataModule pointing at a single tiny image dir for train/val/test."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "blur_sigma": 1.0,
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
            "blur_sigma": 1.0,
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
            "blur_sigma": 1.0,
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
            "blur_sigma": 1.0,
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


def test_train_dataset_built_from_class_path_spec(tiny_rgb_image_dir: Path):
    """train_dataset accepts {class_path, init_args} and setup() instantiates it."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "blur_sigma": 1.0,
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
