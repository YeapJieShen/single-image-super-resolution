from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "config.vdsr.template.yaml"


def _cfg():
    return yaml.safe_load(TEMPLATE.read_text(encoding="ascii"))


def test_the_template_carries_the_papers_recipe():
    cfg = _cfg()

    assert cfg["optimizer"]["init_args"] == {"lr": 0.1, "momentum": 0.9, "weight_decay": 1.0e-4}
    sched = cfg["lr_scheduler"]
    assert sched["class_path"].endswith("StepLR")
    assert sched["init_args"] == {"step_size": 2490, "gamma": 0.1}  # /10 every 20 epochs
    assert (
        cfg["trainer"]["max_steps"] == 9960
        and cfg["data"]["train_dataloader_kwargs"]["batch_size"] == 64
    )


def test_the_template_trains_one_multi_scale_model_with_lr_scaled_clipping():
    cfg = _cfg()
    train = cfg["model"]["init_args"]["training_config"]["init_args"]

    assert train["scale"] == [2, 3, 4] and train["scale_clip_with_lr"] is True
    assert cfg["trainer"]["gradient_clip_algorithm"] == "value"
    data = cfg["data"]["train_dataset"]["init_args"]
    assert data["subimg_size"] == 41 and data["stride"] == 41 and data["scale"] == [2, 3, 4]
    for ds in cfg["data"]["test_datasets"].values():
        assert ds["init_args"]["scale"] == [2, 3, 4]  # scored once per scale
