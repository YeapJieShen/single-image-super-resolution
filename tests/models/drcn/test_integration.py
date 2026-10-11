import lightning
import pytest
import torch

from sisr.losses import WeightedSumLoss
from sisr.models.drcn import DRCN, DRCNEvalConfig, DRCNTrainingConfig, RecursiveSupervisionLoss
from sisr.processors import YChannelProcessor
from sisr.training import LossWeightScheduler, SRLightning


def _lit(weights) -> SRLightning:
    torch.manual_seed(0)
    return SRLightning(
        model=DRCN(in_out_channels=1, features=8, recursions=3),
        processor=YChannelProcessor(),
        training_config=DRCNTrainingConfig(scale=2, example_input_shape=(1, 8, 8)),
        eval_config=DRCNEvalConfig(),
        criterion=WeightedSumLoss(
            terms={"recursive": RecursiveSupervisionLoss(), "final": torch.nn.MSELoss()},
            weights=weights,
        ),
    )


def _batch():
    g = torch.Generator().manual_seed(1)
    return torch.rand(2, 3, 8, 8, generator=g), torch.rand(2, 3, 8, 8, generator=g)


def test_the_recursion_term_reaches_the_loss_through_the_real_step():
    lit = _lit({"recursive": 1.0, "final": 0.0})

    loss = lit._step(_batch(), need_sr_rgb=False)[0]

    assert torch.isfinite(loss) and loss.item() > 0
    assert set(lit.criterion.last_terms) == {"recursive", "final"}
    assert lit.criterion.last_terms["recursive"].item() == pytest.approx(loss.item())
    assert lit.criterion.last_terms["final"].item() == 0.0


def test_with_the_recursive_weight_zero_the_loss_is_plain_mse_of_the_primary():
    lit = _lit({"recursive": 0.0, "final": 1.0})
    batch = _batch()

    loss = lit._step(batch, need_sr_rgb=False)[0]

    out = lit._forward_sr(*batch, need_sr_rgb=False)
    sr, hr = out[0].primary, lit.processor.extract_target(out[2])
    assert loss.item() == pytest.approx(torch.nn.functional.mse_loss(sr, hr).item())


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_two_schedulers_move_both_weights_through_a_short_fit():
    lit = _lit({"recursive": 1.0, "final": 0.0})
    batches = [_batch() for _ in range(4)]
    trainer = lightning.Trainer(
        max_steps=4,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            LossWeightScheduler("recursive", 1.0, 0.0, duration=4),
            LossWeightScheduler("final", 0.0, 1.0, duration=4),
        ],
    )
    trainer.fit(lit, torch.utils.data.DataLoader(batches, batch_size=None))

    assert lit.criterion.weights["recursive"] < 0.5 < lit.criterion.weights["final"]
    assert lit.criterion.weights["recursive"] + lit.criterion.weights["final"] == pytest.approx(1.0)


def test_predict_rgb_reads_the_primary_of_a_record_model():
    lit = _lit({"recursive": 1.0, "final": 0.0})
    lr, hr = _batch()

    sr = lit.predict_rgb(lr, hr)[0]

    assert isinstance(sr, torch.Tensor) and sr.shape == lr.shape
