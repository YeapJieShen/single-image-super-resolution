import pytest
import torch

from sisr.models.base import SRModelOutput
from sisr.models.drcn import RecursiveSupervisionLoss


def test_recursive_loss_is_the_mean_over_every_recursion():
    g = torch.Generator().manual_seed(0)
    recs = torch.rand(2, 4, 1, 5, 5, generator=g)
    target = torch.rand(2, 1, 5, 5, generator=g)
    record = SRModelOutput(primary=recs.mean(1), extras={"recursions": recs})

    got = RecursiveSupervisionLoss()(record, target)

    manual = torch.stack([((recs[:, d] - target) ** 2).mean() for d in range(4)]).mean()
    assert got.item() == pytest.approx(manual.item())
    assert RecursiveSupervisionLoss.wants_model_output is True


def test_every_recursion_receives_gradient():
    recs = torch.rand(1, 3, 1, 4, 4, requires_grad=True)
    record = SRModelOutput(primary=recs.mean(1), extras={"recursions": recs})

    RecursiveSupervisionLoss()(record, torch.zeros(1, 1, 4, 4)).backward()

    assert all(recs.grad[:, d].abs().sum() > 0 for d in range(3))


def test_a_bare_tensor_is_a_clear_error():
    with pytest.raises(TypeError, match="SRModelOutput"):
        RecursiveSupervisionLoss()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
