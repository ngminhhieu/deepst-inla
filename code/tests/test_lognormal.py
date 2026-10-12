import torch

from dengue_h3.models import lognormal_loss


def test_lognormal_loss_is_minimized_at_log1p_target():
    target = torch.tensor([[[0.0, 9.0, 99.0]]])
    optimum = torch.log1p(target)
    scale = torch.ones_like(target)
    weights = torch.ones(3)
    exact = lognormal_loss(target, optimum, scale, weights)
    shifted = lognormal_loss(target, optimum + 1.0, scale, weights)
    assert exact < shifted


def test_lognormal_loss_respects_horizon_weights():
    target = torch.zeros((1, 1, 3))
    location = torch.tensor([[[0.0, 0.0, 2.0]]])
    scale = torch.ones_like(target)
    low_h3 = lognormal_loss(target, location, scale, torch.tensor([1.0, 1.0, 0.1]))
    high_h3 = lognormal_loss(target, location, scale, torch.tensor([1.0, 1.0, 2.0]))
    assert low_h3 < high_h3
