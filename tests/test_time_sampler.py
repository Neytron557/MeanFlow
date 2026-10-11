"""Pinned official v1 algorithm and untouched v0 distribution checks."""

import pytest
import torch
from torch import nn

from cifar10.objective import meanflow_loss, sample_times


def _upstream_v1(size, ratio):
    t = torch.sigmoid(torch.randn(size, dtype=torch.float32) * 1.6 - 0.6)
    r = torch.sigmoid(torch.randn(size, dtype=torch.float32) * 1.6 - 4.0)
    r = torch.where(torch.rand(size) < 1 - ratio, t, r)
    return torch.minimum(r, t), t


@pytest.mark.parametrize("ratio", [0.0, 0.75, 1.0])
def test_seeded_v1_exactly_matches_upstream_without_changing_t(ratio):
    torch.manual_seed(41)
    expected_r, expected_t = _upstream_v1(5000, ratio)
    torch.manual_seed(41)
    r, t = sample_times(5000, "cpu", ratio=ratio, time_sampler="v1", time_mu=11, time_sigma=0.1)
    torch.testing.assert_close(r, expected_r, rtol=0, atol=0)
    torch.testing.assert_close(t, expected_t, rtol=0, atol=0)
    assert r.dtype == t.dtype == torch.float32
    assert torch.all((r >= 0) & (r <= t) & (t <= 1))
    if ratio == 0:
        assert torch.equal(r, t)
    elif ratio == 1:
        assert (r == t).any(), "Clamping still creates diagonal samples at ratio=1."
        assert (r < t).any()


def test_v0_default_is_unchanged_seeded_joint_algorithm():
    torch.manual_seed(19)
    first, second = torch.sigmoid(torch.randn(1000) * 2 - 2), torch.sigmoid(torch.randn(1000) * 2 - 2)
    expected_t = torch.maximum(first, second)
    expected_r = torch.where(torch.rand(1000) < 0.25, expected_t, torch.minimum(first, second))
    torch.manual_seed(19)
    r, t = sample_times(1000, "cpu")
    torch.testing.assert_close(r, expected_r, rtol=0, atol=0)
    torch.testing.assert_close(t, expected_t, rtol=0, atol=0)


def test_v1_diagonal_mass_and_time_histograms():
    torch.manual_seed(23)
    r, t = sample_times(100000, "cpu", time_sampler="v1")
    # P(diagonal) = .25 + .75 * P(N(-4,1.6^2) >= N(-.6,1.6^2)).
    assert 0.294 < (r == t).float().mean() < 0.306
    # Independent logit-normal t, never the maximum of two draws.
    assert abs(torch.logit(t).mean().item() + 0.6) < 0.025
    assert abs(torch.logit(t).std().item() - 1.6) < 0.025
    torch.manual_seed(23)
    expected_r, expected_t = _upstream_v1(100000, 0.75)
    for value, expected in ((r, expected_r), (t, expected_t)):
        assert torch.equal(torch.histc(value, bins=10, min=0, max=1),
                           torch.histc(expected, bins=10, min=0, max=1))


def test_invalid_sampler_rejected():
    with pytest.raises(ValueError, match="time_sampler"):
        sample_times(2, "cpu", time_sampler="sorted_v1")


def test_supplied_times_bypass_sampler_dispatch(monkeypatch):
    class Field(nn.Module):
        def forward(self, z, r, t, augment_labels=None):
            return z.square() + t[:, None, None, None]

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit times must bypass random sampling.")

    monkeypatch.setattr("cifar10.objective.sample_times", forbidden)
    t = torch.ones(2)
    images = torch.zeros(2, 1, 2, 2)
    field = Field()
    loss, _ = meanflow_loss(field, field, images, noise=torch.ones_like(images), times=(t, t), time_sampler="v1")
    assert torch.isfinite(loss)
