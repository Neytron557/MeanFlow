"""Independent analytic checks for original-MeanFlow research measurements."""

import json
import math
import time

import numpy as np
import pytest
import torch

from cifar10.data import SyntheticCIFAR10
from cifar10.diagnostics import (
    FixedProbe, adaptive_error_summary, build_time_bin_probes, flow_gradient_alignment,
    run_endpoint_mean_probe, run_research_diagnostics, run_time_bin_probe, training_mean_image,
)
from cifar10.evaluate import main, parse_args
from cifar10.sampling import flow_sample, meanflow_sample, sampling_nfe, to_uint8


class LinearField(torch.nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.calls = []

    def forward(self, z, r, t, augment_labels=None):
        self.calls.append((r.detach().clone(), t.detach().clone(), augment_labels))
        return z * self.scale


@pytest.mark.parametrize("steps,solver,nfe", [(32, "euler", 32), (64, "euler", 64), (32, "heun", 64)])
def test_diagonal_ode_has_exact_calls_and_analytic_convergence(steps, solver, nfe):
    model = LinearField(0.7)
    noise = torch.full((2, 3, 32, 32), 3.0)
    result = flow_sample(model, noise, steps, field="diagonal", solver=solver, expected_nfe=nfe)
    factor = 1 - 0.7 / steps
    if solver == "heun":
        factor += 0.5 * (0.7 / steps) ** 2
    torch.testing.assert_close(result, noise * factor ** steps)
    assert len(model.calls) == nfe
    assert all(torch.equal(r, t) and labels is None for r, t, labels in model.calls)
    assert result.dtype == torch.float32 and not result.requires_grad
    assert model.calls[0][1].eq(1).all()
    if solver == "heun":
        assert model.calls[-1][1].eq(0).all()
        assert (result - noise * math.exp(-0.7)).abs().max() < 0.0003
    # This stays >1 during integration; only final pixel conversion clips it.
    assert result.min() > 1
    assert to_uint8(result).eq(255).all()


def test_default_average_path_is_bitwise_unchanged():
    noise = torch.randn(3, 3, 32, 32)
    old = meanflow_sample(LinearField(), noise, 7)
    new = flow_sample(LinearField(), noise, 7)
    assert torch.equal(old, new)


@pytest.mark.parametrize("kwargs", [
    {"field": "avg", "solver": "heun"}, {"field": "unknown"}, {"solver": "rk4"},
    {"field": "diagonal", "solver": "heun", "expected_nfe": 32}, {"expected_nfe": True},
])
def test_sampling_rejects_ambiguous_or_wrong_nfe(kwargs):
    with pytest.raises(ValueError):
        sampling_nfe(32, **kwargs)


def test_evaluation_cli_reports_heun_true_nfe_and_rejects_wrong_count():
    args = parse_args(["--ckpt", "x", "--field", "diagonal", "--solver", "heun",
                       "--num-steps", "32", "--nfe", "64"])
    assert args.nfe == 64 and len(args.times) == 33
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "x", "--field", "diagonal", "--solver", "heun",
                    "--num-steps", "64", "--nfe", "64"])


def test_tiny_heun_evaluation_metadata_distinguishes_diagnostic(tmp_path):
    from cifar10.model import create_model

    checkpoint = tmp_path / "weights.pt"
    torch.save({"ema": create_model("DiT-Tiny/4").state_dict(),
                "config": {"model": "DiT-Tiny/4", "augment_dim": 6}, "step": 1}, checkpoint)
    output = tmp_path / "evaluation"
    main(parse_args(["--ckpt", str(checkpoint), "--sample-dir", str(output),
                     "--device", "cpu", "--field", "diagonal", "--solver", "heun",
                     "--num-steps", "2", "--nfe", "4", "--num-fid-samples", "2"]))
    receipt = json.loads((output / "metrics.json").read_text())
    assert receipt["nfe"] == 4 and receipt["num_steps"] == 2
    assert receipt["field"] == "diagonal" and receipt["solver"] == "heun"
    assert receipt["evaluation_role"] == "diagonal_field_diagnostic"
    assert receipt["pixel_conversion"].startswith("floor")


def test_declared_training_mean_excludes_held_out_raw_rows():
    class RawDataset:
        data = np.stack([np.full((32, 32, 3), value, dtype=np.uint8) for value in (0, 64, 255)])

        def __len__(self):
            return len(self.data)

    dataset = RawDataset()
    subset = torch.utils.data.Subset(dataset, [0, 1])
    mean = training_mean_image(subset)
    torch.testing.assert_close(mean, torch.full((3, 32, 32), 32 / 127.5 - 1))
    dataset.data[2] = 0
    assert torch.equal(mean, training_mean_image(subset))
    with pytest.raises(ValueError, match="distinct"):
        training_mean_image(dataset, indices=[0, 0])


def test_medium_bins_include_both_exact_endpoints_and_preserve_rng():
    rng = torch.get_rng_state().clone()
    probes = build_time_bin_probes(SyntheticCIFAR10(size=37), "cpu", count=16)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(probes[0].images) == 16
    pairs = {(float(p.r[0]), float(p.t[0])) for p in probes}
    assert {(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)} <= pairs
    assert all(torch.equal(p.images, probes[0].images) and torch.equal(p.noise, probes[0].noise)
               for p in probes)


def test_time_bin_residual_and_adaptive_weights_follow_original_target():
    image = torch.full((2, 3, 32, 32), 0.2)
    noise = torch.full_like(image, 0.8)
    probe = FixedProbe(image, noise, torch.zeros(2), torch.ones(2), torch.zeros(2, 6), (1, 2), 7)
    # u=.5epsilon; derivative=.5(epsilon-x); residual=.5x exactly at r0,t1.
    result = run_time_bin_probe(LinearField(), probe)
    assert result["fp32_finite"]
    assert result["residual"]["rms"] == pytest.approx(0.1, abs=1e-7)
    assert result["per_image_mse"] == pytest.approx([0.01, 0.01])
    assert all(row["effective_sample_size"] == pytest.approx(2) for row in result["adaptive_weights"])
    assert result["fp16_finite"] is None
    json.dumps(result, allow_nan=False)


def test_exact_endpoint_mean_and_conditional_sample_decomposition():
    model = LinearField()
    noise = torch.randn(2, 3, 32, 32)
    images = torch.stack([torch.full((3, 32, 32), value) for value in (-0.8, -0.2, 0.3, 0.8)])
    mean = images.mean(0)
    result = run_endpoint_mean_probe(model, noise, mean, conditional_images=images)
    assert result["mean_identity_residual"]["rms"] == pytest.approx(0.5 * float(mean.abs().mean()), abs=1e-7)
    weighting = result["conditional_weighting"]
    assert weighting["unweighted_sample_mean_error_vs_analytic"]["rms"] < 1e-7
    zero = next(row for row in weighting["powers"] if row["p"] == 0)
    assert zero["ess_per_noise"] == pytest.approx([4, 4])
    assert zero["error_vs_analytic_mean"]["rms"] < 1e-7
    stronger = next(row for row in weighting["powers"] if row["p"] == 1)
    assert max(stronger["ess_per_noise"]) < 4
    assert stronger["error_vs_analytic_mean"]["rms"] > 0.01
    json.dumps(result, allow_nan=False)


def test_adaptive_ess_and_nonfinite_diagnostics_are_json_safe():
    summary = adaptive_error_summary(torch.tensor([1.0, 9.0, float("inf")]))
    assert summary[0]["effective_sample_size"] == pytest.approx(2)
    assert all(row["excluded_rows"] == 1 for row in summary)
    assert summary[-1]["effective_sample_size"] < 2
    json.dumps(summary, allow_nan=False)


def test_gradient_alignment_restores_parameter_flags_modes_and_existing_grads():
    model = LinearField().train()
    model.scale.requires_grad_(False)
    model.scale.grad = torch.tensor(3.0)
    probes = build_time_bin_probes(SyntheticCIFAR10(size=4), "cpu", count=2, pairs=((0, 0.8),))
    rng = torch.get_rng_state().clone()
    parameter = model.scale.detach().clone()
    result = flow_gradient_alignment(model, probes[0], count=2)
    assert result["available"] and abs(result["parameter_gradient_cosine"]) == pytest.approx(1)
    assert model.training and not model.scale.requires_grad
    assert model.scale.grad == 3 and torch.equal(model.scale, parameter)
    assert torch.equal(rng, torch.get_rng_state())


def test_research_deadline_does_not_call_model_after_budget_expires():
    model = LinearField()
    probes = build_time_bin_probes(SyntheticCIFAR10(size=4), "cpu", count=2)
    result = run_research_diagnostics(model, probes, torch.zeros(3, 32, 32), deadline=time.monotonic() - 1)
    assert result == {"status": "budget_limited", "time_bins": []}
    assert not model.calls
