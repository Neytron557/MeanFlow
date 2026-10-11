"""Fixed probes observe numerical health without altering training state."""

import json

import pytest
import torch

from cifar10.data import SyntheticCIFAR10
from cifar10.diagnostics import attention_logit_max, build_fixed_probe, run_fixed_probe
from cifar10.model import create_model


def nonlinear_model():
    model = create_model("DiT-Tiny/4", attn_projection_dropout=0.1, mlp_output_dropout=0.1)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.03)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.04)
    return model


def test_probe_inputs_are_fixed_and_do_not_consume_rng():
    dataset = SyntheticCIFAR10(size=17)
    rng = torch.get_rng_state().clone()
    first = build_fixed_probe(dataset, "cpu")
    second = build_fixed_probe(dataset, "cpu")
    assert torch.equal(torch.get_rng_state(), rng)
    assert first.indices == (0, 2, 4, 6, 8, 10, 12, 14)
    for field in ("images", "noise", "r", "t", "augment_labels"):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert not first.augment_labels.any()
    assert torch.all(first.r < first.t)


def test_fixed_probe_reports_nontrivial_corrections_and_preserves_mode_rng_gradients():
    torch.manual_seed(9)
    model = nonlinear_model().train()
    model.blocks[0].mlp.eval()  # Preserve heterogeneous modes exactly.
    probe = build_fixed_probe(SyntheticCIFAR10(size=9), "cpu", count=3)
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    parameters = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    result = run_fixed_probe(model, probe)
    json.dumps(result, allow_nan=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in model.modules()] == modes
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in parameters.items())
    assert result["fp32_finite"] is True
    assert result["fp16_finite"] is None
    assert result["precision_comparison"] == "unavailable_on_cpu"
    assert result["correction_abs_p99"] > 0
    assert result["dataset_role"] == "training_diagnostic_not_holdout"
    for block in result["blocks"]:
        assert abs(block["centered_logits"]["mean"]) < 1e-6
        assert block["q_activation"]["rms"] > 0
        assert block["k_weight_frobenius"] > 0
        assert block["attention_entropy"]["rms"] > 0
        assert block["adaln"]["gate_msa"]["rms"] > 0
    assert run_fixed_probe(model, probe) == result


def test_legacy_attention_probe_is_safe_with_training_dropout():
    model = nonlinear_model().train()
    probe = build_fixed_probe(SyntheticCIFAR10(size=2), "cpu", count=2)
    rng = torch.get_rng_state().clone()
    values = attention_logit_max(model, probe.images, probe.r, probe.t, probe.augment_labels)
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(values) == len(model.blocks)
    assert all(value > 0 for value in values)


def test_probe_exception_removes_hooks_and_restores_modes_rng(monkeypatch):
    model = nonlinear_model().train()
    probe = build_fixed_probe(SyntheticCIFAR10(size=2), "cpu", count=2)
    rng = torch.get_rng_state().clone()
    modes = [module.training for module in model.modules()]

    def broken_forward(*args, **kwargs):
        torch.rand(10)
        raise RuntimeError("probe failure")

    monkeypatch.setattr(model, "forward", broken_forward)
    with pytest.raises(RuntimeError, match="probe failure"):
        run_fixed_probe(model, probe)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in model.modules()] == modes
    assert not any(module._forward_hooks for module in model.modules())


def test_nonfinite_target_is_reported_with_json_safe_diagnostics():
    class NonfiniteField(torch.nn.Module):
        blocks = ()

        def forward(self, z, r, t, augment_labels=None):
            return z * float("inf")

    probe = build_fixed_probe(SyntheticCIFAR10(size=2), "cpu", count=2)
    result = run_fixed_probe(NonfiniteField(), probe)
    assert result["fp32_finite"] is False
    assert result["correction_abs_p99"] is None
    assert result["correction"]["nonfinite_count"] > 0
    json.dumps(result, allow_nan=False)
