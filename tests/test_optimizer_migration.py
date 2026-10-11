"""CPU fixtures: prove migration preserves the actual Adam update, not just keys."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from cifar10.config import TrainConfig
from cifar10.optim import create_optimizer, load_optimizer_state, optimizer_metadata


class FrozenPositionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3), requires_grad=False)
        self.proj = torch.nn.Linear(3, 2)
        self.gain = torch.nn.Parameter(torch.ones(()))


def legacy_source():
    torch.manual_seed(42)
    model = FrozenPositionModel()
    config = TrainConfig(warmup_steps=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2),
                                 eps=config.adam_epsilon, foreach=False, fused=False)
    for update in range(3):
        assign_grads(model, update)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return model, optimizer, config


def assign_grads(model, update=0):
    for index, parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) / 10
            parameter.grad.add_((index + 1) * (update + 1) / 7)


def test_legacy_zero_decay_adamw_matches_adam_continuation_exactly():
    model, legacy, config = legacy_source()
    target = deepcopy(model)
    adamw_config = replace(config, optimizer="adamw")
    adamw = create_optimizer(target, adamw_config)
    saved = deepcopy(legacy.state_dict())
    load_optimizer_state(adamw, saved, target, adamw_config, warm_start=True)
    assert len(saved["param_groups"]) == 1 and "param_names" not in saved["param_groups"][0]
    assert len(adamw.param_groups) == 2
    assert all(target.pos_embed is not p for group in adamw.param_groups for p in group["params"])
    for name, parameter in target.named_parameters():
        if parameter.requires_grad:
            original = dict(model.named_parameters())[name]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(adamw.state[parameter][key], legacy.state[original][key], rtol=0, atol=0)
    for update in range(3, 6):
        assign_grads(model, update)
        assign_grads(target, update)
        legacy.step()
        adamw.step()
        for left, right in zip(model.parameters(), target.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_migration_reapplies_hyperparameters_and_decays_only_matrices():
    source, legacy, config = legacy_source()
    zero, decayed = deepcopy(source), deepcopy(source)
    zero_config = replace(config, optimizer="adamw", learning_rate=0.003, adam_beta2=0.97,
                          adam_epsilon=2e-8)
    decay_config = replace(zero_config, weight_decay=0.01)
    zero_optimizer, decayed_optimizer = create_optimizer(zero, zero_config), create_optimizer(decayed, decay_config)
    for optimizer, model, cfg in ((zero_optimizer, zero, zero_config),
                                  (decayed_optimizer, decayed, decay_config)):
        load_optimizer_state(optimizer, deepcopy(legacy.state_dict()), model, cfg, warm_start=True)
        for group in optimizer.param_groups:
            assert group["lr"] == cfg.learning_rate
            assert group["initial_lr"] == cfg.learning_rate
            assert group["betas"] == (cfg.adam_beta1, cfg.adam_beta2)
            assert group["eps"] == cfg.adam_epsilon
            assert group["weight_decay"] == (cfg.weight_decay if group["params"][0].ndim >= 2 else 0)
        assign_grads(model, 5)
        optimizer.step()
    for name, parameter in decayed.named_parameters():
        difference = dict(zero.named_parameters())[name] - parameter
        expected = (dict(source.named_parameters())[name] * decay_config.learning_rate * 0.01
                    if parameter.requires_grad and parameter.ndim >= 2 else torch.zeros_like(parameter))
        # Subtracting nearly equal FP32 weights loses up to one ULP here.
        torch.testing.assert_close(difference, expected, rtol=1e-3, atol=1e-7)


def test_named_adam_groups_preserve_legacy_including_frozen_parameter():
    model, _, config = legacy_source()
    optimizer = create_optimizer(model, config)
    names = [name for name, _ in model.named_parameters()]
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["param_names"] == names
    assert "pos_embed" in names
    assert optimizer_metadata(optimizer, model) == {"version": 1, "param_groups": [names]}


@pytest.mark.parametrize("corruption", ["shape", "dtype", "step", "nonfinite", "missing", "unknown_id"])
def test_migration_rejects_invalid_moments(corruption):
    model, legacy, config = legacy_source()
    saved = deepcopy(legacy.state_dict())
    state = next(iter(saved["state"].values()))
    if corruption == "shape":
        state["exp_avg"] = torch.zeros(123)
    elif corruption == "dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif corruption == "step":
        state["step"] = torch.tensor(1.5)
    elif corruption == "nonfinite":
        state["exp_avg"].fill_(float("nan"))
    elif corruption == "missing":
        del state["exp_avg_sq"]
    else:
        saved["state"][999] = state
    target = create_optimizer(model, replace(config, optimizer="adamw"))
    with pytest.raises(ValueError):
        load_optimizer_state(target, saved, model, config, warm_start=True)
    assert not target.state


def test_named_mapping_rejects_duplicate_or_disagreeing_names():
    model, legacy, config = legacy_source()
    target = create_optimizer(model, config)
    names = optimizer_metadata(target, model)
    names["param_groups"][0][1] = names["param_groups"][0][0]
    with pytest.raises(ValueError, match="duplicate"):
        load_optimizer_state(target, legacy.state_dict(), model, config, names, warm_start=True)


@pytest.mark.parametrize("kwargs", [
    {"time_sampler": "v2"}, {"attn_projection_dropout": 1.0},
    {"mlp_output_dropout": -0.1}, {"attn_projection_dropout": float("nan")},
    {"mlp_output_dropout": float("inf")}, {"dropout_mask_mode": "replay"},
    {"optimizer": "sgd"}, {"weight_decay": 0.01}, {"weight_decay": float("nan")},
    {"weight_decay": float("inf")}, {"weight_decay": -0.1},
    {"weight_decay_policy": "all"},
])
def test_new_scientific_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate(4)


def test_adamw_candidate_configuration_is_valid():
    TrainConfig(optimizer="adamw", weight_decay=0.01, time_sampler="v1",
                attn_projection_dropout=0.1, mlp_output_dropout=0.1).validate(4)
