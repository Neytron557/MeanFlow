"""CPU transaction tests; GPU numerical/throughput acceptance is a separate gate."""

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from accelerate import Accelerator
import pytest
import torch

from cifar10.amp_update import optimizer_attempt
from cifar10.config import TrainConfig
from cifar10.model import create_model
import cifar10.muon as muon_module
from cifar10.muon import HybridMuon, newton_schulz
from cifar10.optim import create_optimizer, load_optimizer_state, optimizer_metadata


class LocalCollective:
    num_processes = 1

    def reduce(self, value, reduction):
        return value


class SmallNamedModel(torch.nn.Module):
    """The canonical parameter names with small matrices and no unused gradients."""

    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.ones(1, 4), requires_grad=False)
        block = torch.nn.Module()
        block.attn = torch.nn.ModuleDict(dict(qkv=torch.nn.Linear(4, 12), proj=torch.nn.Linear(4, 4)))
        block.mlp = torch.nn.ModuleDict(dict(fc1=torch.nn.Linear(4, 8), fc2=torch.nn.Linear(8, 4)))
        block.adaLN_modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(4, 24))
        self.blocks = torch.nn.ModuleList([block])
        self.x_embedder = torch.nn.Linear(4, 4)
        self.final_layer = torch.nn.Linear(4, 2)
        self.t_embedder = torch.nn.Linear(4, 4)
        self.r_embedder = torch.nn.Linear(4, 4)
        self.augment_embedder = torch.nn.Linear(6, 4, bias=False)

    def forward(self, values):
        return values.mean() * sum(parameter.square().mean()
                                   for parameter in self.parameters() if parameter.requires_grad)


def configuration(**overrides):
    values = asdict(TrainConfig())
    values.update(optimizer="muon", learning_rate=2e-4, weight_decay=0,
                  muon_momentum=.95, muon_ns_steps=5, muon_ns_dtype="fp32", muon_rms_scale=.2)
    values.update(overrides)
    return SimpleNamespace(**values)


def assign_gradients(model, seed=17, scale=1):
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn(parameter.shape, generator=generator) * scale


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            nested_equal(a, b)
    else:
        assert left == right


def test_real_dit_partition_excludes_conditioning_and_frozen_positions():
    model = create_model("DiT-Tiny/4")
    before = {name: parameter._version for name, parameter in model.named_parameters()}
    optimizer = create_optimizer(model, configuration())
    groups = {group["kind"]: group for group in optimizer.param_groups}
    assert len(groups["muon"]["params"]) == 8
    assert all(name.endswith(("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"))
               for name in groups["muon"]["param_names"])
    assert any("adaLN_modulation" in name for name in groups["adam"]["param_names"])
    names = sum([group["param_names"] for group in optimizer.param_groups], [])
    assert set(names) == {name for name, p in model.named_parameters() if p.requires_grad}
    assert len(names) == len(set(names)) and "pos_embed" not in names
    assert before == {name: parameter._version for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("shape", [(4, 4), (12, 4), (4, 12)])
def test_newton_schulz_zero_scale_and_transpose(shape):
    generator = torch.Generator().manual_seed(4)
    gradient = torch.randn(shape, generator=generator)
    zeros = newton_schulz(torch.zeros(shape), dtype=torch.float32)
    assert torch.equal(zeros, torch.zeros(shape))
    reference = newton_schulz(gradient, dtype=torch.float32)
    torch.testing.assert_close(reference, newton_schulz(gradient.mT, dtype=torch.float32).mT,
                               rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(reference, newton_schulz(gradient * 1000, dtype=torch.float32),
                               rtol=1e-5, atol=2e-6)
    assert torch.isfinite(reference).all()


def test_staging_preserves_weights_gradients_and_state_then_matches_auxiliary_adam():
    torch.manual_seed(1)
    model = SmallNamedModel()
    optimizer = create_optimizer(model, configuration())
    reference = deepcopy(model)
    auxiliary_names = optimizer.param_groups[1]["param_names"]
    named_reference = dict(reference.named_parameters())
    adam = torch.optim.Adam([named_reference[name] for name in auxiliary_names], lr=2e-4,
                            betas=(.9, .95), eps=1e-8, foreach=False, fused=False)
    for seed in (17, 18, 19):
        assign_gradients(model, seed)
        assign_gradients(reference, seed)
        weights = deepcopy(model.state_dict())
        states = deepcopy(optimizer.state_dict())
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        optimizer.stage_updates()
        assert optimizer.reconcile_staged(LocalCollective())
        nested_equal(weights, model.state_dict())
        nested_equal(states, optimizer.state_dict())
        for name, gradient in gradients.items():
            assert torch.equal(gradient, dict(model.named_parameters())[name].grad)
        optimizer.step()
        adam.step()
        for name in auxiliary_names:
            torch.testing.assert_close(dict(model.named_parameters())[name], named_reference[name], rtol=0, atol=0)
        assert optimizer.last_step_diagnostics["committed"]


@pytest.mark.parametrize("failure", ["loss", "gradient", "ns", "adam_candidate"])
def test_invalid_transaction_preserves_all_states_and_resets_scaler(failure, monkeypatch):
    model = SmallNamedModel()
    optimizer = create_optimizer(model, configuration())
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=2)
    scaler.scale(torch.ones((), requires_grad=True))
    assign_gradients(model, scale=scaler.get_scale())
    assert optimizer_attempt(model, optimizer, scaler, LocalCollective(), nonfinite_loss=False, max_norm=0)[0]
    before_model, before_state = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    assign_gradients(model, seed=18, scale=scaler.get_scale())
    if failure == "gradient":
        next(p for p in model.parameters() if p.requires_grad).grad.fill_(float("inf"))
    elif failure == "ns":
        monkeypatch.setattr(muon_module, "newton_schulz", lambda value, **kwargs: torch.full_like(value, float("nan")))
    elif failure == "adam_candidate":
        optimizer.param_groups[1]["lr"] = float("inf")
        before_state = deepcopy(optimizer.state_dict())
    committed, _ = optimizer_attempt(model, optimizer, scaler, LocalCollective(),
                                    nonfinite_loss=failure == "loss", max_norm=0)
    assert not committed
    nested_equal(before_model, model.state_dict())
    nested_equal(before_state, optimizer.state_dict())
    assert scaler.get_scale() == 4 and scaler.state_dict()["_growth_tracker"] == 0
    assert optimizer._staged is None


def test_half_nonfinite_recomputes_all_failed_matrices_in_fp32(monkeypatch):
    model = SmallNamedModel()
    optimizer = create_optimizer(model, configuration(muon_ns_dtype="fp16"))
    assign_gradients(model)
    reference = muon_module.newton_schulz
    monkeypatch.setattr(muon_module, "newton_schulz", lambda value, **kwargs:
                        torch.full_like(value, float("nan")) if kwargs["dtype"] == torch.float16
                        else reference(value, **kwargs))
    optimizer.stage_updates()
    assert optimizer.reconcile_staged(LocalCollective())
    assert optimizer.last_step_diagnostics["ns_fp32_fallback"] == 4
    assert not optimizer.state
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_commit_rejects_post_stage_changes_before_any_parameter_mutation():
    model = SmallNamedModel()
    optimizer = create_optimizer(model, configuration())
    assign_gradients(model)
    optimizer.stage_updates()
    assert optimizer.reconcile_staged()
    optimizer.param_groups[1]["lr"] *= 2
    before = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="changed after staging"):
        optimizer.commit_staged()
    nested_equal(before, model.state_dict())
    assert not optimizer.state
    optimizer.discard_staged()


def test_named_hybrid_resume_reproduces_the_next_update_exactly():
    model = SmallNamedModel()
    config = configuration()
    optimizer = create_optimizer(model, config)
    for seed in (17, 18):
        assign_gradients(model, seed)
        optimizer.step()
    saved, metadata = deepcopy(optimizer.state_dict()), optimizer_metadata(optimizer, model)
    assert metadata["version"] == 2 and metadata["kind"] == "hybrid_muon"
    target = deepcopy(model)
    restored = create_optimizer(target, config)
    load_optimizer_state(restored, saved, target, config, metadata)
    saved_original = deepcopy(saved)
    assign_gradients(model, 19)
    assign_gradients(target, 19)
    optimizer.step()
    restored.step()
    nested_equal(model.state_dict(), target.state_dict())
    nested_equal(optimizer.state_dict(), restored.state_dict())
    nested_equal(saved, saved_original)


@pytest.mark.parametrize("corruption", ["momentum_shape", "momentum_dtype", "nonfinite", "step", "missing",
                                          "roles", "recipe", "same_run_recipe"])
def test_hybrid_checkpoint_rejection_is_nonmutating(corruption):
    model = SmallNamedModel()
    config = configuration()
    optimizer = create_optimizer(model, config)
    assign_gradients(model)
    optimizer.step()
    saved, metadata = deepcopy(optimizer.state_dict()), optimizer_metadata(optimizer, model)
    state = saved["state"][saved["param_groups"][0]["params"][0]]
    if corruption == "momentum_shape":
        state["momentum_buffer"] = torch.zeros(123)
    elif corruption == "momentum_dtype":
        state["momentum_buffer"] = state["momentum_buffer"].double()
    elif corruption == "nonfinite":
        state["momentum_buffer"].fill_(float("inf"))
    elif corruption == "step":
        state["step"] = torch.tensor(1.5)
    elif corruption == "missing":
        del state["momentum_buffer"]
    elif corruption == "roles":
        metadata["group_kinds"] = ["adam", "muon"]
    elif corruption == "recipe":
        metadata["muon_recipe"]["scaling"] = "original"
    else:
        config = configuration(muon_momentum=.9)
    target = create_optimizer(model, config)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError):
        load_optimizer_state(target, saved, model, config, metadata)
    nested_equal(before, target.state_dict())


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("to_muon", [False, True])
def test_adam_muon_cross_migration_is_rejected(to_muon, warm_start):
    model = SmallNamedModel()
    adam_config, muon_config = configuration(optimizer="adam"), configuration()
    source_config, target_config = (adam_config, muon_config) if to_muon else (muon_config, adam_config)
    source, target = create_optimizer(model, source_config), create_optimizer(model, target_config)
    assign_gradients(model)
    source.step()
    with pytest.raises(ValueError, match="cannot be migrated"):
        load_optimizer_state(target, source.state_dict(), model, target_config,
                             optimizer_metadata(source, model), warm_start=warm_start)
    assert not target.state


def _distributed_worker():
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    torch.manual_seed(7)
    model = SmallNamedModel()
    optimizer = create_optimizer(model, configuration(muon_ns_dtype="fp16"))
    model, optimizer = accelerator.prepare(model, optimizer)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=2)
    original_ns = muon_module.newton_schulz
    for attempt in range(3):
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(scaler.scale(model(torch.ones(2, 2))))
        before_model = deepcopy(accelerator.unwrap_model(model).state_dict())
        before_state = deepcopy(optimizer.state_dict())

        def faulty_ns(value, **kwargs):
            if accelerator.process_index == 1 and (attempt == 0 or
                                                    (attempt == 1 and kwargs["dtype"] == torch.float16)):
                return torch.full_like(value, float("nan"))
            return original_ns(value, **{**kwargs, "dtype": torch.float32})

        muon_module.newton_schulz = faulty_ns
        committed, _ = optimizer_attempt(model, optimizer, scaler, accelerator,
                                        nonfinite_loss=False, max_norm=1)
        assert committed == (attempt != 0)
        if not committed:
            nested_equal(before_model, accelerator.unwrap_model(model).state_dict())
            nested_equal(before_state, optimizer.state_dict())
        if attempt == 1:
            assert optimizer.optimizer.last_step_diagnostics["ns_fp32_fallback"] == 4
        values = torch.tensor([scaler.get_scale(), scaler.state_dict()["_growth_tracker"]])
        gathered = accelerator.gather(values)
        assert torch.equal(gathered[:2], gathered[2:])
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = accelerator.gather(parameters)
        assert torch.equal(gathered[:parameters.numel()], gathered[parameters.numel():])
    accelerator.end_training()


@pytest.mark.slow
def test_rank_local_ns_failure_cannot_partially_commit_or_diverge_scalers():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = "import runpy; runpy.run_path('tests/test_research_muon.py')['_distributed_worker']()"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
               "--master-addr=127.0.0.1", f"--master-port={port}", "--no-python",
               sys.executable, "-c", code]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=90, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
