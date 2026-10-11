"""The explicit AMP outcome gates all parameter and optimizer mutations."""

from copy import deepcopy
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from accelerate import Accelerator

from cifar10.amp_update import optimizer_attempt


class LocalCollective:
    num_processes = 1

    def reduce(self, value, reduction):
        return value


@pytest.mark.parametrize("failure", ["loss", "gradient", "norm"])
def test_skip_preserves_adamw_and_resets_scaler_growth(failure):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=2)
    collective = LocalCollective()

    def backward():
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(2, 2)).square().mean()).backward()

    backward()
    assert optimizer_attempt(model, optimizer, scaler, collective,
                             nonfinite_loss=False, max_norm=1)[0]
    assert scaler.state_dict()["_growth_tracker"] == 1
    before = deepcopy(model.state_dict())
    moments = deepcopy(optimizer.state_dict())
    backward()
    if failure == "gradient":
        next(model.parameters()).grad.fill_(float("inf"))
    elif failure == "norm":
        # Entries remain finite after unscale; the FP32 L2 norm overflows.
        for parameter in model.parameters():
            parameter.grad.fill_(1e30)
    committed, _ = optimizer_attempt(model, optimizer, scaler, collective,
                                    nonfinite_loss=failure == "loss", max_norm=1)
    assert not committed
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    for index, state in moments["state"].items():
        for name, tensor in state.items():
            assert torch.equal(tensor, optimizer.state_dict()["state"][index][name])
    assert scaler.get_scale() == 4
    assert scaler.state_dict()["_growth_tracker"] == 0
    backward()
    assert optimizer_attempt(model, optimizer, scaler, collective,
                             nonfinite_loss=False, max_norm=1)[0]
    assert scaler.get_scale() == 4
    backward()
    assert optimizer_attempt(model, optimizer, scaler, collective,
                             nonfinite_loss=False, max_norm=1)[0]
    assert scaler.get_scale() == 8


def _distributed_worker():
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    original = deepcopy(accelerator.unwrap_model(model).state_dict())
    for attempt in range(2):
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(scaler.scale(model(torch.ones(2, 2)).square().mean()))
        if attempt == 0 and accelerator.process_index == 1:
            next(model.parameters()).grad.fill_(float("inf"))
        committed, _ = optimizer_attempt(model, optimizer, scaler, accelerator,
                                        nonfinite_loss=False, max_norm=1)
        assert committed == (attempt == 1)
        if not committed:
            assert not optimizer.state
            for name, tensor in accelerator.unwrap_model(model).state_dict().items():
                assert torch.equal(tensor, original[name])
        states = accelerator.gather(torch.tensor([scaler.get_scale(),
                                                  scaler.state_dict()["_growth_tracker"]]))
        assert torch.equal(states[:2], states[2:])
    parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = accelerator.gather(parameters)
    assert torch.equal(gathered[:parameters.numel()], gathered[parameters.numel():])
    accelerator.end_training()


@pytest.mark.slow
def test_rank_local_overflow_does_not_partially_step():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = "import runpy; runpy.run_path('tests/test_amp_update.py')['_distributed_worker']()"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
               "--master-addr=127.0.0.1", f"--master-port={port}", "--no-python",
               sys.executable, "-c", code]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=90, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
