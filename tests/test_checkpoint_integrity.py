"""Complete checkpoints preserve runtime/RNG state and selected rollback assets."""

from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest
import torch

from cifar10.checkpoint import (LEGACY_DEFAULTS, init_from_checkpoint, load_checkpoint,
                               protect_checkpoint, save_checkpoint)
from cifar10.config import TrainConfig
from cifar10.optim import create_optimizer
from cifar10.training_state import BatchStream, lr_factor
from torch.utils.data import TensorDataset


class CPUAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0
    is_main_process = True
    step = 0

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


def fixture(tmp_path, **overrides):
    config = TrainConfig(output_dir=str(tmp_path / "run"), device="cpu", num_workers=0,
                         warmup_steps=2, max_train_steps=10, **overrides)
    model = torch.nn.Linear(3, 2)
    optimizer = create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step + 1, config.warmup_steps))
    stream = BatchStream(TensorDataset(torch.arange(20)), 2, rank=0, world_size=1, seed=0, num_workers=0)
    for _ in range(3):
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        stream.next()
    return CPUAccelerator(), model, deepcopy(model), optimizer, scheduler, stream, config


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_checkpoint_restores_runtime_and_dedicated_rng(tmp_path):
    args = fixture(tmp_path, attn_projection_dropout=0.1)
    generator = torch.Generator().manual_seed(456)
    torch.rand(9, generator=generator)
    counters = dict(successful_updates=3, attempted_updates=5, skipped_attempts=2, consecutive_skips=1)
    saved = save_checkpoint(*args, 3, runtime_state=counters, dropout_generator=generator)
    expected = torch.rand(11, generator=generator)
    expected_batch = args[5].next()[0]
    restored = {}
    resumed_config = replace(args[-1], resume="latest")
    assert load_checkpoint(*args[:-1], resumed_config, runtime_state=restored,
                           dropout_generator=generator) == 3
    assert restored == counters
    torch.testing.assert_close(torch.rand(11, generator=generator), expected, rtol=0, atol=0)
    torch.testing.assert_close(args[5].next()[0], expected_batch, rtol=0, atol=0)
    assert load(saved / "training.pt")["optimizer_metadata"]["param_groups"] == [["weight", "bias"]]


def test_protected_checkpoint_survives_rolling_retention(tmp_path):
    args = fixture(tmp_path, checkpoints_to_keep=1)
    first = save_checkpoint(*args, 3)
    protect_checkpoint(first, "matched comparison winner")
    protection = (first / "protected.json").read_bytes()
    protect_checkpoint(first, "repeated call")
    assert (first / "protected.json").read_bytes() == protection
    second = save_checkpoint(*args, 4)
    third = save_checkpoint(*args, 5)
    assert first.is_dir() and third.is_dir() and not second.exists()
    assert all((first / name).is_file() for name in ("weights.pt", "training.pt", "rank-0.pt", "complete"))


@pytest.mark.parametrize("corruption", ["weights", "marker", "rank"])
def test_corrupt_complete_checkpoint_is_rejected_before_loading_model(tmp_path, corruption):
    args = fixture(tmp_path)
    saved = save_checkpoint(*args, 3)
    if corruption == "weights":
        weights = load(saved / "weights.pt")
        weights["step"] = 4
        torch.save(weights, saved / "weights.pt")
    elif corruption == "marker":
        (saved / "complete").write_text("4\n")
    else:
        (saved / "rank-0.pt").unlink()
    before = deepcopy(args[1].state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(*args[:-1], replace(args[-1], resume="latest"))
    with pytest.raises(ValueError):
        protect_checkpoint(saved)
    for key, tensor in before.items():
        torch.testing.assert_close(args[1].state_dict()[key], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("change", [
    {"time_sampler": "v1"}, {"attn_projection_dropout": 0.1},
    {"mlp_output_dropout": 0.1}, {"dropout_mask_mode": "replay"},
    {"optimizer": "adamw"}, {"weight_decay": 0.01}, {"weight_decay_policy": "all"},
    {"optimizer_kernel": "foreach"},
])
def test_resume_rejects_changed_scientific_setting(tmp_path, change):
    args = fixture(tmp_path)
    save_checkpoint(*args, 3)
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        load_checkpoint(*args[:-1], replace(args[-1], resume="latest", **change))


def test_legacy_defaults_and_missing_runtime_remain_loadable(tmp_path):
    args = fixture(tmp_path)
    saved = save_checkpoint(*args, 3)
    for name in ("training.pt", "weights.pt"):
        contents = load(saved / name)
        for key in LEGACY_DEFAULTS:
            contents["config"].pop(key, None)
        if name == "training.pt":
            del contents["runtime_state"]
            del contents["optimizer_metadata"]
            for group in contents["optimizer"]["param_groups"]:
                group.pop("param_names", None)
        torch.save(contents, saved / name)
    runtime = {}
    assert load_checkpoint(*args[:-1], replace(args[-1], resume="latest"), runtime_state=runtime) == 3
    assert runtime == dict(successful_updates=3, attempted_updates=3, skipped_attempts=0, consecutive_skips=0)


def test_warm_start_adamw_preserves_source_and_sets_two_group_schedule(tmp_path):
    args = fixture(tmp_path)
    saved = save_checkpoint(*args, 3)
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in saved.iterdir()}
    config = replace(args[-1], init_from=str(saved), output_dir=str(tmp_path / "warm"),
                     optimizer="adamw", weight_decay=0.01, learning_rate=0.005)
    model, ema = deepcopy(args[1]), deepcopy(args[2])
    optimizer = create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 10, 1))
    assert init_from_checkpoint(args[0], model, ema, optimizer, scheduler, config) == 3
    assert scheduler.last_epoch == 3 and scheduler.base_lrs == [0.005, 0.005]
    assert [group["lr"] for group in optimizer.param_groups] == [0.002, 0.002]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0]
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in saved.iterdir()} == original


def test_enabled_scaler_roundtrip_restores_scale_and_growth_tracker(tmp_path):
    # CPU GradScaler exercises the identical serializable state API without
    # claiming that CPU execution validates CUDA numerical behavior.
    args = fixture(tmp_path, precision="fp16")
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=10)
    state = scaler.state_dict()
    state.update(scale=32.0, _growth_tracker=7)
    scaler.load_state_dict(state)
    save_checkpoint(*args, 3, scaler=scaler)
    restored = torch.amp.GradScaler("cpu", init_scale=2048, growth_interval=5)
    load_checkpoint(*args[:-1], replace(args[-1], resume="latest"), scaler=restored)
    assert restored.state_dict() == state


def test_fp16_resume_rejects_missing_scaler_before_loading_model(tmp_path):
    args = fixture(tmp_path, precision="fp16")
    scaler = torch.amp.GradScaler("cpu")
    saved = save_checkpoint(*args, 3, scaler=scaler)
    training = load(saved / "training.pt")
    del training["scaler"]
    torch.save(training, saved / "training.pt")
    with torch.no_grad():
        args[1].weight.add_(5)
    before = args[1].weight.clone()
    with pytest.raises(ValueError, match="saved scaler"):
        load_checkpoint(*args[:-1], replace(args[-1], resume="latest"), scaler=scaler)
    torch.testing.assert_close(args[1].weight, before, rtol=0, atol=0)


def test_missing_dropout_rng_and_invalid_runtime_rejected_before_model_load(tmp_path):
    args = fixture(tmp_path, attn_projection_dropout=0.1)
    generator = torch.Generator().manual_seed(7)
    saved = save_checkpoint(*args, 3, dropout_generator=generator)
    rank = load(saved / "rank-0.pt")
    del rank["dropout_rng"]
    torch.save(rank, saved / "rank-0.pt")
    with pytest.raises(ValueError, match="dropout RNG"):
        load_checkpoint(*args[:-1], replace(args[-1], resume="latest"), dropout_generator=generator)
    training = load(saved / "training.pt")
    training["runtime_state"]["successful_updates"] = 99
    torch.save(training, saved / "training.pt")
    with pytest.raises(ValueError, match="successful_updates disagrees"):
        load_checkpoint(*args[:-1], replace(args[-1], resume="latest"), dropout_generator=generator)
