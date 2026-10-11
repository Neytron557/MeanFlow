"""CPU scientific-fork acceptance: complete states and actual next updates."""

from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from cifar10.checkpoint import fork_from_checkpoint, load_checkpoint, save_checkpoint
from cifar10.config import TrainConfig
from cifar10.dropout import IndexedDropout, indexed_dropout_context, make_dropout_generator
from cifar10.evaluate import file_sha256
from cifar10.objective import meanflow_loss, training_policy
from cifar10.optim import create_optimizer
from cifar10.training_state import (BatchStream, capture_rng, ema_decay_at, lr_factor,
                                    make_time_generator, update_ema)


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


class TinyVelocity(nn.Module):
    """A genuinely trained original-target field with subset-indexed dropout."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.randn(3) * .05)
        self.bias = nn.Parameter(torch.zeros(3))
        self.dropout = IndexedDropout(.1, (3, 4, 4))

    def forward(self, z, r, t, augment_labels=None):
        result = self.gain[None, :, None, None] * z + self.bias[None, :, None, None]
        result = result + .2 * t[:, None, None, None] + .1 * r[:, None, None, None]
        return self.dropout(result)


def dataset():
    images = (torch.arange(30 * 3 * 4 * 4).reshape(30, 3, 4, 4) % 256).float() / 127.5 - 1
    return TensorDataset(images, torch.arange(30))


def bundle(config, images=None):
    model = TinyVelocity()
    optimizer = create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda completed: lr_factor(completed + 1, config.warmup_steps))
    stream = BatchStream(dataset() if images is None else images, 2,
                         rank=0, world_size=1, seed=config.seed, num_workers=0)
    return SimpleNamespace(accelerator=CPUAccelerator(), model=model,
        ema=deepcopy(model).eval().requires_grad_(False), optimizer=optimizer,
        scheduler=scheduler, stream=stream, config=config,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        dropout=make_dropout_generator("cpu", config.seed), time=make_time_generator("cpu", config.time_rng_seed),
        runtime={"successful_updates": 0, "attempted_updates": 0,
                 "skipped_attempts": 0, "consecutive_skips": 0})


def config_for(tmp_path, **kwargs):
    return TrainConfig(output_dir=str(tmp_path), device="cpu", num_workers=0,
                       per_device_batch_size=2, gradient_accumulation_steps=1,
                       max_train_steps=10, warmup_steps=2, seed=43,
                       time_sampler="v1", dedicated_time_rng=True,
                       attn_projection_dropout=.1, **kwargs)


def update(state):
    age = state.runtime["successful_updates"]
    policy = training_policy(state.config, age)
    images = state.stream.next()[0]
    # Deliberately use every restored global RNG family in this update.
    images = images + (random.random() + float(np.random.rand())) * .01
    labels = torch.zeros((len(images), 6))
    kwargs = {key: value for key, value in policy.items() if key != "phase"}
    with indexed_dropout_context(state.model, batch_size=len(images), device="cpu", generator=state.dropout):
        loss, _ = meanflow_loss(state.model, state.model, images, labels,
                                time_sampler="v1", time_generator=state.time, **kwargs)
        state.scaler.scale(loss).backward()
        state.scaler.step(state.optimizer)
        state.scaler.update()
        state.optimizer.zero_grad(set_to_none=True)
    state.scheduler.step()
    update_ema(state.ema, state.model, ema_decay_at(age + 1, state.config.ema_decay,
                                                  state.config.ema_rampup_ratio))
    state.runtime["successful_updates"] += 1
    state.runtime["attempted_updates"] += 1
    return float(loss)


def save(state):
    return save_checkpoint(state.accelerator, state.model, state.ema, state.optimizer,
                           state.scheduler, state.stream, state.config,
                           state.runtime["successful_updates"], scaler=state.scaler,
                           runtime_state=state.runtime, dropout_generator=state.dropout,
                           time_generator=state.time)


def restore(state, *, fork=False):
    loader = fork_from_checkpoint if fork else load_checkpoint
    return loader(state.accelerator, state.model, state.ema, state.optimizer,
                  state.scheduler, state.stream, state.config, scaler=state.scaler,
                  runtime_state=state.runtime, dropout_generator=state.dropout,
                  time_generator=state.time)


def snapshot(state):
    return deepcopy({"model": state.model.state_dict(), "ema": state.ema.state_dict(),
        "optimizer": state.optimizer.state_dict(), "scheduler": state.scheduler.state_dict(),
        "scaler": state.scaler.state_dict(), "stream": state.stream.state_dict(),
        "dropout_rng": state.dropout.get_state(), "time_rng": state.time.get_state(),
        "rng": capture_rng(torch.device("cpu")), "runtime": state.runtime})


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def trained_source(tmp_path, *, updates=3, config_kwargs=None, images=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    source = bundle(config_for(tmp_path / "source", **(config_kwargs or {})), images)
    for _ in range(updates):
        update(source)
    return source, save(source)


def test_equivalent_explicit_fork_preserves_complete_state_and_exact_next_update(tmp_path):
    source, checkpoint = trained_source(tmp_path)
    original = snapshot(source)
    original_files = {path.name: file_sha256(path) for path in checkpoint.iterdir()}
    resumed = bundle(replace(source.config, output_dir=str(tmp_path / "resumed"), resume=str(checkpoint)))
    restore(resumed)
    assert_equal(snapshot(resumed), original)
    resumed_loss = update(resumed)
    expected = snapshot(resumed)
    forked = bundle(replace(source.config, output_dir=str(tmp_path / "forked"),
        fork_from=str(checkpoint), fork_reason="equivalent-policy scientific comparison"))
    assert restore(forked, fork=True) == 3
    assert_equal(snapshot(forked), original)
    assert forked.scaler.state_dict() == {} and not forked.scaler.is_enabled()
    assert update(forked) == resumed_loss
    assert_equal(snapshot(forked), expected)
    saved = save(forked)
    assert (saved / "complete").read_text().strip() == "4"
    assert {path.name: file_sha256(path) for path in checkpoint.iterdir()} == original_files


@pytest.mark.parametrize("age,allowed", [(2, False), (3, True)])
def test_curriculum_to_constant_requires_completed_weighting_and_same_next_update(tmp_path, age, allowed):
    source, checkpoint = trained_source(tmp_path, updates=age, config_kwargs={
        "train_policy": "fm_curriculum", "curriculum_fm_steps": 0,
        "curriculum_ramp_steps": 1, "curriculum_weight_end": 3})
    changed = replace(source.config, output_dir=str(tmp_path / "constant"), train_policy="constant",
                      fork_from=str(checkpoint), fork_reason="completed curriculum to paired constant policy")
    target = bundle(changed)
    if not allowed:
        before = deepcopy(target.model.state_dict())
        with pytest.raises(ValueError, match="train_policy|completed|equivalent"):
            restore(target, fork=True)
        assert_equal(target.model.state_dict(), before)
        return
    continued = bundle(replace(source.config, output_dir=str(tmp_path / "continued"), resume=str(checkpoint)))
    restore(continued)
    expected_loss = update(continued)
    expected = snapshot(continued)
    assert restore(target, fork=True) == age
    assert update(target) == expected_loss
    assert_equal(snapshot(target), expected)


@pytest.mark.parametrize("changes", [
    {"learning_rate": .0003}, {"adaptive_p": .5}, {"optimizer": "adamw"},
    {"ratio": .5}, {"time_sampler": "v0"}, {"norm_eps": .002},
])
def test_endpoint_focus_fork_cannot_silently_change_other_scientific_settings(tmp_path, changes):
    source, checkpoint = trained_source(tmp_path)
    config = replace(source.config, fork_from=str(checkpoint), fork_reason="endpoint coverage",
                     endpoint_focus_probability=.2, output_dir=str(tmp_path / "focus"), **changes)
    target = bundle(source.config)
    target.config = config
    before = deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        restore(target, fork=True)
    assert_equal(target.model.state_dict(), before)


def test_endpoint_focus_changes_only_the_declared_sampler_policy(tmp_path):
    source, checkpoint = trained_source(tmp_path)
    expected = snapshot(source)
    focused = bundle(replace(source.config, output_dir=str(tmp_path / "focus"),
        fork_from=str(checkpoint), fork_reason="paired positive-gap coverage", endpoint_focus_probability=.2))
    restore(focused, fork=True)
    assert_equal(snapshot(focused), expected)
    assert training_policy(focused.config, 3)["focus_probability"] == .2


def test_partition_change_requires_explicit_stream_reset(tmp_path):
    images = dataset()
    source, checkpoint = trained_source(tmp_path, images=Subset(images, list(range(18))),
        config_kwargs={"data_split": "train", "data_split_manifest": "immutable.json", "data_split_sha256": "0" * 64})
    config = replace(source.config, output_dir=str(tmp_path / "full"), data_split="full",
                     fork_from=str(checkpoint), fork_reason="declared 45k to 50k transition")
    target = bundle(config, images)
    before = deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="stream reset"):
        restore(target, fork=True)
    assert_equal(target.model.state_dict(), before)


def test_full_stage_resets_once_then_ordinary_resume_restores_its_new_stream(tmp_path):
    images = dataset()
    source, checkpoint = trained_source(tmp_path, images=Subset(images, list(range(18))),
        config_kwargs={"data_split": "train", "data_split_manifest": "immutable.json", "data_split_sha256": "0" * 64})
    full = bundle(replace(source.config, output_dir=str(tmp_path / "full"), data_split="full",
        fork_from=str(checkpoint), fork_reason="retire holdout and enter full data", fork_reset_data=True), images)
    fresh_stream = deepcopy(full.stream.state_dict())
    restore(full, fork=True)
    assert_equal(full.stream.state_dict(), fresh_stream)
    assert full.stream.offset == 0 and source.stream.offset == 3
    expected_source = snapshot(source)
    restored = snapshot(full)
    for key in expected_source.keys() - {"stream"}:
        assert_equal(restored[key], expected_source[key])
    update(full)
    full_checkpoint = save(full)
    before = snapshot(full)
    expected_loss = update(full)
    expected_after = snapshot(full)
    resumed = bundle(replace(full.config, output_dir=str(tmp_path / "resumed_full"), resume=str(full_checkpoint)), images)
    restore(resumed)
    assert_equal(snapshot(resumed), before)
    assert resumed.stream.offset == 1, "fork_reset_data must not reset an ordinary full-stage resume."
    assert update(resumed) == expected_loss
    assert_equal(snapshot(resumed), expected_after)


def test_scientific_fork_rejects_missing_time_rng_before_mutating_model(tmp_path):
    source, checkpoint = trained_source(tmp_path)
    rank_file = checkpoint / "rank-0.pt"
    rank = torch.load(rank_file, weights_only=True)
    del rank["time_rng"]
    torch.save(rank, rank_file)
    target = bundle(replace(source.config, output_dir=str(tmp_path / "fork"),
                           fork_from=str(checkpoint), fork_reason="equivalent fork"))
    before = deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="generator state"):
        restore(target, fork=True)
    assert_equal(target.model.state_dict(), before)
