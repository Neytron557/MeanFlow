"""Scientific contracts for the original-target replacement experiments."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from cifar10.config import TrainConfig
from cifar10.data import (load_cifar10_partition, make_cifar10_split_manifest,
                         read_cifar10_split_manifest, write_cifar10_split_manifest)
from cifar10.objective import meanflow_loss, sample_times, training_policy
from cifar10.training_state import capture_time_rng, make_time_generator, restore_time_rng


def curriculum_config(**kwargs):
    return TrainConfig(train_policy="fm_curriculum", time_sampler="v1", dedicated_time_rng=True, **kwargs)


@pytest.mark.parametrize("age,phase,ratio,progress", [
    (0, "fm_pretrain", 0, 0), (19999, "fm_pretrain", 0, 0),
    (20000, "mf_transition", 0, 1), (30000, "mf_transition", .375, 5 / 6),
    (39999, "mf_transition", .75 * 19999 / 20000, 1 - 19999 / 60000),
    (40000, "mf_full", .75, 2 / 3), (79999, "mf_full", .75, 1 / 60000),
    (80000, "mf_full", .75, 0), (120000, "mf_full", .75, 0),
])
def test_successful_age_curriculum_boundaries(age, phase, ratio, progress):
    policy = training_policy(curriculum_config(), age)
    assert policy["phase"] == phase
    assert policy["ratio"] == pytest.approx(ratio)
    assert policy["gap_weight_progress"] == pytest.approx(progress)
    assert policy["gap_weight_lambda"] == 1.61914605
    assert policy["focus_probability"] == 0
    assert policy["adaptive_p"] == (0 if age < 20000 else .75)
    assert policy["error_reduction"] == ("mean" if age < 20000 else "sum")
    assert training_policy(curriculum_config(), age) == policy, "Skipped attempts cannot advance a phase."


def test_direct_policy_is_constant_at_every_segment_boundary():
    config = TrainConfig(time_sampler="v1", dedicated_time_rng=True)
    assert training_policy(config, 0) == training_policy(config, 80000)
    focused = training_policy(TrainConfig(dedicated_time_rng=True, endpoint_focus_probability=.2), 0)
    assert focused["phase"] == "endpoint_focus" and focused["focus_probability"] == .2
    assert focused["gap_weight_progress"] == 0
    for age in (-1, 1.5, True):
        with pytest.raises(ValueError, match="successful_age"):
            training_policy(config, age)


def test_time_policies_consume_identical_dedicated_rng_draw_counts():
    global_state = torch.get_rng_state()
    states = []
    values = []
    for sampler, ratio, focus in (("v1", 0, 0), ("v1", .75, 0), ("v1", .75, .2), ("v0", 1, 0)):
        generator = make_time_generator("cpu")
        values.append(sample_times(101, "cpu", generator=generator, time_sampler=sampler,
                                   ratio=ratio, focus_probability=focus))
        states.append(capture_time_rng(generator))
    assert all(torch.equal(states[0], state) for state in states[1:])
    assert torch.equal(values[0][1], values[1][1]), "Ratio changes preserve the unmodified v1 t draws."
    assert torch.equal(torch.get_rng_state(), global_state)


def test_endpoint_mixture_preserves_all_equal_rows_and_replaces_only_positive_gaps():
    size = 100000
    ordinary = make_time_generator("cpu", seed=41)
    focused = make_time_generator("cpu", seed=41)
    r, t = sample_times(size, "cpu", time_sampler="v1", generator=ordinary)
    rf, tf = sample_times(size, "cpu", time_sampler="v1", generator=focused, focus_probability=.2)
    equal = r == t
    replaced = (r != rf) | (t != tf)
    assert not replaced[equal].any()
    assert ((rf[replaced] >= 0) & (rf[replaced] <= .2)).all()
    assert ((tf[replaced] >= .8) & (tf[replaced] <= 1)).all()
    assert (rf <= tf).all()
    assert replaced[~equal].float().mean().item() == pytest.approx(.2, abs=.005)
    assert (~equal).float().mean().item() == pytest.approx(.7001, abs=.005)
    assert torch.equal(ordinary.get_state(), focused.get_state())


def test_time_rng_resume_is_exact_across_a_policy_transition():
    generator = make_time_generator("cpu", rank=2)
    sample_times(17, "cpu", generator=generator, ratio=0, time_sampler="v1")
    state = capture_time_rng(generator)
    expected = sample_times(17, "cpu", generator=generator, ratio=.75, time_sampler="v1")
    restored = make_time_generator("cpu", seed=999, rank=2)
    restore_time_rng(restored, state)
    actual = sample_times(17, "cpu", generator=restored, ratio=.75, time_sampler="v1")
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    assert torch.equal(capture_time_rng(generator), capture_time_rng(restored))
    assert not torch.equal(make_time_generator("cpu", rank=0).get_state(),
                           make_time_generator("cpu", rank=1).get_state())
    with pytest.raises(ValueError, match="uint8"):
        restore_time_rng(restored, torch.zeros(3))


class AnalyticalField(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.calls = []

    def forward(self, z, r, t, augment_labels=None):
        self.calls.append(torch.is_grad_enabled())
        return self.scale * z + 2 * t[:, None, None, None] + 3 * r[:, None, None, None]


def test_progressive_gap_weights_follow_adaptive_weighting_and_preserve_conditional_jvp():
    model = AnalyticalField()
    images = torch.tensor([[[[.1, -.7]]], [[[.8, .2]]], [[[.2, .6]]]])
    noise = torch.tensor([[[[1.2, .4]]], [[[-.5, .6]]], [[[.3, -.1]]]])
    r, t = torch.tensor([.1, .6, 0.]), torch.tensor([.8, .6, 1.])
    z = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
    velocity = noise - images
    derivative = model.scale.detach() * velocity + 2  # Conditional tangent (v,0,1).
    target = velocity - (t - r)[:, None, None, None] * derivative
    prediction = model.scale.detach() * z + 2 * t[:, None, None, None] + 3 * r[:, None, None, None]
    error = prediction - target
    sums = error.square().flatten(1).sum(1)
    adaptive = (sums + .001).pow(-.75)
    beta = torch.tensor([1.61914605 * .3, 1., 0.])
    expected = (sums * adaptive * beta).mean()
    gradient = ((2 * error * z).flatten(1).sum(1) * adaptive * beta).mean()
    diagnostics = {}
    loss, mse = meanflow_loss(model, model, images, noise=noise, times=(r, t),
                             gap_weight_progress=1., diagnostics=diagnostics)
    loss.backward()
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(model.scale.grad, gradient)
    torch.testing.assert_close(mse, error.square().mean())
    torch.testing.assert_close(diagnostics["per_image_gap_weight"], beta)
    assert diagnostics["actual_mf_fraction"] == pytest.approx(2 / 3)
    assert model.calls == [False, True]


def test_diagonal_preparation_is_ordinary_mean_pixel_mse_and_skips_jvp():
    model = AnalyticalField()
    images, noise = torch.zeros(2, 3, 2, 2), torch.ones(2, 3, 2, 2)
    t = torch.tensor([.3, .7])
    prediction = model.scale.detach() * t[:, None, None, None] + 5 * t[:, None, None, None]
    expected = (prediction - 1).square().mean()
    loss, mse = meanflow_loss(model, model, images, noise=noise, times=(t, t),
                             error_reduction="mean", adaptive_p=0, gap_weight_progress=1.)
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(mse, expected)
    assert model.calls == [True]


def test_manifest_is_disjoint_complete_deterministic_and_does_not_touch_rng(tmp_path):
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    manifest = write_cifar10_split_manifest(tmp_path / "split.json")
    assert manifest == make_cifar10_split_manifest() == read_cifar10_split_manifest(tmp_path / "split.json")
    assert len(manifest["train_indices"]) == 45000 and len(manifest["holdout_indices"]) == 5000
    assert set(manifest["train_indices"]).isdisjoint(manifest["holdout_indices"])
    assert set(manifest["train_indices"] + manifest["holdout_indices"]) == set(range(50000))
    assert torch.equal(torch.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert np.array_equal(current_numpy[1], numpy_state[1]) and current_numpy[2:] == numpy_state[2:]
    before = (tmp_path / "split.json").read_bytes()
    assert write_cifar10_split_manifest(tmp_path / "split.json") == manifest
    with pytest.raises(ValueError, match="SHA256"):
        write_cifar10_split_manifest(tmp_path / "split.json", seed=3)
    assert (tmp_path / "split.json").read_bytes() == before


def test_manifest_hash_detects_tampering_and_allows_json_reformatting(tmp_path):
    path = tmp_path / "split.json"
    manifest = write_cifar10_split_manifest(path)
    path.write_text(json.dumps(manifest))
    assert read_cifar10_split_manifest(path, manifest["sha256"]) == manifest
    with pytest.raises(ValueError, match="configured SHA256"):
        read_cifar10_split_manifest(path, "0" * 64)
    corrupted = copy.deepcopy(manifest)
    corrupted["seed"] += 1
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="SHA256"):
        read_cifar10_split_manifest(path)
    payload = {key: value for key, value in corrupted.items() if key != "sha256"}
    corrupted["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="seed and algorithm"):
        read_cifar10_split_manifest(path)


def test_partition_loader_maps_exact_original_indices_and_full_stage_lineage(tmp_path):
    class IndexDataset(Dataset):
        def __len__(self):
            return 50000

        def __getitem__(self, index):
            return index

    dataset = IndexDataset()
    path = tmp_path / "split.json"
    manifest = write_cifar10_split_manifest(path)
    train = load_cifar10_partition(dataset, path, partition="train", expected_sha256=manifest["sha256"])
    holdout = load_cifar10_partition(dataset, manifest, partition="holdout")
    assert len(train) == 45000 and len(holdout) == 5000
    assert train[0] == manifest["train_indices"][0] and holdout[-1] == manifest["holdout_indices"][-1]
    assert load_cifar10_partition(dataset, path, partition="full") is dataset
    with pytest.raises(ValueError, match="configured SHA256"):
        load_cifar10_partition(dataset, path, partition="full", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="50,000"):
        load_cifar10_partition(torch.arange(10), manifest)


@pytest.mark.parametrize("kwargs", [
    {"train_policy": "fm_curriculum"}, {"data_split": "holdout"}, {"data_split": "train"},
    {"data_split_sha256": "x" * 64}, {"time_rng_seed": True},
    {"curriculum_ramp_steps": 0}, {"curriculum_fm_steps": 2.5}, {"curriculum_gap_lambda": float("nan")},
    {"curriculum_weight_end": 40000}, {"endpoint_focus_probability": .2},
    {"fork_from": "checkpoint"}, {"fork_reset_data": True},
    {"fork_from": "checkpoint", "fork_reason": "transition", "init_from": "old"},
    {"optimizer": "muon", "weight_decay": .01}, {"muon_ns_steps": True},
    {"muon_momentum": 1.}, {"muon_ns_dtype": "bf16"}, {"muon_rms_scale": 0.},
])
def test_research_config_rejects_ambiguous_or_invalid_recipes(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate(1)


def test_research_config_allows_explicit_forks_and_approved_pilot_recipes():
    curriculum_config(data_split="train", data_split_manifest="split.json", data_split_sha256="0" * 64).validate(4)
    TrainConfig(optimizer="muon", dedicated_time_rng=True, time_sampler="v1").validate(4)
    TrainConfig(fork_from="checkpoint", fork_reason="45k to full-data transition", fork_reset_data=True,
                data_split_manifest="split.json").validate(4)
    TrainConfig(endpoint_focus_probability=.2, dedicated_time_rng=True).validate(4)
