import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from cifar10.config import TrainConfig
from cifar10.training_state import BatchStream, capture_rng, restore_rng, update_ema


def test_recipe_and_batch_validation():
    config = TrainConfig()
    config.validate(4)
    assert config.effective_batch_size(4) == 1024
    assert config.max_train_steps == 800000
    assert config.model == "DiT-B/4"
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(per_device_batch_size=0).validate(4)
    with pytest.raises(ValueError, match="smoke"):
        TrainConfig(model="DiT-Tiny/4").validate(1)


def test_batch_stream_resumes_across_epoch_boundary():
    dataset = TensorDataset(torch.arange(20))
    stream = BatchStream(dataset, 3, rank=0, world_size=2, seed=17, num_workers=0)
    for _ in range(5):
        stream.next()
    state = stream.state_dict()
    expected = [stream.next()[0] for _ in range(5)]
    restored = BatchStream(dataset, 3, rank=0, world_size=2, seed=17, num_workers=0)
    restored.load_state_dict(state)
    actual = [restored.next()[0] for _ in range(5)]
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)


def test_distributed_stream_has_disjoint_examples():
    dataset = TensorDataset(torch.arange(24))
    streams = [BatchStream(dataset, 3, rank=r, world_size=2, seed=2, num_workers=0)
               for r in range(2)]
    values = [torch.cat([s.next()[0] for _ in range(4)]).tolist() for s in streams]
    assert set(values[0]).isdisjoint(values[1])
    assert set(values[0] + values[1]) == set(range(24))


def test_rng_and_ema_restore():
    state = capture_rng(torch.device("cpu"))
    expected = torch.randn(4)
    restore_rng(state, torch.device("cpu"))
    torch.testing.assert_close(torch.randn(4), expected)
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model)
    old = ema.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    update_ema(ema, model, 0.75)
    torch.testing.assert_close(ema.weight, old + 0.5)


def test_stream_does_not_advance_training_rng():
    dataset = TensorDataset(torch.arange(20))
    state = torch.get_rng_state()
    stream = BatchStream(dataset, 2, rank=0, world_size=1, seed=0, num_workers=0)
    for _ in range(12):
        stream.next()
    assert torch.equal(torch.get_rng_state(), state)


def test_output_cannot_mix_runs_or_roll_back(tmp_path):
    from cifar10.train import validate_output
    output = tmp_path / "run"
    checkpoints = output / "checkpoints"
    first = checkpoints / "step-00000001"
    latest = checkpoints / "step-00000002"
    for path in (first, latest):
        path.mkdir(parents=True)
        (path / "complete").touch()
    (output / "config.json").write_text("{}")
    with pytest.raises(FileExistsError):
        validate_output(TrainConfig(output_dir=str(output)))
    with pytest.raises(ValueError, match="older"):
        validate_output(TrainConfig(output_dir=str(output), resume=str(first)))
    validate_output(TrainConfig(output_dir=str(output), resume="latest"))
    unrelated = tmp_path / "another-run"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="different run"):
        validate_output(TrainConfig(output_dir=str(unrelated), resume=str(latest)))
    assert (unrelated / "important.txt").read_text() == "keep"
    validate_output(TrainConfig(output_dir=str(tmp_path / "new-run"), resume=str(first)))


def test_truncated_log_recovery(tmp_path):
    from cifar10.train import recover_log
    path = tmp_path / "train.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"ste')
    recover_log(path, 2)
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1, 2]
    path.write_text('{"step": 1}\nnot-json\n{"step": 2}\n')
    with pytest.raises(ValueError, match="interior"):
        recover_log(path, 2)


def test_config_file_and_cli_precedence(tmp_path):
    from cifar10.config import parse_config
    path = tmp_path / "config.json"
    path.write_text('{"per_device_batch_size": 8, "gradient_accumulation_steps": 32}')
    config = parse_config(["--config", str(path), "--per-device-batch-size", "4", "--no-augmentation"])
    assert config.per_device_batch_size == 4
    assert config.gradient_accumulation_steps == 32
    assert not config.augmentation
    path.write_text('{"misspelled_key": 1}')
    with pytest.raises(SystemExit):
        parse_config(["--config", str(path)])
    path.write_text('{"smoke": "false"}')
    with pytest.raises(SystemExit):
        parse_config(["--config", str(path)])
    path.write_text('[]')
    with pytest.raises(SystemExit):
        parse_config(["--config", str(path)])


@pytest.mark.parametrize("seed", [-1, 2**32 - 1])
def test_seed_range_accounts_for_all_ranks(seed):
    with pytest.raises(ValueError, match="seed"):
        TrainConfig(seed=seed).validate(4)


def test_checked_in_recipe():
    from cifar10.config import parse_config
    recipe = Path(__file__).resolve().parents[1] / "configs/cifar10_dit_b4.json"
    config = parse_config(["--config", str(recipe)])
    config.validate(4)
    assert config == TrainConfig()


def test_ema_decay_rampup():
    from cifar10.training_state import ema_decay_at
    assert ema_decay_at(1, 0.99995, 0.05) < 1e-5
    values = [ema_decay_at(n, 0.99995, 0.05) for n in (1, 10, 100, 1000, 10000, 100000, 300000)]
    assert values == sorted(values)
    # Half-life equals 5% of the completed updates until the cap applies.
    assert ema_decay_at(1000, 0.99995, 0.05) ** 50 == pytest.approx(0.5)
    assert ema_decay_at(10000, 0.99995, 0.05) == pytest.approx(0.5 ** (1 / 500))
    assert ema_decay_at(300000, 0.99995, 0.05) == 0.99995
    assert ema_decay_at(5, 0.99995, 0) == 0.99995
    with pytest.raises(ValueError):
        ema_decay_at(0, 0.99995, 0.05)


@pytest.mark.parametrize("kwargs", [{"max_grad_norm": -1.0}, {"max_grad_norm": float("inf")},
                                    {"ema_rampup_ratio": 1.5}, {"divergence_loss": -1.0},
                                    {"diag_every": -1}])
def test_new_stability_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate(4)
    TrainConfig(max_grad_norm=0.0, ema_rampup_ratio=0.0, divergence_loss=0.0, diag_every=0).validate(4)


def test_four_gpu_recipe():
    from cifar10.config import parse_config
    recipe = Path(__file__).resolve().parents[1] / "configs/cifar10_4x2080.json"
    config = parse_config(["--config", str(recipe)])
    config.validate(4)
    assert config.effective_batch_size(4) == 1024 and config.model == "DiT-B/4"
    assert (config.per_device_batch_size, config.gradient_accumulation_steps) == (8, 32)
    assert (config.precision, config.optimizer, config.optimizer_kernel) == ("fp16", "adam", "reference")
    assert not (config.init_from or config.fork_from or config.resume)
    assert config.max_grad_norm == 1.0 and config.adam_beta2 == 0.95 and config.ema_rampup_ratio > 0
    assert config.divergence_loss == 7.0 and config.warmup_steps <= 0.25 * 20000


def test_loss_guard_trips_only_on_collapse():
    from cifar10.diagnostics import LossGuard
    guard = LossGuard(7.0, window=5, after_step=10)
    assert all(guard.update(step, 7.9) is None for step in range(1, 11))  # Initial loss is exempt.
    assert all(guard.update(step, 5.5) is None for step in range(11, 30))
    reasons = [guard.update(step, 9.0) for step in range(30, 36)]
    # Rolling-5 mean crosses 7.0 once three of five values are 9.0.
    assert reasons[:2] == [None, None] and "divergence_loss" in reasons[2]
    assert LossGuard(0.0, window=1, after_step=0).update(5, 100.0) is None


def test_log_from_a_failed_launch_does_not_block_a_new_run(tmp_path):
    from cifar10.train import validate_output
    output = tmp_path / "run"
    output.mkdir()
    (output / "train.log").write_text("CUDA is unavailable\n")
    validate_output(TrainConfig(output_dir=str(output)))
    (output / "config.json").write_text("{}")
    with pytest.raises(FileExistsError):
        validate_output(TrainConfig(output_dir=str(output)))


def test_lr_factor_warmup_constant_and_cooldown():
    from cifar10.training_state import lr_factor
    # Identical to the original warmup-then-constant schedule when no cooldown is set.
    for update in range(1, 30):
        assert lr_factor(update, 10) == min(update / 10, 1.0)
        assert lr_factor(update, 10, 20, 0) == min(update / 10, 1.0)
    assert lr_factor(20, 10, 20, 30) == 1.0          # last update before the cooldown
    assert lr_factor(21, 10, 20, 30) == pytest.approx(10 / 11)
    assert lr_factor(30, 10, 20, 30) == pytest.approx(1 / 11)   # final update still moves
    values = [lr_factor(u, 10, 20, 30) for u in range(20, 31)]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("kwargs", [{"cooldown_start": 100, "cooldown_end": 100},
                                    {"cooldown_start": 10, "cooldown_end": 20, "warmup_steps": 15},
                                    {"cooldown_start": 5, "cooldown_end": 0},
                                    {"cooldown_start": 10, "cooldown_end": 20, "max_train_steps": 30}])
def test_invalid_cooldown_is_rejected(kwargs):
    with pytest.raises(ValueError, match="ooldown|cooldown_end"):
        TrainConfig(**{"warmup_steps": 5, "max_train_steps": 20, **kwargs}).validate(4)
    TrainConfig(warmup_steps=5, max_train_steps=20, cooldown_start=10, cooldown_end=20).validate(4)


@pytest.mark.parametrize("kwargs", [{"precision": "fp8"}, {"jvp_fp16": True},
                                    {"precision": "fp16", "device": "cpu"}])
def test_invalid_precision_settings(kwargs):
    with pytest.raises(ValueError, match="precision|jvp_fp16"):
        TrainConfig(**kwargs).validate(4)
    TrainConfig(precision="fp16", jvp_fp16=True).validate(4)
    TrainConfig(precision="bf16", device="cpu").validate(4)
