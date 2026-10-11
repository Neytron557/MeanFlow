"""Exercise the public training command, including a real restart."""

import json
import os
from pathlib import Path
import subprocess
import socket
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def run_train(output, steps, *extra, processes=1, check=True):
    launch = [sys.executable]
    if processes > 1:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        launch += ["-m", "torch.distributed.run", "--nnodes=1", "--master-addr=127.0.0.1",
                   f"--master-port={port}", f"--nproc_per_node={processes}"]
    command = launch + ["train.py", "--dataset", "cifar10", "--smoke", "--model", "DiT-Tiny/4",
                       "--device", "cpu", "--per-device-batch-size", "2",
                       "--gradient-accumulation-steps", "2", "--max-train-steps", str(steps),
                       "--warmup-steps", "2", "--num-workers", "0", "--checkpointing-steps", "1",
                       "--output-dir", str(output), *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    if not check:
        return result
    assert result.returncode == 0, result.stdout + result.stderr
    return output / "checkpoints" / f"step-{steps:08d}"


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_uninterrupted_equals_resume_with_augmentation(tmp_path):
    full = run_train(tmp_path / "full", 3)
    run_train(tmp_path / "resumed", 2)
    resumed = run_train(tmp_path / "resumed", 3, "--resume", "latest")
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    a, b = load(full / "weights.pt"), load(resumed / "weights.pt")
    assert_nested_equal(a["model"], b["model"])
    assert_nested_equal(a["ema"], b["ema"])
    a, b = load(full / "training.pt"), load(resumed / "training.pt")
    assert_nested_equal(a["optimizer"], b["optimizer"])
    assert_nested_equal(a["scheduler"], b["scheduler"])
    # Iterator creation may advance only the dedicated loader generator.
    a, b = load(full / "rank-0.pt"), load(resumed / "rank-0.pt")
    assert_nested_equal(a["rng"], b["rng"])
    assert a["stream"]["epoch"] == b["stream"]["epoch"]
    assert a["stream"]["offset"] == b["stream"]["offset"]


@pytest.mark.slow
def test_two_process_training_and_resume(tmp_path):
    checkpoint = run_train(tmp_path / "distributed", 2, processes=2)
    assert (checkpoint / "rank-0.pt").is_file()
    assert (checkpoint / "rank-1.pt").is_file()
    checkpoint = run_train(tmp_path / "distributed", 3, "--resume", "latest", processes=2)
    assert (checkpoint / "complete").is_file()
    logs = [json.loads(line) for line in (tmp_path / "distributed" / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logs] == [1, 2, 3]
    assert len(list(checkpoint.parent.glob("step-*"))) == 3


def test_logs_diagnostics_loss_plots_and_train_log(tmp_path):
    from cifar10.training_state import ema_decay_at

    output = tmp_path / "run"
    run_train(output, 2, "--diag-every", "1", "--plot-every", "1", "--log-every", "1")
    run_train(output, 4, "--diag-every", "1", "--plot-every", "1", "--log-every", "1", "--resume", "latest")
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    for row in rows:
        assert row["grad_norm"] > 0 and row["ema_decay"] == ema_decay_at(row["step"], 0.99995, 0.05)
        assert len(row["attn_logit_max_by_block"]) == 2 and row["attn_logit_max"] >= 0
        # The total loss is the fm_fraction-weighted mean of loss 1 and loss 2.
        parts = [(row["fm_fraction"], row["loss_fm"]), (1 - row["fm_fraction"], row["loss_mf"])]
        assert abs(sum(f * v for f, v in parts if v is not None) - row["loss"]) < 1e-5 * row["loss"]
        assert (row["loss_fm"] is None) == (row["fm_fraction"] == 0) == (row["mse_fm"] is None)
    # A curve is drawn once its loss has any value (tiny batches may lack one sample type).
    expected = {"total_loss.png"}
    expected |= {"loss1_flow_matching.png"} if any(row["loss_fm"] is not None for row in rows) else set()
    expected |= {"loss2_meanflow.png"} if any(row["loss_mf"] is not None for row in rows) else set()
    assert {path.name for path in (output / "plots").glob("*.png")} == expected
    log = (output / "train.log").read_text()
    assert "Starting a new run" in log and "Resuming from update 2" in log
    assert "Resolved configuration" in log and "update 4/4 | total loss" in log
    assert "Saved checkpoint step-00000004" in log and "Training stopped at update 4" in log


def test_divergence_guard_stops_and_marks_the_run(tmp_path):
    output = tmp_path / "diverged"
    command = [sys.executable, "train.py", "--dataset", "cifar10", "--smoke", "--model", "DiT-Tiny/4",
               "--device", "cpu", "--per-device-batch-size", "2", "--gradient-accumulation-steps", "1",
               "--max-train-steps", "80", "--warmup-steps", "2", "--num-workers", "0",
               "--checkpointing-steps", "20", "--divergence-loss", "0.001", "--diag-every", "0",
               "--output-dir", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    assert result.returncode == 3, result.stdout + result.stderr
    marker = json.loads((output / "divergence.json").read_text())
    assert marker["step"] == 52 and "divergence_loss" in marker["reason"]
    assert marker["last_checkpoint_step"] == 40
    assert sorted(p.name for p in (output / "checkpoints").glob("step-*")) == ["step-00000020", "step-00000040"]
    assert "Training diverged" in (output / "train.log").read_text()
    # The tripping update is logged, so the plots show the collapse.
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert rows[-1]["step"] == 52


def test_cooldown_added_on_resume_matches_uninterrupted(tmp_path):
    cooldown = ("--cooldown-start", "4", "--cooldown-end", "6")
    full = run_train(tmp_path / "full", 6, *cooldown)
    first = run_train(tmp_path / "resumed", 3)
    # Simulate a checkpoint written before the cooldown settings existed.
    training = torch.load(first / "training.pt", map_location="cpu", weights_only=True)
    for key in ("cooldown_start", "cooldown_end"):
        del training["config"][key]
    torch.save(training, first / "training.pt")
    resumed = run_train(tmp_path / "resumed", 6, "--resume", "latest", *cooldown)
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    assert_nested_equal(load(full / "weights.pt")["model"], load(resumed / "weights.pt")["model"])
    assert_nested_equal(load(full / "weights.pt")["ema"], load(resumed / "weights.pt")["ema"])
    assert_nested_equal(load(full / "training.pt")["optimizer"], load(resumed / "training.pt")["optimizer"])
    rows = [json.loads(line) for line in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()]
    base = rows[2]["lr"]
    assert [row["lr"] / base for row in rows[3:]] == pytest.approx([1.0, 2 / 3, 1 / 3])


def test_cooldown_cannot_change_once_started(tmp_path):
    run_train(tmp_path / "run", 5, "--cooldown-start", "4", "--cooldown-end", "8")
    result = run_train(tmp_path / "run", 8, "--resume", "latest", "--cooldown-start", "4",
                       "--cooldown-end", "9", check=False)
    assert result.returncode != 0 and "cannot change after the cooldown started" in result.stdout + result.stderr


def test_warm_start_continues_update_count_and_weights(tmp_path):
    source = run_train(tmp_path / "source", 3)
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    # A vanishing learning rate leaves the weights unchanged, proving the source was loaded.
    tiny = ("--init-from", str(source), "--extra-emas", "--learning-rate", "1e-30")
    warm = run_train(tmp_path / "warm", 5, *tiny)
    rows = [json.loads(line) for line in (tmp_path / "warm" / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [4, 5]
    assert "Warm start from" in (tmp_path / "warm" / "train.log").read_text()
    weights, original = load(warm / "weights.pt"), load(source / "weights.pt")
    assert_nested_equal(weights["model"], original["model"])
    assert {"ema_short", "ema_long"} <= weights.keys()
    # Resuming the warm run uses its own checkpoints; init_from is then ignored.
    run_train(tmp_path / "warm", 6, *tiny, "--resume", "latest")
    rows = [json.loads(line) for line in (tmp_path / "warm" / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [4, 5, 6]


def test_resume_is_exact_with_extra_emas_and_bf16(tmp_path):
    extra = ("--extra-emas", "--precision", "bf16")
    full = run_train(tmp_path / "full", 3, *extra)
    run_train(tmp_path / "resumed", 2, *extra)
    resumed = run_train(tmp_path / "resumed", 3, *extra, "--resume", "latest")
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    a, b = load(full / "weights.pt"), load(resumed / "weights.pt")
    for key in ("model", "ema", "ema_short", "ema_long"):
        assert_nested_equal(a[key], b[key])
    assert_nested_equal(load(full / "training.pt")["optimizer"], load(resumed / "training.pt")["optimizer"])


INJECT_NONFINITE = """
import sys, torch
import cifar10.train as trainer
calls = [0]
real = trainer.meanflow_loss
def flaky(*args, **kwargs):
    calls[0] += 1
    loss, mse = real(*args, **kwargs)
    return (loss * float("inf"), mse) if calls[0] == 5 else (loss, mse)
trainer.meanflow_loss = flaky
trainer.main(sys.argv[1:])
"""


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_nonfinite_loss_skips_update_in_mixed_precision_only(tmp_path, precision):
    output = tmp_path / precision
    command = [sys.executable, "-c", INJECT_NONFINITE, "--dataset", "cifar10", "--smoke", "--model", "DiT-Tiny/4",
               "--device", "cpu", "--per-device-batch-size", "2", "--gradient-accumulation-steps", "2",
               "--max-train-steps", "4", "--warmup-steps", "2", "--num-workers", "0",
               "--checkpointing-steps", "10", "--precision", precision, "--output-dir", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    if precision == "fp32":
        assert result.returncode == 3 and (output / "divergence.json").is_file()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (output / "divergence.json").exists()
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert not any(row.get("nonfinite_loss_skipped") for row in rows)
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert [row["committed"] for row in attempts] == [True, True, False, True, True]
    assert attempts[2]["nonfinite_loss"]
    assert attempts[-1]["attempted_updates"] == 5
    assert attempts[-1]["successful_updates"] == 4
    weights = torch.load(output / "checkpoints" / "step-00000004" / "weights.pt", weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights["model"].values())
    assert "skipping this update" in (output / "train.log").read_text()


def test_dropout_v1_adamw_resume_is_exact(tmp_path):
    extra = ("--extra-emas", "--attn-projection-dropout", "0.1", "--mlp-output-dropout", "0.1",
             "--time-sampler", "v1", "--optimizer", "adamw", "--weight-decay", "0.01")
    full = run_train(tmp_path / "full_dropout", 3, *extra)
    run_train(tmp_path / "resumed_dropout", 2, *extra)
    resumed = run_train(tmp_path / "resumed_dropout", 3, *extra, "--resume", "latest")
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    a, b = load(full / "weights.pt"), load(resumed / "weights.pt")
    for key in ("model", "ema", "ema_short", "ema_long"):
        assert_nested_equal(a[key], b[key])
    for key in ("optimizer", "scheduler", "runtime_state"):
        assert_nested_equal(load(full / "training.pt")[key], load(resumed / "training.pt")[key])
    assert_nested_equal(load(full / "rank-0.pt"), load(resumed / "rank-0.pt"))


def test_fifty_consecutive_skips_terminate_without_successful_updates(tmp_path):
    code = INJECT_NONFINITE.replace('calls[0] == 5', 'True')
    output = tmp_path / "fifty_skips"
    command = [sys.executable, "-c", code, "--dataset", "cifar10", "--smoke", "--model", "DiT-Tiny/4",
               "--device", "cpu", "--per-device-batch-size", "1", "--gradient-accumulation-steps", "2",
               "--max-train-steps", "1", "--warmup-steps", "2", "--num-workers", "0",
               "--checkpointing-steps", "1", "--precision", "bf16", "--output-dir", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    assert result.returncode == 3, result.stdout + result.stderr
    marker = json.loads((output / "divergence.json").read_text())
    assert marker["step"] == 0 and "50 consecutive" in marker["reason"]
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 50
    assert all(not row["committed"] for row in attempts)
    assert not (output / "train.jsonl").exists()
