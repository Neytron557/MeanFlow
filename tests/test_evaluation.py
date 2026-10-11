"""Sampling and evaluation checks that do not download datasets or weights."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from PIL import Image

from cifar10.evaluate import (
    compute_fid,
    ensure_empty_output,
    inception_cache_path,
    main,
    parse_args,
    require_metric_cache,
    sample_index_batches,
)
from cifar10.sampling import meanflow_sample, to_uint8


class ConstantVelocity(torch.nn.Module):
    def forward(self, images, r, t, augment_labels=None):
        assert augment_labels is None
        assert torch.all(r <= t)
        return torch.full_like(images, 0.25)


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_constant_velocity_sampling_is_exact(steps):
    noise = torch.randn(3, 3, 32, 32)
    sampled = meanflow_sample(ConstantVelocity(), noise, num_steps=steps)
    torch.testing.assert_close(sampled, noise - 0.25)
    assert not sampled.requires_grad


def test_one_step_calls_correct_endpoints_once():
    class Field(torch.nn.Module):
        def forward(self, images, r, t, augment_labels=None):
            assert torch.equal(r, torch.zeros(2))
            assert torch.equal(t, torch.ones(2))
            return images * 0.5

    noise = torch.ones(2, 3, 32, 32)
    torch.testing.assert_close(meanflow_sample(Field(), noise), noise * 0.5)


@pytest.mark.parametrize("steps", [0, -1, 1.5])
def test_invalid_step_count_is_rejected(steps):
    with pytest.raises(ValueError, match="positive integer"):
        meanflow_sample(ConstantVelocity(), torch.zeros(1, 3, 32, 32), steps)


def test_png_conversion_clamps_and_floors():
    values = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    assert to_uint8(values).tolist() == [0, 0, 127, 191, 255, 255]


@pytest.mark.parametrize("total,world,batch", [(17, 4, 3), (2, 4, 16), (50000, 4, 16)])
def test_distributed_indices_are_exact_and_disjoint(total, world, batch):
    indices = []
    for rank in range(world):
        for group in sample_index_batches(total, rank, world, batch):
            assert 0 < len(group) <= batch
            indices.extend(group)
    assert sorted(indices) == list(range(total))


def test_nonempty_output_is_rejected_without_deleting_files(tmp_path):
    stale = tmp_path / "000001.png"
    stale.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        ensure_empty_output(tmp_path)
    assert stale.read_bytes() == b"keep"


def test_cache_preflight_refuses_download(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert inception_cache_path().parent == tmp_path / "hub" / "checkpoints"
    with pytest.raises(FileNotFoundError, match="prepare-metrics"):
        require_metric_cache()


def test_prepare_metrics_does_not_require_checkpoint():
    args = parse_args(["--prepare-metrics", "--device", "cpu"])
    assert args.prepare_metrics and args.ckpt is None


@pytest.mark.parametrize("arguments", [
    [], ["--ckpt", "x", "--num-steps", "0"],
    ["--ckpt", "x", "--seed", "-1"],
    ["--ckpt", "x", "--compute-metrics", "--num-fid-samples", "1"],
])
def test_invalid_cli_is_rejected(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_metric_preparation_is_explicit(tmp_path, monkeypatch, capsys):
    import cifar10.evaluate as evaluate

    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    def fake_download(url, **kwargs):
        assert kwargs["check_hash"] is True
        assert kwargs["weights_only"] is True
        assert url == evaluate.INCEPTION_URL
        path = inception_cache_path()
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test-cache")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_download)
    main(parse_args(["--prepare-metrics", "--device", "cpu"]))
    assert "Inception weights ready" in capsys.readouterr().out


def test_fid_protocol_uses_unaugmented_uint8_and_local_compute(tmp_path, monkeypatch):
    import torchvision.datasets
    import torchmetrics.image.fid

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache"))
    cached = inception_cache_path()
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not-loaded-by-fake-metric")
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        Image.new("RGB", (32, 32), (index, 0, 0)).save(images / f"{index:06d}.png")
    observed = {"real": 0, "fake": 0}

    def reference(root, train, download, transform):
        assert train is True and download is False
        sample = transform(Image.new("RGB", (32, 32)))
        assert sample.dtype == torch.uint8 and sample.shape == (3, 32, 32)
        return torch.utils.data.TensorDataset(sample.repeat(4, 1, 1, 1), torch.zeros(4))

    class Metric:
        def __init__(self, **kwargs):
            assert kwargs == {
                "feature": 2048, "normalize": False, "sync_on_compute": False,
                "feature_extractor_weights_path": str(cached),
            }

        def to(self, device):
            return self

        def update(self, batch, real):
            assert batch.dtype == torch.uint8
            observed["real" if real else "fake"] += len(batch)

        def compute(self):
            return torch.tensor(3.5)

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", reference)
    monkeypatch.setattr(torchmetrics.image.fid, "FrechetInceptionDistance", Metric)
    args = parse_args(["--ckpt", "unused", "--num-fid-samples", "3", "--device", "cpu"])
    assert compute_fid(args, images, torch.device("cpu")) == 3.5
    assert observed == {"real": 4, "fake": 3}


def test_missing_fid_reference_fails_before_sampling(tmp_path, monkeypatch):
    import torchvision.datasets

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache"))
    cached = inception_cache_path()
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"test-cache")

    def missing_reference(*args, **kwargs):
        raise RuntimeError("CIFAR reference is missing")

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", missing_reference)
    args = parse_args(["--ckpt", "unread-checkpoint", "--compute-metrics", "--device", "cpu"])
    with pytest.raises(RuntimeError, match="CIFAR reference is missing"):
        main(args)


def test_tiny_cpu_evaluation_uses_ema_and_writes_exact_samples(tmp_path):
    from cifar10.model import create_model

    model = create_model("DiT-Tiny/4", augment_dim=6)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "model": {},  # Evaluation must select EMA, never the training weights.
            "ema": model.state_dict(),
            "config": {"model": "DiT-Tiny/4", "augment_dim": 6},
            "step": 7,
        },
        checkpoint / "weights.pt",
    )
    output = tmp_path / "samples"
    args = parse_args([
        "--ckpt", str(checkpoint), "--sample-dir", str(output),
        "--num-fid-samples", "3", "--per-proc-batch-size", "2",
        "--device", "cpu", "--seed", "9",
    ])
    main(args)
    images = sorted((output / "images").glob("*.png"))
    assert [image.name for image in images] == ["000000.png", "000001.png", "000002.png"]
    with Image.open(images[0]) as image:
        assert image.mode == "RGB" and image.size == (32, 32)
    results = json.loads((output / "metrics.json").read_text())
    assert results["num_samples"] == 3
    assert results["checkpoint_step"] == 7
    assert results["weights"] == "ema"
    assert results["seed"] == 9
    assert results["fid"] is None
    assert results["versions"]["torch"]
    assert len(results["checkpoint_sha256"]) == 64
    assert results["sample_grid_indices"] == [0, 1, 2]
    with Image.open(output / "grid.png") as grid, Image.open(images[0]) as first:
        assert grid.size == (288, 96)
        assert grid.getpixel((0, 0)) == first.getpixel((0, 0))
    with pytest.raises(FileExistsError, match="nonempty"):
        main(args)


def test_two_rank_cpu_evaluation_writes_exact_nondivisible_count(tmp_path):
    from cifar10.model import create_model

    checkpoint = tmp_path / "weights.pt"
    torch.save({
        "ema": create_model("DiT-Tiny/4", augment_dim=6).state_dict(),
        "config": {"model": "DiT-Tiny/4", "augment_dim": 6}, "step": 0,
    }, checkpoint)
    output = tmp_path / "distributed"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--master-addr=127.0.0.1", f"--master-port={port}",
        str(Path(__file__).resolve().parents[1] / "evaluate.py"), "--dataset", "cifar10",
        "--ckpt", str(checkpoint), "--sample-dir", str(output),
        "--num-fid-samples", "7", "--per-proc-batch-size", "2", "--device", "cpu",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((output / "images").glob("*.png"))) == 7
    assert json.loads((output / "metrics.json").read_text())["world_size"] == 2


def test_online_weights_are_selectable_and_recorded(tmp_path):
    from cifar10.evaluate import load_checkpoint_weights
    from cifar10.model import create_model

    torch.manual_seed(3)
    online, ema = create_model("DiT-Tiny/4", augment_dim=6), create_model("DiT-Tiny/4", augment_dim=6)
    with torch.no_grad():  # Distinct nonzero outputs; zero-initialized heads would hide the choice.
        for model in (online, ema):
            model.final_layer.linear.weight.normal_(std=0.05)
            for block in model.blocks:
                block.adaLN_modulation[-1].weight.normal_(std=0.05)
    path = tmp_path / "weights.pt"
    # A pre-v2 config (no new keys) must still load.
    torch.save({"model": online.state_dict(), "ema": ema.state_dict(),
                "config": {"model": "DiT-Tiny/4", "augment_dim": 6}, "step": 5}, path)
    noise = torch.randn(2, 3, 32, 32)
    loaded = {w: load_checkpoint_weights(str(path), torch.device("cpu"), w)[0] for w in ("model", "ema")}
    torch.testing.assert_close(meanflow_sample(loaded["model"], noise), meanflow_sample(online.eval(), noise))
    assert not torch.equal(meanflow_sample(loaded["model"], noise), meanflow_sample(loaded["ema"], noise))
    output = tmp_path / "samples"
    main(parse_args(["--ckpt", str(path), "--sample-dir", str(output), "--num-fid-samples", "2",
                     "--per-proc-batch-size", "2", "--device", "cpu", "--weights", "model"]))
    assert json.loads((output / "metrics.json").read_text())["weights"] == "model"
    torch.save({"model": {}, "ema": ema.state_dict(), "config": {"model": "DiT-Tiny/4", "augment_dim": 6},
                "step": 5}, path)
    with pytest.raises(ValueError, match="empty"):
        load_checkpoint_weights(str(path), torch.device("cpu"), "model")
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", str(path), "--weights", "online"])


def test_evaluation_allocator_limits_are_explicit_and_collective(monkeypatch):
    import cifar10.evaluate as evaluate
    import cifar10.runtime as runtime

    args = parse_args(["--ckpt", "unused", "--device", "cpu",
                       "--gpu-memory-limit-gib", "64", "--gpu-free-margin-gib", "8"])
    observed = []

    def configure(config, device):
        observed.append((config.gpu_memory_limit_gib, config.gpu_free_margin_gib))
        return {"allocator_limit_gib": 64., "initial_free_gib": 80., "free_margin_gib": 8.}

    monkeypatch.setattr(runtime, "configure_gpu_memory", configure)
    receipt = evaluate._configure_evaluation_memory(args, 0, 1, torch.device("cpu"))
    assert observed == [(64., 8.)]
    assert receipt["gpu_free_margin_gib"] == 8.
    assert receipt["initial_limits_by_rank"][0]["limits"]["allocator_limit_gib"] == 64.

    def gather(records, local):
        records[:] = [local, {"rank": 1, "limits": None, "error": "peer has insufficient free memory"}]

    monkeypatch.setattr(evaluate.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1"):
        evaluate._configure_evaluation_memory(args, 0, 2, torch.device("cpu"))
    for option, value in (("--gpu-memory-limit-gib", "-1"), ("--gpu-free-margin-gib", "nan"),
                          ("--gpu-memory-limit-gib", "inf")):
        with pytest.raises(SystemExit):
            parse_args(["--ckpt", "unused", option, value])
