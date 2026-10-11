"""Offline CIFAR-10 sampling (EMA or online weights) and optional TorchMetrics FID evaluation."""

import argparse
from datetime import datetime, timezone
from datetime import timedelta
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from .sampling import flow_sample, sampling_nfe, sampling_times, to_uint8
from .fid_cache import (apply_real_statistics, load_real_statistics,
                        real_statistics_provenance, save_real_statistics)


INCEPTION_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    "weights-inception-2015-12-05-6726825d.pth"
)


def inception_cache_path() -> Path:
    """Use torch.hub's cache, which honors the TORCH_HOME environment variable."""
    return Path(torch.hub.get_dir()) / "checkpoints" / INCEPTION_URL.rsplit("/", 1)[1]


def require_metric_cache() -> Path:
    path = inception_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Inception weights at {path}. Run "
            "python evaluate.py --dataset cifar10 --prepare-metrics first."
        )
    return path


def prepare_metric_cache() -> Path:
    """Explicitly download the published, hash-checked Inception weights."""
    torch.hub.load_state_dict_from_url(
        INCEPTION_URL, map_location="cpu", check_hash=True, progress=True, weights_only=True,
    )
    return require_metric_cache()


def sample_index_batches(total: int, rank: int, world_size: int, batch_size: int):
    """Partition sample identities exactly, including a partial final batch."""
    if min(total, world_size, batch_size) <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid sample count, batch size, rank, or world size")
    indices = range(rank, total, world_size)
    for offset in range(0, len(indices), batch_size):
        yield list(indices[offset : offset + batch_size])


def ensure_empty_output(path: Path) -> None:
    """Never mix an evaluation with stale samples or overwrite another run."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing nonempty evaluation output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir()


WEIGHT_SETS = ("ema", "model", "ema_short", "ema_long")


def load_checkpoint_weights(checkpoint_path: str, device: torch.device, weights: str = "ema"):
    """Build the model from weights.pt metadata and load the EMA or online state dict."""
    from .model import create_model

    if weights not in WEIGHT_SETS:
        raise ValueError(f"weights must be one of {WEIGHT_SETS}")
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        path = path / "weights.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {weights, "config", "step"} <= checkpoint.keys():
        raise ValueError(f"Expected a CIFAR-10 weights.pt checkpoint with {weights}, config, and step")
    if not checkpoint[weights]:
        raise ValueError(f"Checkpoint has an empty {weights!r} state dict")
    config = checkpoint["config"]
    if not isinstance(config, dict) or not {"model", "augment_dim"} <= config.keys():
        raise ValueError("Checkpoint config must declare model and augment_dim")
    model = create_model(
        config["model"], augment_dim=config["augment_dim"],
        attn_projection_dropout=config.get("attn_projection_dropout", 0.0),
        mlp_output_dropout=config.get("mlp_output_dropout", 0.0),
        dropout_mask_mode=config.get("dropout_mask_mode", "indexed"),
    )
    model.load_state_dict(checkpoint[weights], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, config, int(checkpoint["step"]), path.resolve()


def _distributed_device(requested: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke evaluation")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    return rank, world_size, device


def _prepare_output_distributed(path: Path, rank: int, world_size: int):
    error = [None]
    if rank == 0:
        try:
            ensure_empty_output(path)
        except OSError as exception:
            error[0] = str(exception)
    if world_size > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise FileExistsError(error[0])


def _configure_evaluation_memory(args, rank, world_size, device):
    """Apply initial allocator limits and make a failed preflight collective."""
    from .runtime import configure_gpu_memory

    configuration = SimpleNamespace(gpu_memory_limit_gib=getattr(args, "gpu_memory_limit_gib", 0.),
                                    gpu_free_margin_gib=getattr(args, "gpu_free_margin_gib", 0.))
    local = {"rank": rank, "limits": None, "error": None}
    try:
        local["limits"] = configure_gpu_memory(configuration, device)
    except (RuntimeError, ValueError) as error:
        local["error"] = str(error)
    records = [local]
    if world_size > 1:
        records = [None] * world_size
        dist.all_gather_object(records, local)
    errors = [f"rank {record['rank']}: {record['error']}" for record in records if record["error"]]
    if errors:
        raise RuntimeError("Evaluation GPU memory preflight failed: " + "; ".join(errors))
    return {"gpu_memory_limit_gib": configuration.gpu_memory_limit_gib,
            "gpu_free_margin_gib": configuration.gpu_free_margin_gib,
            "initial_limits_by_rank": records, "policy": "initial_margin_and_allocator_cap"}


def _sample_images(model, args, output: Path, rank: int, world_size: int, device):
    for indices in sample_index_batches(
        args.num_fid_samples, rank, world_size, args.per_proc_batch_size
    ):
        # A sample's noise is independent of batch size and distributed topology.
        noise = torch.stack([
            torch.randn(
                3, 32, 32, device=device,
                generator=torch.Generator(device=device).manual_seed((args.seed + index) % (2**63)),
            )
            for index in indices
        ])
        with torch.autocast(device_type=device.type, enabled=False):
            images = flow_sample(
                model, noise, args.num_steps, times=getattr(args, "times", None),
                field=getattr(args, "field", "avg"), solver=getattr(args, "solver", "euler"),
                expected_nfe=getattr(args, "nfe", None),
            )
        if not torch.isfinite(images).all():
            raise FloatingPointError("Sampling produced nonfinite pixels; no FID was computed")
        arrays = to_uint8(images).permute(0, 2, 3, 1).cpu().numpy()
        for index, array in zip(indices, arrays):
            Image.fromarray(array).save(output / "images" / f"{index:06d}.png")


class GeneratedImages(Dataset):
    def __init__(self, image_dir: Path, count: int):
        self.image_dir, self.count = image_dir, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with Image.open(self.image_dir / f"{index:06d}.png") as image:
            array = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(array).permute(2, 0, 1)


def _metric_class():
    try:
        import torch_fidelity  # noqa: F401 -- verify the optional feature backend too.
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exception:
        raise ImportError("FID requires the optional requirements-eval.txt dependencies") from exception
    return FrechetInceptionDistance


def _reference_dataset(data_dir):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import PILToTensor

    return CIFAR10(data_dir, train=True, download=False, transform=PILToTensor())


def compute_fid(args, image_dir: Path, device: torch.device) -> float:
    """Compare generated uint8 images with the unaugmented 50k training set."""
    weights_path = require_metric_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = _reference_dataset(args.data_dir)
    metric = _metric_class()(
        feature=2048, normalize=False, sync_on_compute=False,
        feature_extractor_weights_path=str(weights_path),
    ).to(device)
    loader_kwargs = dict(batch_size=args.per_proc_batch_size, num_workers=0, shuffle=False)
    cache = Path(args.real_stats_cache).expanduser() if getattr(args, "real_stats_cache", None) else None
    args.real_statistics_cache_receipt = None
    provenance = real_statistics_provenance(reference, weights_path, _package_versions()) if cache else None
    cache_hit = cache is not None and cache.is_file()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        if cache_hit:
            apply_real_statistics(metric, load_real_statistics(cache, provenance), rank=0)
        else:
            for images, _ in DataLoader(reference, **loader_kwargs):
                metric.update(images.to(device), real=True)
            if cache is not None:
                save_real_statistics(cache, metric, provenance)
        generated = GeneratedImages(image_dir, args.num_fid_samples)
        for images in DataLoader(generated, **loader_kwargs):
            metric.update(images.to(device), real=False)
        fid = float(metric.compute().cpu())
    if not math.isfinite(fid):
        raise FloatingPointError("FID is nonfinite")
    if cache is not None:
        args.real_statistics_cache_receipt = {
            "path": str(cache.resolve()), "sha256": file_sha256(cache),
            "hit": cache_hit, "provenance": provenance,
        }
    return fid


def _validate_outputs(output: Path, count: int):
    actual = {path.name for path in (output / "images").iterdir()}
    expected = {f"{index:06d}.png" for index in range(count)}
    if actual != expected:
        raise RuntimeError("Generated image set does not match the exact requested sample identities")


def save_sample_grid(output: Path, count: int):
    """Show the first 64 sample identities, with nearest-neighbor enlargement."""
    shown = min(64, count)
    columns = min(8, shown)
    grid = Image.new("RGB", (columns * 32, math.ceil(shown / columns) * 32))
    for index in range(shown):
        with Image.open(output / "images" / f"{index:06d}.png") as image:
            grid.paste(image.convert("RGB"), ((index % columns) * 32, (index // columns) * 32))
    grid.resize((grid.width * 3, grid.height * 3), Image.Resampling.NEAREST).save(output / "grid.png")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions():
    versions = {}
    for package in ("torch", "torchvision", "torchmetrics", "torch-fidelity"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(args):
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.prepare_metrics:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Run --prepare-metrics once, without torchrun")
        print(f"Inception weights ready: {prepare_metric_cache()}")
        return
    args.times = sampling_times(args.num_steps, getattr(args, "times", None))
    args.field, args.solver = getattr(args, "field", "avg"), getattr(args, "solver", "euler")
    args.nfe = sampling_nfe(args.num_steps, field=args.field, solver=args.solver,
                            expected_nfe=getattr(args, "nfe", None))
    if args.compute_metrics:
        weights_path = require_metric_cache()
        _metric_class()
        # Fail before sampling if the offline reference data is unavailable.
        reference = _reference_dataset(args.data_dir)
        cache = getattr(args, "real_stats_cache", None)
        if cache is not None and Path(cache).expanduser().is_file():
            load_real_statistics(Path(cache).expanduser(),
                real_statistics_provenance(reference, weights_path, _package_versions()))
    rank, world_size, device = _distributed_device(args.device)
    output = Path(args.sample_dir).expanduser()
    try:
        memory_limits = _configure_evaluation_memory(args, rank, world_size, device)
        model, config, step, checkpoint = load_checkpoint_weights(args.ckpt, device, args.weights)
        _prepare_output_distributed(output, rank, world_size)
        _sample_images(model, args, output, rank, world_size, device)
        if world_size > 1:
            dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    # Release sampling weights before allocating Inception. Other ranks are done;
    # rank-zero metric computation must never wait for distributed collectives.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank != 0:
        return
    _validate_outputs(output, args.num_fid_samples)
    save_sample_grid(output, args.num_fid_samples)
    sampling_seconds = time.perf_counter() - started
    metric_started = time.perf_counter()
    fid = compute_fid(args, output / "images", device) if args.compute_metrics else None
    metric_seconds = time.perf_counter() - metric_started
    results = {
        "fid": fid, "num_samples": args.num_fid_samples, "num_steps": args.num_steps,
        "nfe": args.nfe, "field": args.field, "solver": args.solver,
        "evaluation_role": "original_meanflow" if args.field == "avg" else "diagonal_field_diagnostic",
        "seed": args.seed, "world_size": world_size,
        "times": list(args.times), "sampling_precision": "fp32", "tf32": False,
        "gpu_memory_limits": memory_limits,
        "sample_identity": {"first_index": 0, "stop_index_exclusive": args.num_fid_samples,
                            "noise_seed_rule": "(seed + sample_index) % 2**63",
                            "first_seed": args.seed % (2**63),
                            "last_seed": (args.seed + args.num_fid_samples - 1) % (2**63),
                            "seed_modulus": 2**63,
                            "seed_range_wraps": (args.seed % (2**63)) + args.num_fid_samples > 2**63},
        "per_proc_batch_size": args.per_proc_batch_size, "checkpoint": str(checkpoint),
        "checkpoint_step": step, "weights": args.weights, "config": config,
        "checkpoint_sha256": file_sha256(checkpoint),
        "num_real_samples": 50000 if args.compute_metrics else None,
        "sampling_seconds": sampling_seconds,
        "metric_seconds": metric_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_grid": "grid.png", "sample_grid_indices": list(range(min(64, args.num_fid_samples))),
        "fid_backend": "torchmetrics.image.fid.FrechetInceptionDistance",
        "fid_features": 2048, "fid_reference": "cifar10-train-unaugmented-50000",
        "inception_sha256": file_sha256(require_metric_cache()) if args.compute_metrics else None,
        "augmentation_conditioning": "zero",
        "real_statistics_cache": getattr(args, "real_statistics_cache_receipt", None),
        "pixel_conversion": "floor(clamp(x * 0.5 + 0.5, 0, 1) * 255)",
        "versions": _package_versions(),
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.num_fid_samples} images to {output}" + (f"; FID={fid:.4f}" if fid is not None else ""))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--ckpt", help="Checkpoint directory or weights.pt")
    parser.add_argument("--sample-dir", default="samples/cifar10", help="An empty directory for this evaluation run")
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--field", choices=["avg", "diagonal"], default="avg")
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--nfe", type=int, help="Verify model calls: Euler steps, or twice the Heun steps")
    parser.add_argument("--times", help="Strictly descending integration endpoints, e.g. 1,0.5,0")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--real-stats-cache", type=Path,
                        help="Build or reuse a provenance-checked cache of the 50k real Inception statistics")
    parser.add_argument("--prepare-metrics", action="store_true", help="Explicitly download Inception weights, then exit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=0., help="Optional per-process allocator cap")
    parser.add_argument("--gpu-free-margin-gib", type=float, default=0., help="Free headroom required at initial preflight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", choices=WEIGHT_SETS, default="ema",
                        help="Sample from the main EMA (default), the online weights, or an extra EMA width "
                             "(ema_short / ema_long, saved when training with --extra-emas)")
    args = parser.parse_args(argv)
    if not args.prepare_metrics and not args.ckpt:
        parser.error("--ckpt is required unless --prepare-metrics is specified")
    if min(args.num_fid_samples, args.per_proc_batch_size, args.num_steps) < 1:
        parser.error("sample count, batch size, and num-steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.gpu_memory_limit_gib, args.gpu_free_margin_gib)):
        parser.error("GPU memory limits and margin must be finite and nonnegative")
    if args.compute_metrics and args.num_fid_samples < 2:
        parser.error("FID requires at least two generated samples")
    try:
        args.times = sampling_times(
            args.num_steps, None if args.times is None else args.times.split(",")
        )
        args.nfe = sampling_nfe(args.num_steps, field=args.field, solver=args.solver,
                                expected_nfe=args.nfe)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    main(parse_args())
