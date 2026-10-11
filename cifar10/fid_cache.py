"""Provenance-checked real Inception sufficient statistics, without image changes."""

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


REAL_STATES = ("real_features_sum", "real_features_cov_sum", "real_features_num_samples")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def real_statistics_provenance(reference, inception_path, versions):
    """Identify the actual ordered, unaugmented 50k reference pixels."""
    pixels = getattr(reference, "data", None)
    if (len(reference) != 50000 or not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.uint8 or pixels.shape != (50000, 32, 32, 3)):
        raise ValueError("Real FID caching requires the original 50,000 uint8 CIFAR-10 training images")
    pixel_hash = hashlib.sha256(memoryview(np.ascontiguousarray(pixels))).hexdigest()
    return {
        "schema_version": 1, "dataset": "cifar10-train-unaugmented-50000",
        "num_real_samples": 50000, "ordered_nhwc_uint8_sha256": pixel_hash,
        "inception_sha256": sha256_file(inception_path), "versions": dict(versions),
        "backend": "torchmetrics.image.fid.FrechetInceptionDistance",
        "feature": 2048, "normalize": False, "feature_precision": "fp32",
        "tf32": False, "accumulator_precision": "fp64", "real_pixel_conversion": "original_uint8",
    }


def _validate_states(states, provenance):
    if not isinstance(states, dict) or set(states) != set(REAL_STATES):
        raise ValueError("Invalid real FID cache sufficient-statistic fields")
    total, covariance, count = (states[key] for key in REAL_STATES)
    features = provenance["feature"]
    if (not all(isinstance(value, torch.Tensor) for value in states.values())
            or total.shape != (features,) or covariance.shape != (features, features)
            or count.ndim != 0 or total.dtype != torch.float64 or covariance.dtype != torch.float64
            or count.dtype != torch.int64 or count.item() != provenance["num_real_samples"]
            or not torch.isfinite(total).all() or not torch.isfinite(covariance).all()):
        raise ValueError("Invalid real FID cache sufficient-statistic shapes, precision, count, or finiteness")


def load_real_statistics(path, provenance):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(record, dict) or set(record) != {"provenance", "states"}
            or record["provenance"] != provenance):
        raise ValueError("Real FID cache provenance mismatch; use a new cache path for the changed protocol")
    _validate_states(record["states"], provenance)
    return record["states"]


def save_real_statistics(path, metric, provenance):
    """Publish a complete cache atomically; never replace a conflicting cache."""
    path = Path(path)
    states = {key: getattr(metric, key).detach().cpu().clone() for key in REAL_STATES}
    _validate_states(states, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save({"provenance": provenance, "states": states}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            load_real_statistics(path, provenance)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return states


def apply_real_statistics(metric, states, *, rank=0):
    """Only rank zero contributes cached real states to any sum reduction.

    Current evaluation computes metrics locally after the sampling group exits.
    Explicitly zeroing other ranks also prevents accidental world-size copies if
    metric computation becomes distributed in a future caller.
    """
    if type(rank) is not int or rank < 0:
        raise ValueError("Metric rank must be a nonnegative integer")
    for key in REAL_STATES:
        destination = getattr(metric, key)
        source = states[key]
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError(f"Real FID cache does not match metric state {key}")
        destination.copy_(source.to(destination.device) if rank == 0 else torch.zeros_like(destination))
