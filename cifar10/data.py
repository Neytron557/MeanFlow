"""Deterministic CIFAR-10 inputs and explicit, checksum-verified preparation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


_SPLIT_FIELDS = {"schema_version", "dataset", "source_split", "dataset_size", "train_size",
                 "holdout_size", "seed", "algorithm", "train_indices", "holdout_indices", "sha256"}


def _split_hash(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_cifar10_split_manifest(seed: int = 20260930) -> dict:
    """Define a reproducible 45k/5k split without changing any training RNG.

    The SHA256 identifies the canonical JSON payload, excluding its own digest;
    JSON whitespace and key order do not affect the scientific partition identity.
    Indices refer to torchvision's checksum-verified 50,000-image training split.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Split seed must be an integer in the uint32 range")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(50000)
    manifest = {
        "schema_version": 1, "dataset": "CIFAR-10", "source_split": "train",
        "dataset_size": 50000, "train_size": 45000, "holdout_size": 5000,
        "seed": seed, "algorithm": "numpy.PCG64.permutation;sorted-partitions",
        "train_indices": np.sort(permutation[:45000]).tolist(),
        "holdout_indices": np.sort(permutation[45000:]).tolist(),
    }
    manifest["sha256"] = _split_hash(manifest)
    return manifest


def _validate_split_manifest(manifest: dict, expected_sha256: str = "") -> dict:
    if not isinstance(manifest, dict) or set(manifest) != _SPLIT_FIELDS:
        raise ValueError("Invalid CIFAR-10 split manifest fields")
    constants = {"schema_version": 1, "dataset": "CIFAR-10", "source_split": "train",
                 "dataset_size": 50000, "train_size": 45000, "holdout_size": 5000,
                 "algorithm": "numpy.PCG64.permutation;sorted-partitions"}
    for key, expected in constants.items():
        if type(manifest[key]) is not type(expected) or manifest[key] != expected:
            raise ValueError(f"Invalid CIFAR-10 split manifest {key}")
    seed = manifest["seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Invalid CIFAR-10 split manifest seed")
    for key, count in (("train_indices", 45000), ("holdout_indices", 5000)):
        indices = manifest[key]
        if (not isinstance(indices, list) or len(indices) != count or
                any(type(index) is not int or not 0 <= index < 50000 for index in indices) or
                any(left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError(f"Invalid sorted unique CIFAR-10 split {key}")
    if set(manifest["train_indices"]) & set(manifest["holdout_indices"]):
        raise ValueError("CIFAR-10 training and holdout indices overlap")
    digest = manifest["sha256"]
    if not isinstance(digest, str) or digest != _split_hash(manifest):
        raise ValueError("CIFAR-10 split manifest SHA256 mismatch")
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("CIFAR-10 split differs from the configured SHA256")
    # A correct digest alone must not permit relabeling a different partition
    # with the declared seed and algorithm.
    expected = make_cifar10_split_manifest(seed)
    if digest != expected["sha256"]:
        raise ValueError("CIFAR-10 split indices do not match its seed and algorithm")
    return manifest


def read_cifar10_split_manifest(path: str | Path, expected_sha256: str = "") -> dict:
    with Path(path).open(encoding="utf-8") as source:
        manifest = json.load(source)
    return _validate_split_manifest(manifest, expected_sha256)


def write_cifar10_split_manifest(path: str | Path, seed: int = 20260930) -> dict:
    """Create once; an existing manifest must contain the identical partition."""
    path = Path(path)
    manifest = make_cifar10_split_manifest(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, sort_keys=True, indent=2)
            destination.write("\n")
    except FileExistsError:
        return read_cifar10_split_manifest(path, manifest["sha256"])
    return manifest


def load_cifar10_partition(dataset, manifest: dict | str | Path, *, partition: str = "train",
                          expected_sha256: str = ""):
    """Apply the manifest to an already verified dataset, including full-stage lineage.

    ``holdout`` is available for read-only diagnostics; TrainConfig disallows
    training on it. ``full`` validates the manifest and returns the 50k dataset
    itself, so the full-data stage retains ordinary CIFAR-10 loader behavior.
    """
    if partition not in ("train", "holdout", "full"):
        raise ValueError("CIFAR-10 partition must be train, holdout, or full")
    if len(dataset) != 50000:
        raise ValueError("CIFAR-10 split manifests require the 50,000-image training dataset")
    manifest = (_validate_split_manifest(manifest, expected_sha256) if isinstance(manifest, dict)
                else read_cifar10_split_manifest(manifest, expected_sha256))
    return dataset if partition == "full" else Subset(dataset, manifest[f"{partition}_indices"])


def preprocess_image(image) -> torch.Tensor:
    """Convert an RGB PIL image or CHW uint8 tensor to 32x32 FP32 in [-1, 1]."""
    if not isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import pil_to_tensor

        image = pil_to_tensor(image)
    if image.dtype != torch.uint8:
        raise ValueError("CIFAR-10 preprocessing requires uint8 pixels")
    if tuple(image.shape) != (3, 32, 32):
        raise ValueError("CIFAR-10 images must have shape (3, 32, 32)")
    return image.to(torch.float32).div(127.5).sub(1)


def _load_split(root: str | Path, *, train: bool, download: bool):
    from torchvision.datasets import CIFAR10

    try:
        # torchvision verifies extracted batches (and the archive on download)
        # against known MD5 hashes. Random transforms stay out of workers.
        return CIFAR10(str(root), train=train, transform=preprocess_image, download=download)
    except RuntimeError as error:
        if download:
            raise
        raise RuntimeError(
            f"CIFAR-10 is missing or failed checksum verification at {root}. "
            f"Prepare it on a connected machine with: python -m cifar10.data "
            f"--data-dir {str(root)!r}. Original error: {error}"
        ) from error


def load_cifar10(root: str | Path, download: bool = False):
    """Load the 50,000 training images; network access requires explicit opt-in."""
    return _load_split(root, train=True, download=download)


def prepare_cifar10(root: str | Path, download: bool = True) -> dict[str, int]:
    """Download/check both splits; set download=False for an offline verification."""
    return {
        name: len(_load_split(root, train=train, download=download))
        for name, train in (("train", True), ("test", False))
    }


class SyntheticCIFAR10(Dataset):
    """Deterministic by-index random pixels for offline smoke checks, never data training."""

    def __init__(self, size: int = 128, seed: int = 0):
        if size <= 0:
            raise ValueError("Synthetic dataset size must be positive")
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        pixels = torch.randint(256, (3, 32, 32), dtype=torch.uint8, generator=generator)
        return preprocess_image(pixels), index % 10


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/cifar10", type=Path)
    parser.add_argument("--offline", action="store_true", help="Verify local data without any downloads")
    args = parser.parse_args(argv)
    counts = prepare_cifar10(args.data_dir, download=not args.offline)
    print(json.dumps({"data_dir": str(args.data_dir.resolve()), "verified_images": counts}, indent=2))


if __name__ == "__main__":
    main()
