"""FID real-statistic reuse preserves the measured score and real sample count."""

import numpy as np
import pytest
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from cifar10.fid_cache import (REAL_STATES, apply_real_statistics, load_real_statistics,
                              real_statistics_provenance, save_real_statistics)


class ChannelFeatures(torch.nn.Module):
    num_features = 3

    def forward(self, pixels):
        return pixels.float().mean(dim=(-2, -1))


def make_metric():
    return FrechetInceptionDistance(feature=ChannelFeatures(), normalize=False, sync_on_compute=False)


def test_cached_and_uncached_identical_images_have_identical_fid(tmp_path):
    generator = torch.Generator().manual_seed(19)
    real = torch.randint(0, 256, (23, 3, 8, 8), dtype=torch.uint8, generator=generator)
    fake = torch.randint(0, 256, (17, 3, 8, 8), dtype=torch.uint8, generator=generator)
    provenance = {"feature": 3, "num_real_samples": 23, "protocol": "same_images_custom_feature_test"}
    original = make_metric()
    original.update(real, real=True)
    cache = tmp_path / "real.pt"
    save_real_statistics(cache, original, provenance)
    original.update(fake, real=False)
    restored = make_metric()
    apply_real_statistics(restored, load_real_statistics(cache, provenance))
    restored.update(fake, real=False)
    torch.testing.assert_close(restored.compute(), original.compute(), rtol=0, atol=0)
    assert restored.real_features_num_samples.item() == len(real)


def test_cached_real_statistics_are_contributed_once_across_ranks(tmp_path):
    original = make_metric()
    original.update(torch.randint(0, 255, (11, 3, 4, 4), dtype=torch.uint8), real=True)
    provenance = {"feature": 3, "num_real_samples": 11}
    states = save_real_statistics(tmp_path / "real.pt", original, provenance)
    metrics = [make_metric() for _ in range(4)]
    for rank, metric in enumerate(metrics):
        metric.update(torch.ones((2, 3, 4, 4), dtype=torch.uint8), real=True)
        apply_real_statistics(metric, states, rank=rank)
    for state in REAL_STATES:
        reduced = sum(getattr(metric, state) for metric in metrics)
        torch.testing.assert_close(reduced, states[state], rtol=0, atol=0)


def test_cache_rejects_changed_provenance_and_invalid_count(tmp_path):
    metric = make_metric()
    metric.update(torch.ones((3, 3, 4, 4), dtype=torch.uint8), real=True)
    provenance = {"feature": 3, "num_real_samples": 3, "inception_sha256": "one"}
    cache = tmp_path / "real.pt"
    save_real_statistics(cache, metric, provenance)
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        load_real_statistics(cache, {**provenance, "inception_sha256": "other"})
    with pytest.raises(ValueError, match="provenance"):
        save_real_statistics(cache, metric, {**provenance, "inception_sha256": "other"})
    assert cache.read_bytes() == before
    record = torch.load(cache, weights_only=True)
    record["states"]["real_features_num_samples"] += 1
    torch.save(record, cache)
    with pytest.raises(ValueError, match="count"):
        load_real_statistics(cache, provenance)


def test_provenance_identifies_actual_real_pixels_inception_and_versions(tmp_path):
    class Reference:
        data = np.zeros((50000, 32, 32, 3), dtype=np.uint8)

        def __len__(self):
            return 50000

    weights = tmp_path / "inception.pth"
    weights.write_bytes(b"local_inception")
    reference = Reference()
    provenance = real_statistics_provenance(reference, weights, {"torchmetrics": "version"})
    assert provenance["num_real_samples"] == 50000 and provenance["feature"] == 2048
    assert provenance["real_pixel_conversion"] == "original_uint8"
    reference.data[0, 0, 0, 0] = 1
    changed = real_statistics_provenance(reference, weights, {"torchmetrics": "version"})
    assert changed["ordered_nhwc_uint8_sha256"] != provenance["ordered_nhwc_uint8_sha256"]
    assert changed["inception_sha256"] == provenance["inception_sha256"]
    class PartialReference(Reference):
        def __len__(self):
            return 3

    with pytest.raises(ValueError, match="50,000"):
        real_statistics_provenance(PartialReference(), weights, {})
