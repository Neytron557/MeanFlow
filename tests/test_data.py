"""Offline checks for the CIFAR input and paper augmentation pipeline."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from cifar10.augmentation import augment_batch
from cifar10.data import SyntheticCIFAR10, load_cifar10, main, prepare_cifar10, preprocess_image


def test_preprocess_preserves_rgb_geometry_and_scales_uint8():
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[..., 1] = 127
    pixels[..., 2] = 255
    result = preprocess_image(Image.fromarray(pixels))
    assert result.shape == (3, 32, 32)
    assert result.dtype == torch.float32
    assert torch.equal(result[0], torch.full((32, 32), -1.0))
    assert torch.equal(result[2], torch.ones(32, 32))
    torch.testing.assert_close(result[1], torch.full((32, 32), 127 / 127.5 - 1))


def test_preprocess_tensor_is_deterministic_and_validated():
    pixels = torch.randint(256, (3, 32, 32), dtype=torch.uint8)
    state = torch.get_rng_state()
    first = preprocess_image(pixels)
    assert torch.equal(first, preprocess_image(pixels))
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(ValueError, match="uint8"):
        preprocess_image(pixels.float())
    with pytest.raises(ValueError, match="3, 32, 32"):
        preprocess_image(pixels[:, :16])


def test_synthetic_is_by_index_and_does_not_consume_training_rng():
    dataset = SyntheticCIFAR10(size=8, seed=31)
    state = torch.get_rng_state()
    image, label = dataset[3]
    assert len(dataset) == 8
    assert image.shape == (3, 32, 32)
    assert image.min() >= -1 and image.max() <= 1
    assert label == 3
    assert torch.equal(image, dataset[3][0])
    assert not torch.equal(image, dataset[4][0])
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(IndexError):
        dataset[8]
    with pytest.raises(ValueError):
        SyntheticCIFAR10(size=0)


def test_missing_real_data_has_preparation_instruction_and_never_downloads(tmp_path):
    with pytest.raises(RuntimeError, match=r"python -m cifar10.data"):
        load_cifar10(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_prepare_validates_both_splits_and_respects_offline(monkeypatch, tmp_path):
    calls = []

    class FakeCIFAR10:
        def __init__(self, root, train, transform=None, download=False):
            calls.append((root, train, download, transform))
            self.train = train

        def __len__(self):
            return 50000 if self.train else 10000

    monkeypatch.setattr("torchvision.datasets.CIFAR10", FakeCIFAR10)
    assert prepare_cifar10(tmp_path, download=False) == {"train": 50000, "test": 10000}
    assert [(train, download) for _, train, download, _ in calls] == [(True, False), (False, False)]
    calls.clear()
    load_cifar10(tmp_path, download=True)
    assert calls[0][1:3] == (True, True)
    assert calls[0][3] is preprocess_image


def test_prepare_cli_reports_verified_splits_and_offline_mode(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_prepare(root, download):
        calls.append((root, download))
        return {"train": 50000, "test": 10000}

    monkeypatch.setattr("cifar10.data.prepare_cifar10", fake_prepare)
    main(["--data-dir", str(tmp_path), "--offline"])
    assert calls == [(tmp_path, False)]
    report = json.loads(capsys.readouterr().out)
    assert report["verified_images"] == {"train": 50000, "test": 10000}
    assert report["data_dir"] == str(tmp_path.resolve())


def test_explicit_download_failure_is_not_hidden(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Server unavailable")

    monkeypatch.setattr("torchvision.datasets.CIFAR10", unavailable)
    with pytest.raises(RuntimeError, match="^Server unavailable$"):
        load_cifar10(tmp_path, download=True)


def test_disabled_augmentation_returns_zero_condition_without_rng():
    images = torch.randn(4, 3, 32, 32)
    state = torch.get_rng_state()
    augmented, labels = augment_batch(images, enabled=False)
    assert augmented is images
    assert labels.shape == (4, 6)
    assert labels.dtype == images.dtype
    assert torch.count_nonzero(labels) == 0
    assert torch.equal(torch.get_rng_state(), state)


def test_full_augmentation_repeats_exactly_when_rank_rng_is_restored():
    images = torch.linspace(-1, 1, 32 * 3 * 32 * 32).reshape(32, 3, 32, 32)
    torch.manual_seed(11)
    state = torch.get_rng_state()
    first, first_labels = augment_batch(images)
    torch.set_rng_state(state)
    second, second_labels = augment_batch(images)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(first_labels, second_labels)
    assert first.shape == images.shape
    assert first_labels.shape == (32, 6)
    assert torch.isfinite(first).all() and torch.isfinite(first_labels).all()
    assert torch.count_nonzero(first_labels[:, 1:]) > 0
    assert not torch.equal(first, images)


@pytest.mark.parametrize("images", [torch.zeros(3, 32, 32), torch.zeros(1, 1, 32, 32), torch.zeros(1, 3, 16, 16), torch.zeros(0, 3, 32, 32), torch.zeros(1, 3, 32, 32, dtype=torch.uint8)])
def test_augmentation_rejects_invalid_inputs(images):
    with pytest.raises(ValueError):
        augment_batch(images)
