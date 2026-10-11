"""Numerical contracts for the unconditional transformer and MeanFlow loss."""

import os

import pytest
import torch
from torch import nn

from cifar10.model import create_model
from cifar10.objective import meanflow_loss, sample_times
from sit import SiT


class AffineVelocity(nn.Module):
    """A nonzero field with analytically known total time derivative."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, z, r, t, y=None, augment_labels=None):
        self.calls.append(torch.is_grad_enabled())
        aug = 0 if augment_labels is None else augment_labels[:, 0, None, None, None]
        return self.scale * z + 2 * t[:, None, None, None] + 3 * r[:, None, None, None] + aug


def test_analytical_jvp_target_weight_and_gradient_are_detached():
    model = AffineVelocity()
    images = torch.tensor([[[[0.1, -0.7]]], [[[0.8, 0.2]]]])
    noise = torch.tensor([[[[1.2, 0.4]]], [[[-0.5, 0.6]]]])
    r, t = torch.tensor([0.1, 0.3]), torch.tensor([0.8, 0.6])
    augmentation = torch.tensor([[0.2] * 6, [-0.1] * 6])
    z = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
    velocity = noise - images
    derivative = model.scale.detach() * velocity + 2
    target = velocity - (t - r)[:, None, None, None] * derivative
    prediction = model.scale.detach() * z + 2 * t[:, None, None, None] + 3 * r[:, None, None, None] + augmentation[:, 0, None, None, None]
    error = prediction - target
    per_image = error.square().flatten(1).sum(1)
    weights = (per_image + 0.001).pow(-0.75)
    expected_loss = (per_image * weights).mean()
    expected_gradient = (2 * error * z).flatten(1).sum(1).mul(weights).mean()

    loss, mse = meanflow_loss(model, model, images, augmentation, noise=noise, times=(r, t))
    loss.backward()

    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(mse, error.square().mean())
    torch.testing.assert_close(model.scale.grad, expected_gradient)
    assert not mse.requires_grad
    assert model.calls == [False, True], "Detached JVP must run before the trainable forward."


def test_equal_times_reduce_to_flow_matching():
    model = AffineVelocity()
    images = torch.tensor([[[[0.4, 0.7]]]])
    noise = torch.tensor([[[[-0.5, 0.9]]]])
    t = torch.tensor([0.3])
    z = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
    expected = (model.scale.detach() * z + 5 * t[:, None, None, None] - (noise - images)).square()

    loss, mse = meanflow_loss(model, model, images, noise=noise, times=(t, t), adaptive_p=0)

    torch.testing.assert_close(loss.detach(), expected.sum())
    torch.testing.assert_close(mse, expected.mean())


def test_time_sampler_matches_original_joint_distribution():
    torch.manual_seed(17)
    first = torch.sigmoid(torch.randn(1000) * 2 - 2)
    second = torch.sigmoid(torch.randn(1000) * 2 - 2)
    expected_t = torch.maximum(first, second)
    expected_r = torch.minimum(first, second)
    expected_r = torch.where(torch.rand(1000) < 0.25, expected_t, expected_r)
    torch.manual_seed(17)

    r, t = sample_times(1000, "cpu")

    torch.testing.assert_close(r, expected_r)
    torch.testing.assert_close(t, expected_t)
    assert torch.all((0 <= r) & (r <= t) & (t <= 1))


@pytest.mark.parametrize("ratio,equal", [(0.0, True), (1.0, False)])
def test_time_sampler_ratio_endpoints(ratio, equal):
    r, t = sample_times(16, "cpu", ratio=ratio)
    assert torch.all(r == t).item() == equal


@pytest.mark.parametrize("kwargs", [{"ratio": -0.1}, {"ratio": 1.1}, {"time_sigma": -1.0}, {"time_mu": float("nan")}])
def test_invalid_time_sampling_configuration(kwargs):
    with pytest.raises(ValueError):
        sample_times(2, "cpu", **kwargs)


def test_time_sampler_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch_size"):
        sample_times(0, "cpu")


@pytest.mark.parametrize("kwargs,message", [
    ({"norm_eps": 0}, "norm_eps"),
    ({"adaptive_p": -1}, "adaptive_p"),
    ({"noise": torch.zeros(1, 3, 1, 1)}, "noise"),
    ({"times": (torch.zeros(1, 1), torch.ones(1, 1))}, "r and t"),
])
def test_objective_rejects_incompatible_inputs(kwargs, message):
    model = AffineVelocity()
    with pytest.raises(ValueError, match=message):
        meanflow_loss(model, model, torch.zeros(1, 3, 2, 2), **kwargs)


def test_objective_requires_fp32_input():
    model = AffineVelocity()
    with pytest.raises(ValueError, match="FP32"):
        meanflow_loss(model, model, torch.zeros(1, 3, 2, 2, dtype=torch.float16))


def test_base_architecture_matches_dit_b4_without_allocating_weights():
    with torch.device("meta"):
        model = create_model()
    assert len(model.blocks) == 12
    assert model.pos_embed.shape == (1, 64, 768)
    assert not model.pos_embed.requires_grad
    assert model.x_embedder.proj.in_channels == 3
    assert model.x_embedder.patch_size == (4, 4)
    assert model.final_layer.linear.out_features == 4 * 4 * 3
    assert model.blocks[0].attn.num_heads == 12
    assert model.blocks[0].mlp.fc1.out_features == 3072
    assert model.y_embedder is None
    assert model.augment_embedder.in_features == 6
    assert model.augment_embedder.bias is None
    assert all(not block.attn.fused_attn for block in model.blocks)
    assert all(module.p == 0 for module in model.modules() if isinstance(module, nn.Dropout))


def test_zero_initialization_and_zero_augmentation_sampling_default():
    model = create_model("DiT-Tiny/4")
    x, r, t = torch.randn(2, 3, 32, 32), torch.zeros(2), torch.ones(2)
    output = model(x, r, t)
    assert output.shape == x.shape
    assert torch.count_nonzero(output) == 0
    for block in model.blocks:
        assert torch.count_nonzero(block.adaLN_modulation[-1].weight) == 0
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
    torch.testing.assert_close(model(x, r, t), model(x, r, t, augment_labels=torch.zeros(2, 6)))
    assert not torch.equal(model(x, r, t), model(x, r, t, augment_labels=torch.ones(2, 6)))


def test_nonzero_transformer_jvp_and_backward():
    torch.manual_seed(32)
    model = create_model("DiT-Tiny/4")
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.02)
    images = torch.randn(2, 3, 32, 32)
    loss, mse = meanflow_loss(model, model, images, torch.randn(2, 6))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(mse)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.blocks[0].attn.qkv.weight.grad) > 0
    assert torch.count_nonzero(model.augment_embedder.weight.grad) > 0


def test_imagenet_class_conditioned_constructor_is_compatible():
    model = SiT(input_size=4, patch_size=2, in_channels=4, hidden_size=32,
                decoder_hidden_size=32, depth=1, num_heads=4, num_classes=10,
                qk_norm=False, fused_attn=False)
    assert model.y_embedder.embedding_table.num_embeddings == 11
    assert model.augment_embedder is None
    x, r, t = torch.randn(2, 4, 4, 4), torch.zeros(2), torch.ones(2)
    assert model(x, r, t, torch.tensor([0, 9])).shape == x.shape
    torch.testing.assert_close(model(x, r, t), model(x, r, t, torch.zeros(2, dtype=torch.long)))


def test_model_rejects_unknown_architecture_and_wrong_augmentation():
    with pytest.raises(ValueError, match="model"):
        create_model("DiT-NotReal")
    model = create_model("DiT-Tiny/4")
    with pytest.raises(ValueError, match="augment"):
        model(torch.randn(1, 3, 32, 32), torch.zeros(1), torch.ones(1), augment_labels=torch.zeros(1, 5))


def test_disabled_conditioning_rejects_labels():
    with pytest.raises(ValueError, match="augment_dim"):
        create_model("DiT-Tiny/4", augment_dim=-1)
    model = create_model("DiT-Tiny/4", augment_dim=0)
    assert model.augment_embedder is None
    x, r, t = torch.randn(1, 3, 32, 32), torch.zeros(1), torch.ones(1)
    assert model(x, r, t).shape == x.shape
    with pytest.raises(ValueError, match="Class labels"):
        model(x, r, t, y=torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="augmentation"):
        model(x, r, t, augment_labels=torch.zeros(1, 6))


@pytest.mark.skipif(os.getenv("RUN_FULL_MODEL_SMOKE") != "1", reason="Opt-in full DiT-B/4 CPU/GPU smoke")
def test_full_dit_b4_forward_jvp_backward():
    model = create_model().to(os.getenv("SMOKE_DEVICE", "cpu"))
    images = torch.randn(1, 3, 32, 32, device=next(model.parameters()).device)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.002)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.002)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.002)
    loss, mse = meanflow_loss(model, model, images, torch.ones(1, 6, device=images.device))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(mse)
    assert torch.isfinite(model.blocks[0].attn.qkv.weight.grad).all()
    assert torch.count_nonzero(model.blocks[0].attn.qkv.weight.grad) > 0


def _full_batch_reference(model, images, labels, noise, r, t):
    """The original objective: one JVP over the whole batch, then the trainable forward."""
    shape = (-1, 1, 1, 1)
    z = (1 - t.view(shape)) * images + t.view(shape) * noise
    velocity = noise - images
    with torch.no_grad():
        _, derivative = torch.func.jvp(lambda a, b, c: model(a, b, c, augment_labels=labels),
                                       (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
    target = velocity - (t - r).view(shape) * derivative
    error = (model(z, r, t, augment_labels=labels) - target).square().flatten(1).sum(1)
    return (error * (error.detach() + 0.001).pow(-0.75)).mean()


def test_jvp_on_mean_flow_samples_only_matches_full_batch_jvp():
    torch.manual_seed(5)
    model = create_model("DiT-Tiny/4")
    with torch.no_grad():  # Nonzero outputs so the JVP term matters.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.05)
    images, noise, labels = torch.randn(6, 3, 32, 32), torch.randn(6, 3, 32, 32), torch.randn(6, 6)
    t = torch.tensor([0.9, 0.5, 0.7, 0.3, 0.95, 0.6])
    r = torch.tensor([0.9, 0.1, 0.7, 0.0, 0.2, 0.6])  # three flow-matching samples (r = t)
    reference = _full_batch_reference(model, images, labels, noise, r, t)
    reference.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    loss, _ = meanflow_loss(model, model, images, labels, noise=noise, times=(r, t))
    loss.backward()
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], rtol=1e-4, atol=1e-6)


def test_all_flow_matching_batch_skips_the_jvp():
    model = AffineVelocity()
    t = torch.tensor([0.4, 0.8])
    meanflow_loss(model, model, torch.zeros(2, 1, 1, 2), noise=torch.ones(2, 1, 1, 2), times=(t, t))
    assert model.calls == [True], "r = t everywhere: only the trainable forward runs"


@pytest.mark.parametrize("jvp_amp", [False, True])
def test_mixed_precision_loss_tracks_fp32(jvp_amp):
    # bf16 on CPU exercises the autocast paths; FP16 itself is checked on the GPU.
    torch.manual_seed(9)
    model = create_model("DiT-Tiny/4")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.05)
    images, noise = torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32)
    r, t = torch.full((8,), 0.2), torch.linspace(0.3, 1.0, 8)
    exact, _ = meanflow_loss(model, model, images, noise=noise, times=(r, t))
    mixed, _ = meanflow_loss(model, model, images, noise=noise, times=(r, t),
                             amp_dtype=torch.bfloat16, jvp_amp=jvp_amp)
    assert mixed.dtype == torch.float32 and torch.isfinite(mixed)
    torch.testing.assert_close(mixed, exact, rtol=2e-2, atol=0)
    mixed.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_overflowing_reduced_precision_jvp_is_recomputed_in_fp32(monkeypatch):
    torch.manual_seed(11)
    model = create_model("DiT-Tiny/4")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.05)
    images, noise = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    r, t = torch.full((4,), 0.1), torch.tensor([0.4, 0.6, 0.8, 0.9])
    expected, _ = meanflow_loss(model, model, images, noise=noise, times=(r, t),
                                amp_dtype=torch.bfloat16, jvp_amp=False)
    real_jvp = torch.func.jvp

    def overflowing_jvp(*args, **kwargs):  # FP16-style overflow whenever autocast is active
        primal, tangent = real_jvp(*args, **kwargs)
        if torch.is_autocast_enabled("cpu"):
            tangent = tangent.clone()
            tangent[1] = float("inf")
        return primal, tangent

    monkeypatch.setattr(torch.func, "jvp", overflowing_jvp)
    stats = {}
    loss, _ = meanflow_loss(model, model, images, noise=noise, times=(r, t),
                            amp_dtype=torch.bfloat16, jvp_amp=True, diagnostics=stats)
    assert torch.isfinite(loss) and stats["jvp_fp32_fallback"] == 1
    # Sample 1 now uses the FP32 derivative, the others the bf16 one.
    torch.testing.assert_close(loss, expected, rtol=2e-2, atol=0)
