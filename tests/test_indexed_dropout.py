"""Fixed-mask forward-AD, accumulation, and stream reproducibility contracts."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from cifar10.dropout import IndexedDropout, indexed_dropout_context, indexed_dropout_rows, make_dropout_generator
from cifar10.model import create_model
from cifar10.objective import meanflow_loss


class NonlinearDropoutField(nn.Module):
    """Small nonzero nonlinear fixture; both output masks affect its derivative."""

    def __init__(self, p=0.3):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.37))
        self.weight = nn.Parameter(torch.tensor(0.71))
        self.first = IndexedDropout(p, (1, 2, 3))
        self.second = IndexedDropout(p, (1, 2, 3))

    def forward(self, z, r, t, augment_labels=None):
        bias = t[:, None, None, None] + 0.2 * r[:, None, None, None]
        if augment_labels is not None:
            bias = bias + 0.1 * augment_labels[:, 0, None, None, None]
        value = self.first(torch.sin(self.scale * z + bias))
        return self.weight * self.second((value + 0.4 * z).square()) + 0.17 * z


def _batch(size=6):
    generator = torch.Generator().manual_seed(37)
    images = torch.randn(size, 1, 2, 3, generator=generator)
    noise = torch.randn(size, 1, 2, 3, generator=generator)
    labels = torch.randn(size, 6, generator=generator)
    t = torch.linspace(0.2, 0.9, size)
    r = 0.3 * t
    r[::2] = t[::2]
    return images, noise, labels, r, t


def _reference(model, batch):
    images, noise, labels, r, t = batch
    shape = (-1, 1, 1, 1)
    z = (1 - t.view(shape)) * images + t.view(shape) * noise
    velocity = noise - images
    with torch.no_grad():
        primal, derivative = torch.func.jvp(
            lambda a, b, c: model(a, b, c, augment_labels=labels),
            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)),
        )
    target = (velocity - (t - r).view(shape) * derivative).detach()
    prediction = model(z, r, t, augment_labels=labels)
    per_image = (prediction - target).square().flatten(1).sum(1)
    loss = (per_image * (per_image.detach() + 0.001).pow(-0.75)).mean()
    return loss, primal, derivative, target, prediction


def _objective(model, unwrapped, batch, **kwargs):
    images, noise, labels, r, t = batch
    return meanflow_loss(model, unwrapped, images, labels, noise=noise, times=(r, t), **kwargs)


@pytest.mark.parametrize("kind", ["mixed", "diagonal", "nondiagonal", "single"])
def test_subset_objective_equals_full_batch_fixed_mask_reference(kind):
    model = NonlinearDropoutField()
    batch = list(_batch(1 if kind == "single" else 6))
    if kind == "diagonal":
        batch[3] = batch[4].clone()
    elif kind in ("nondiagonal", "single"):
        batch[3] = batch[4] * 0.3
    generator = make_dropout_generator("cpu", 17)
    with indexed_dropout_context(model, batch_size=batch[0].shape[0], device="cpu", generator=generator):
        reference, _, _, target, prediction = _reference(model, batch)
        reference.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        diagnostics = {}
        loss, mse = _objective(model, model, batch, diagnostics=diagnostics)
        loss.backward()
        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(mse, (prediction - target).square().mean(), rtol=1e-4, atol=1e-6)
        assert diagnostics["target_finite"]
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6)


def test_reordered_subset_primal_derivative_and_targets_use_original_rows():
    model = NonlinearDropoutField()
    batch = _batch()
    images, noise, labels, r, t = batch
    z = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
    velocity = noise - images
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=make_dropout_generator("cpu", 31)):
        _, expected_primal, expected_derivative, expected_target, _ = _reference(model, batch)
        rows = torch.tensor([5, 1, 4])
        with indexed_dropout_rows(model, rows), torch.no_grad():
            primal, derivative = torch.func.jvp(
                lambda a, b, c: model(a, b, c, augment_labels=labels[rows]),
                (z[rows], r[rows], t[rows]),
                (velocity[rows], torch.zeros_like(r[rows]), torch.ones_like(t[rows])),
            )
            target = velocity[rows] - (t[rows] - r[rows])[:, None, None, None] * derivative
            torch.testing.assert_close(primal, expected_primal[rows], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(derivative, expected_derivative[rows], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(target, expected_target[rows], rtol=1e-4, atol=1e-6)
            # A nested fallback is also indexed against the original full batch.
            with indexed_dropout_rows(model, torch.tensor([4])):
                torch.testing.assert_close(model(z[4:5], r[4:5], t[4:5], labels[4:5]), expected_primal[4:5])


def test_fp32_fallback_reuses_noncontiguous_original_row_masks(monkeypatch):
    model = NonlinearDropoutField()
    batch = list(_batch())
    # JVP rows [1, 3, 5]; fallback must use original row 3, not row 1.
    original_jvp = torch.func.jvp
    calls = []

    def forced_overflow(*args, **kwargs):
        primal, tangent = original_jvp(*args, **kwargs)
        calls.append((primal.detach().clone(), tangent.detach().clone()))
        if torch.is_autocast_enabled("cpu"):
            tangent = tangent.clone()
            tangent[1] = float("inf")
        return primal, tangent

    generator = make_dropout_generator("cpu", 52)
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=generator):
        reference, primal, derivative, _, _ = _reference(model, batch)
        state_before = generator.get_state().clone()
        monkeypatch.setattr(torch.func, "jvp", forced_overflow)
        diagnostics = {}
        loss, _ = _objective(model, model, batch, amp_dtype=torch.bfloat16, jvp_amp=True, diagnostics=diagnostics)
        loss.backward()
        assert diagnostics["jvp_fp32_fallback"] == 1
        assert [call[0].shape[0] for call in calls] == [3, 1]
        torch.testing.assert_close(calls[1][0], primal[3:4])
        torch.testing.assert_close(calls[1][1], derivative[3:4])
        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)
        assert torch.equal(generator.get_state(), state_before)


def test_dropout_context_cleanup_and_missing_context_fail_fast():
    model = NonlinearDropoutField()
    batch = _batch()
    with pytest.raises(RuntimeError, match="indexed_dropout_context"):
        _objective(model, model, batch)
    with pytest.raises(RuntimeError, match="deliberate"):
        with indexed_dropout_context(model, batch_size=6, device="cpu", generator=make_dropout_generator("cpu", 2)):
            _objective(model, model, batch)[0].backward()
            raise RuntimeError("deliberate failure")
    with pytest.raises(RuntimeError, match="indexed_dropout_context"):
        _objective(model, model, batch)
    assert not list(model.buffers())
    assert set(model.state_dict()) == {"scale", "weight"}


def test_dedicated_stream_is_fresh_rank_distinct_and_exactly_resumable():
    model = NonlinearDropoutField()
    generator = make_dropout_generator("cpu", 8, rank=0)
    global_state = torch.get_rng_state().clone()
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=generator) as first:
        pass
    saved = generator.get_state().clone()
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=generator) as second:
        pass
    resumed = make_dropout_generator("cpu", 8, rank=0)
    resumed.set_state(saved)
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=resumed) as replay:
        pass
    other_rank = make_dropout_generator("cpu", 8, rank=1)
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=other_rank) as other:
        pass
    for name in first:
        assert not torch.equal(first[name], second[name])
        assert not torch.equal(first[name], other[name])
        assert torch.equal(second[name], replay[name])
        assert first[name].dtype == torch.bool
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize("p,evaluation", [(0.0, False), (0.3, True)])
def test_disabled_dropout_is_rng_free_identity(p, evaluation):
    model = NonlinearDropoutField(p=p)
    if evaluation:
        model.eval()
    generator = make_dropout_generator("cpu", 6)
    state = generator.get_state().clone()
    with indexed_dropout_context(model, batch_size=6, device="cpu", generator=generator) as masks:
        assert masks == {}
        loss, _ = _objective(model, model, _batch())
        loss.backward()
    assert torch.equal(generator.get_state(), state)
    assert model.first(torch.ones(2, 1, 2, 3)).equal(torch.ones(2, 1, 2, 3))


def test_transformer_instruments_only_output_sites_without_state_changes():
    torch.manual_seed(14)
    baseline = create_model("DiT-Tiny/4")
    regularized = create_model("DiT-Tiny/4", attn_projection_dropout=0.1, mlp_output_dropout=0.1)
    with torch.no_grad():
        for parameter in baseline.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.04)
    regularized.load_state_dict(baseline.state_dict(), strict=True)
    assert {name: value.shape for name, value in baseline.state_dict().items()} == {
        name: value.shape for name, value in regularized.state_dict().items()}
    active = {name for name, module in regularized.named_modules() if isinstance(module, nn.Dropout) and module.p}
    assert active == {f"blocks.{block}.{site}" for block in range(2) for site in ("attn.proj_drop", "mlp.drop2")}
    images = torch.randn(2, 3, 32, 32)
    r, t = torch.tensor([0.1, 0.5]), torch.tensor([0.6, 0.5])
    baseline.eval()
    regularized.eval()
    torch.testing.assert_close(regularized(images, r, t), baseline(images, r, t), rtol=0, atol=0)
    regularized.train()
    with indexed_dropout_context(regularized, batch_size=2, device="cpu", generator=make_dropout_generator("cpu", 5)):
        loss, _ = meanflow_loss(regularized, regularized, images, times=(r, t))
        loss.backward()
    assert regularized.blocks[0].attn.qkv.weight.grad.abs().max() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in regularized.parameters() if parameter.grad is not None)


def test_nonlinear_transformer_full_and_subset_jvp_gradients_match():
    torch.manual_seed(46)
    model = create_model("DiT-Tiny/4", attn_projection_dropout=0.1, mlp_output_dropout=0.1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.04)
    batch = (torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), torch.randn(4, 6),
             torch.tensor([0.7, 0.1, 0.4, 0.2]), torch.tensor([0.7, 0.8, 0.4, 0.9]))
    with indexed_dropout_context(model, batch_size=4, device="cpu", generator=make_dropout_generator("cpu", 46)):
        reference = _reference(model, batch)[0]
        reference.backward()
        expected = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                    if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        loss, _ = _objective(model, model, batch)
        loss.backward()
        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-6)


def test_zero_dropout_objective_matches_original_plain_dropout_sites():
    torch.manual_seed(9)
    model = create_model("DiT-Tiny/4")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.04)
    original = deepcopy(model)
    for block in original.blocks:
        block.attn.proj_drop = nn.Dropout(0)
        block.mlp.drop2 = nn.Dropout(0)
    batch = (torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 6),
             torch.tensor([0.4, 0.1]), torch.tensor([0.4, 0.8]))
    expected, expected_mse = _objective(original, original, batch)
    expected.backward()
    actual, actual_mse = _objective(model, model, batch)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_mse, expected_mse, rtol=0, atol=0)
    for parameter, expected_parameter in zip(model.parameters(), original.parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("p", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError, match="probability"):
        IndexedDropout(p, (3,))


def _dropout_ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        module = NonlinearDropoutField()
        reference = deepcopy(module)
        ddp = DistributedDataParallel(module)
        batch = _batch(8)
        with indexed_dropout_context(reference, batch_size=8, device="cpu", generator=make_dropout_generator("cpu", 12)) as masks:
            _reference(reference, batch)[0].backward()
        for microbatch in range(2):
            start = rank * 4 + microbatch * 2
            rows = slice(start, start + 2)
            local_batch = tuple(value[rows] for value in batch)
            local_masks = {name: mask[rows] for name, mask in masks.items()}
            with ddp.no_sync() if microbatch == 0 else nullcontext():
                with indexed_dropout_context(module, batch_size=2, device="cpu", masks=local_masks):
                    (_objective(ddp, module, local_batch)[0] / 2).backward()
        for parameter, expected in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_fixed_dropout_ddp_accumulation_matches_full_reference(tmp_path):
    mp.spawn(_dropout_ddp_worker, args=(str(tmp_path / "dropout-store"),), nprocs=2, join=True)
