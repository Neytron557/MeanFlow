"""Numerical checks of MeanFlow accumulation and actual Gloo/DDP reduction."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from cifar10.model import create_model
from cifar10.objective import meanflow_loss


def _nonzero_model():
    torch.manual_seed(73)
    model = create_model("DiT-Tiny/4")
    # Zero-output initialization would hide missing time, attention, and
    # augmentation gradients. Exercise every trainable layer from update one.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(mean=0, std=0.03)
    return model


def _fixed_batch(update):
    generator = torch.Generator().manual_seed(101 + update)
    images = torch.rand(8, 3, 32, 32, generator=generator).mul(2).sub(1)
    noise = torch.randn(8, 3, 32, 32, generator=generator)
    labels = torch.randn(8, 6, generator=generator)
    r = torch.linspace(0.02, 0.4, 8)
    t = torch.linspace(0.1, 0.97, 8)
    # Include one ordinary flow-matching example and unequal-time examples.
    r[0] = t[0]
    return images, noise, labels, r, t


def _loss(model, unwrapped, batch, indices=slice(None)):
    images, noise, labels, r, t = (tensor[indices] for tensor in batch)
    return meanflow_loss(model, unwrapped, images, labels, noise=noise, times=(r, t))[0]


def _assert_gradients_match(actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        if not parameter.requires_grad:
            continue
        reference = expected_parameters[name]
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert reference.grad is not None, f"Missing reference gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        torch.testing.assert_close(
            parameter.grad, reference.grad, rtol=3e-4, atol=2e-7,
            msg=lambda message: f"Gradient mismatch for {name}: {message}",
        )


def _assert_parameters_match(actual, expected):
    for (name, parameter), (_, reference) in zip(actual.named_parameters(), expected.named_parameters()):
        torch.testing.assert_close(
            parameter, reference, rtol=3e-5, atol=2e-7,
            msg=lambda message: f"Parameter mismatch for {name}: {message}",
        )


def test_accumulated_meanflow_gradients_and_adam_updates_match_full_batch():
    torch.set_num_threads(1)
    full = _nonzero_model()
    accumulated = deepcopy(full)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
                  for model in (full, accumulated)]
    for update in range(2):
        batch = _fixed_batch(update)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        _loss(full, full, batch).backward()
        for start in range(0, 8, 2):
            (_loss(accumulated, accumulated, batch, slice(start, start + 2)) / 4).backward()
        _assert_gradients_match(accumulated, full)
        for optimizer in optimizers:
            optimizer.step()
        _assert_parameters_match(accumulated, full)


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        module = _nonzero_model()
        full = deepcopy(module)
        ddp = DistributedDataParallel(module, gradient_as_bucket_view=True)
        distributed_optimizer = torch.optim.Adam(ddp.parameters(), lr=1e-4, foreach=False)
        reference_optimizer = torch.optim.Adam(full.parameters(), lr=1e-4, foreach=False)
        for update in range(2):
            batch = _fixed_batch(update)
            distributed_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            _loss(full, full, batch).backward()
            for microbatch in range(2):
                # Rank 0 receives examples 0..3; rank 1 receives 4..7.
                start = rank * 4 + microbatch * 2
                with ddp.no_sync() if microbatch == 0 else nullcontext():
                    (_loss(ddp, module, batch, slice(start, start + 2)) / 2).backward()
                flat_gradients = torch.cat([parameter.grad.flatten() for parameter in module.parameters()
                                            if parameter.requires_grad])
                gathered = [torch.empty_like(flat_gradients) for _ in range(2)]
                dist.all_gather(gathered, flat_gradients)
                if microbatch == 0:
                    assert not torch.equal(gathered[0], gathered[1]), "no_sync unexpectedly reduced gradients"
                else:
                    assert torch.equal(gathered[0], gathered[1]), "DDP did not synchronize every gradient"
            _assert_gradients_match(module, full)
            distributed_optimizer.step()
            reference_optimizer.step()
            _assert_parameters_match(module, full)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_two_rank_ddp_accumulation_matches_unsplit_meanflow_updates(tmp_path):
    # A FileStore avoids torchrun rendezvous hostname resolution on macOS.
    mp.spawn(_ddp_worker, args=(str(tmp_path / "gloo-store"),), nprocs=2, join=True)
