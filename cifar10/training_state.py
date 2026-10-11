"""Deterministic data continuation, RNG snapshots, and EMA updates."""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


def make_time_generator(device, seed: int = 314159, rank: int = 0):
    """Dedicated per-rank time RNG, independent of noise and augmentation draws."""
    if type(seed) is not int or type(rank) is not int or seed < 0 or rank < 0 or seed + rank >= 2**63:
        raise ValueError("Time RNG seed and rank must be nonnegative integers fitting signed int64")
    return torch.Generator(device=torch.device(device)).manual_seed(seed + rank)


def capture_time_rng(generator: torch.Generator) -> torch.Tensor:
    """Return a CPU byte state suitable for a per-rank checkpoint."""
    return generator.get_state().clone()


def restore_time_rng(generator: torch.Generator, state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("Time RNG state must be a one-dimensional uint8 tensor")
    generator.set_state(state.cpu())


class BatchStream:
    """Full per-rank microbatches; accumulation can span shuffled epochs."""

    def __init__(self, dataset, batch_size, *, rank, world_size, seed, num_workers):
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                          shuffle=True, seed=seed, drop_last=True)
        self.generator = torch.Generator().manual_seed(seed + rank)
        self.loader = DataLoader(dataset, batch_size=batch_size, sampler=self.sampler,
                                 num_workers=num_workers, drop_last=True,
                                 pin_memory=torch.cuda.is_available(), generator=self.generator)
        if not len(self.loader):
            raise ValueError("Dataset is too small for a full batch on every process")
        self.epoch = 0
        self.offset = 0
        self.iterator = None
        self.reconstructing = False

    def _start(self):
        self.sampler.set_epoch(self.epoch)
        generator_state = self.generator.get_state() if self.reconstructing else None
        self.iterator = iter(self.loader)
        if generator_state is not None:
            # Iterator creation must not advance the saved generator twice.
            self.generator.set_state(generator_state)
            self.reconstructing = False
        for _ in range(self.offset):
            next(self.iterator)

    def next(self):
        if self.offset == len(self.loader):
            self.epoch += 1
            self.offset = 0
            self.iterator = None
            self.reconstructing = False
        if self.iterator is None:
            self._start()
        value = next(self.iterator)
        self.offset += 1
        return value

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["epoch"] < 0 or not 0 <= state["offset"] <= len(self.loader):
            raise ValueError("Invalid saved data position")
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.generator.set_state(state["generator"])
        self.iterator = None
        self.reconstructing = self.offset > 0


def capture_rng(device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("Checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"], device)


def lr_factor(update, warmup_steps, cooldown_start=0, cooldown_end=0):
    """Learning-rate multiplier for 1-based optimizer update ``update``.

    Linear warmup to 1, constant, then (when ``cooldown_end > 0``) a linear
    decay over updates ``cooldown_start + 1 .. cooldown_end``; the last update
    keeps a small nonzero rate. Before ``cooldown_start`` the factor is exactly
    that of a schedule without cooldown, so a cooldown can be added to a run
    that has not reached it yet.
    """
    factor = min(update / max(1, warmup_steps), 1.0)
    if cooldown_end > 0 and update > cooldown_start:
        factor *= max(0.0, (cooldown_end - update + 1) / (cooldown_end - cooldown_start + 1))
    return factor


# Extra EMA widths kept on rank 0 when ``extra_emas`` is set: name -> (rampup ratio, decay cap).
# The main EMA uses ``ema_rampup_ratio``/``ema_decay`` (half-life 5% of updates).
EXTRA_EMAS = {"ema_short": (0.025, 0.99995), "ema_long": (0.10, 0.99998)}


def ema_decay_at(update, decay, rampup_ratio):
    """EMA decay for 1-based optimizer update ``update`` with an EDM-style rampup.

    The half-life is ``rampup_ratio * update`` updates, capped by ``decay``, so the
    average never retains a large share of the random initialization. A pure
    function of the step keeps resume bitwise exact. ``rampup_ratio=0`` returns
    the fixed ``decay``.
    """
    if update < 1:
        raise ValueError("update must be a positive 1-based count")
    if rampup_ratio <= 0:
        return decay
    return min(decay, 0.5 ** (1 / (rampup_ratio * update)))


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)
