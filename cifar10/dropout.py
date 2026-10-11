"""Training-only dropout with explicit masks indexed by original sample rows.

Mask contexts are process-local, temporary Python state, never model buffers.
The trainer keeps the full-microbatch context alive through backward; the
objective selects original rows *before* entering forward-mode AD. A separate
generator keeps these draws independent of augmentation, noise, and times.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import Tensor, nn


_MASKS: ContextVar[dict | None] = ContextVar("meanflow_indexed_dropout_masks", default=None)


class IndexedDropout(nn.Dropout):
    """Inverted output dropout requiring a full-microbatch mask context."""

    def __init__(self, p: float, sample_shape: tuple[int, ...]):
        if not math.isfinite(p) or not 0 <= p < 1:
            raise ValueError("Indexed dropout probability must be finite and in [0, 1).")
        super().__init__(p=p, inplace=False)
        self.sample_shape = tuple(sample_shape)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        context = _MASKS.get()
        if context is None or self not in context:
            raise RuntimeError("Training dropout requires indexed_dropout_context through backward.")
        _, mask = context[self]
        if mask.shape != x.shape or mask.device != x.device:
            raise ValueError("Dropout mask shape/device does not match the selected original rows.")
        # Boolean storage avoids retaining FP32 copies of all output masks.
        # Neither operation is in-place: forward AD must see the fixed mask.
        return (x * mask) / (1.0 - self.p)


def make_dropout_generator(device: torch.device | str, seed: int, rank: int = 0) -> torch.Generator:
    """Create the separately checkpointed per-rank MeanFlow dropout stream.

    The fixed stream salt and rank stride are part of same-run reproducibility.
    New arms and legacy checkpoints start this stream from the same base seed.
    """
    if rank < 0:
        raise ValueError("Dropout generator rank must be nonnegative.")
    dropout_seed = (int(seed) + 0x4D45414E464C4F57 + 104729 * rank) % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(dropout_seed)


def _active_sites(model: nn.Module) -> dict[str, IndexedDropout]:
    return {name: module for name, module in model.named_modules()
            if isinstance(module, IndexedDropout) and module.training and module.p > 0}


@contextmanager
def indexed_dropout_context(
    model: nn.Module,
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    masks: Mapping[str, Tensor] | None = None,
) -> Iterator[dict[str, Tensor]]:
    """Create/reuse full masks; scope must include loss construction and backward.

    ``masks`` accepts explicit boolean tensors keyed by module name for numerical
    references and fixed-mask accumulation tests. Otherwise ``generator`` is
    required when any training dropout is active. Evaluation/zero dropout draw
    nothing. Full masks are yielded for inspection and are never attached to the
    model, EMA, or state dict. Cleanup is guaranteed even after an exception.
    """
    if batch_size <= 0:
        raise ValueError("Dropout batch_size must be positive.")
    device = torch.device(device)
    sites = _active_sites(model)
    if masks is not None and set(masks) != set(sites):
        raise ValueError("Explicit dropout mask names must exactly match active dropout sites.")
    if sites and masks is None and generator is None:
        raise ValueError("Active dropout requires a dedicated generator or explicit masks.")
    full_masks = {}
    context = dict(_MASKS.get() or {})
    for name, module in sites.items():
        shape = (batch_size, *module.sample_shape)
        mask = (torch.rand(shape, device=device, dtype=torch.float32, generator=generator) >= module.p
                if masks is None else masks[name])
        # A bare 'cuda' resolves to the current device, as torch factory calls do.
        expected_device = torch.device("cuda", torch.cuda.current_device()) if device == torch.device("cuda") else device
        if mask.dtype != torch.bool or mask.shape != shape or mask.device != expected_device:
            raise ValueError(f"Dropout mask {name!r} must be boolean with shape {shape} on {expected_device}.")
        full_masks[name] = mask
        context[module] = (mask, mask)
    token = _MASKS.set(context)
    try:
        yield full_masks
    finally:
        _MASKS.reset(token)


@contextmanager
def indexed_dropout_rows(model: nn.Module, rows: Tensor) -> Iterator[None]:
    """Select masks by original full-microbatch indices outside transformed code.

    Nested scopes always index the original masks, including reordered subsets
    and FP32 fallback rows; they never re-index an already reduced mask tensor.
    """
    sites = _active_sites(model)
    if not sites:
        yield
        return
    current = _MASKS.get()
    if current is None or any(module not in current for module in sites.values()):
        raise RuntimeError("Training dropout requires indexed_dropout_context through backward.")
    if rows.ndim != 1 or rows.dtype != torch.long:
        raise ValueError("Dropout rows must be a one-dimensional original-index int64 tensor.")
    context = dict(current)
    for module in sites.values():
        full, _ = current[module]
        context[module] = (full, full.index_select(0, rows))
    token = _MASKS.set(context)
    try:
        yield
    finally:
        _MASKS.reset(token)
