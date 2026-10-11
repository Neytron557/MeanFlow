"""Collective operational stops, checked only at optimizer transaction boundaries."""

from datetime import datetime, timezone
from pathlib import Path
import signal
import threading

import torch


def parse_deadline(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError("deadline_utc must be an ISO timestamp with a timezone") from error
    if parsed.tzinfo is None:
        raise ValueError("deadline_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def configure_gpu_memory(config, device):
    """Bound this process's allocator and preserve currently free peer headroom.

    The cap is at most the requested GiB and initial free memory minus the margin.
    CUDA contexts/NCCL and other processes are outside the allocator cap; the
    collective runtime check also monitors physical free memory after updates.
    """
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    limit, margin = config.gpu_memory_limit_gib, config.gpu_free_margin_gib
    if not limit and not margin:
        return None
    if free <= margin * 2**30:
        raise RuntimeError("GPU free memory is already below the configured safety margin")
    cap = min(limit * 2**30 if limit else total, free - margin * 2**30)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    return {"allocator_limit_gib": cap / 2**30, "initial_free_gib": free / 2**30,
            "free_margin_gib": margin}


class TrainingControl:
    """Signals set a local flag; every rank makes the same boundary decision."""

    REASONS = ("signal", "stop_file", "deadline", "gpu_free_margin")

    def __init__(self, config):
        self.config = config
        self.deadline = parse_deadline(config.deadline_utc)
        self.signal_received = None
        self.previous_handlers = {}

    def _signal(self, signum, frame):
        self.signal_received = signum

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                self.previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal)
        return self

    def __exit__(self, *exception):
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)

    def local_flags(self, device, *, now=None):
        memory_stop = False
        if device.type == "cuda" and self.config.gpu_free_margin_gib:
            free, _ = torch.cuda.mem_get_info(device)
            memory_stop = free < self.config.gpu_free_margin_gib * 2**30
        return [self.signal_received is not None,
                bool(self.config.stop_file and Path(self.config.stop_file).exists()),
                bool(self.deadline and (now or datetime.now(timezone.utc)) >= self.deadline),
                memory_stop]

    def reason(self, accelerator):
        # Never branch on a rank-local flag before the collective.
        flags = torch.tensor(self.local_flags(accelerator.device), dtype=torch.int32,
                             device=accelerator.device)
        reduced = accelerator.reduce(flags, reduction="sum").tolist()
        active = [name for name, value in zip(self.REASONS, reduced) if value]
        return ",".join(active) if active else None
