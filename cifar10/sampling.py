"""Unconditional MeanFlow integration in CIFAR-10 pixel space."""

import math

import torch


def sampling_times(num_steps: int = 1, times=None) -> tuple[float, ...]:
    """Validate or construct the exact integration schedule, including endpoints."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if times is None:
        return tuple(1.0 - index / num_steps for index in range(num_steps + 1))
    try:
        raw = tuple(times)
        if any(isinstance(value, bool) for value in raw):
            raise ValueError("boolean time")
        schedule = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("times must be a sequence of finite numbers") from error
    if len(schedule) != num_steps + 1:
        raise ValueError("len(times) must equal num_steps + 1")
    if not all(math.isfinite(value) for value in schedule):
        raise ValueError("times must be finite")
    if schedule[0] != 1.0 or schedule[-1] != 0.0:
        raise ValueError("times must start at exactly 1 and end at exactly 0")
    if any(current <= following for current, following in zip(schedule, schedule[1:])):
        raise ValueError("times must be strictly descending")
    return schedule


@torch.no_grad()
def meanflow_sample(model, noise: torch.Tensor, num_steps: int = 1, times=None) -> torch.Tensor:
    """Integrate from noise at t=1 to data at r=0, without augmentation."""
    schedule = sampling_times(num_steps, times)
    if noise.ndim != 4 or noise.shape[1] != 3 or not noise.is_floating_point():
        raise ValueError("noise must be a floating-point NCHW RGB tensor")
    samples = noise
    for current, following in zip(schedule, schedule[1:]):
        t = noise.new_full((noise.shape[0],), current)
        r = noise.new_full((noise.shape[0],), following)
        velocity = model(samples, r, t, augment_labels=None)
        samples = samples - (current - following) * velocity
    return samples


def sampling_nfe(num_steps=1, *, field="avg", solver="euler", expected_nfe=None):
    """Return the actual model-call count; Heun is a diagonal-field diagnostic.

    ``num_steps`` always counts integration intervals. Average velocities already
    represent an interval, so applying Heun to them is deliberately unsupported.
    """
    sampling_times(num_steps)
    if field not in ("avg", "diagonal") or solver not in ("euler", "heun"):
        raise ValueError("field must be avg or diagonal; solver must be euler or heun")
    if field == "avg" and solver != "euler":
        raise ValueError("Heun requires field=diagonal")
    nfe = num_steps * (2 if solver == "heun" else 1)
    if expected_nfe is not None:
        if isinstance(expected_nfe, bool) or not isinstance(expected_nfe, int) or expected_nfe < 1:
            raise ValueError("nfe must be a positive integer")
        if expected_nfe != nfe:
            raise ValueError(f"requested nfe={expected_nfe} disagrees with actual model calls {nfe}")
    return nfe


@torch.no_grad()
def flow_sample(model, noise, num_steps=1, times=None, *, field="avg", solver="euler", expected_nfe=None):
    """Sample the original average field or diagnose its instantaneous diagonal.

    The default delegates to the original sampler without changing its numeric
    operations. Diagonal Euler/Heun integrate ``u(z,t,t)`` in FP32, with no
    augmentation or intermediate clipping/quantization. Their FIDs are diagnostic
    results, not a substitute for original one-NFE MeanFlow generation.
    """
    sampling_nfe(num_steps, field=field, solver=solver, expected_nfe=expected_nfe)
    if field == "avg":
        return meanflow_sample(model, noise, num_steps, times)
    schedule = sampling_times(num_steps, times)
    if noise.ndim != 4 or noise.shape[0] < 1 or noise.shape[1] != 3 or noise.dtype != torch.float32:
        raise ValueError("diagonal sampling requires a nonempty FP32 NCHW RGB tensor")
    samples = noise
    with torch.autocast(device_type=noise.device.type, enabled=False):
        for current, following in zip(schedule, schedule[1:]):
            t = noise.new_full((noise.shape[0],), current)
            velocity = model(samples, t, t, augment_labels=None).float()
            predicted = samples + (following - current) * velocity
            if solver == "heun":
                next_t = noise.new_full((noise.shape[0],), following)
                next_velocity = model(predicted, next_t, next_t, augment_labels=None).float()
                samples = samples + (following - current) * (velocity + next_velocity) * 0.5
            else:
                samples = predicted
            if not torch.isfinite(samples).all():
                raise FloatingPointError("Diagonal integration produced nonfinite pixels")
    return samples


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Match the paper's uint8 conversion: clamp, scale, then truncate."""
    return ((images.float() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
