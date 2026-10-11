"""MeanFlow objective with a detached forward-mode JVP target.

References: https://arxiv.org/abs/2505.13447 and the authors' implementation at
https://github.com/Gsunshine/py-meanflow/tree/main/meanflow/models.
"""

import math

import torch
from torch import Tensor, nn

from .dropout import indexed_dropout_rows


def training_policy(config, successful_age: int) -> dict:
    """Resolve a recipe for the next attempted update from successful update age.

    Skipped attempts must reuse this policy. The trainer supplies the age of the
    scientific run, rather than resetting it at checkpoint/segment boundaries.
    ``gap_weight_progress`` is the paper's s: one at the start, zero at the end.
    """
    if type(successful_age) is not int or successful_age < 0:
        raise ValueError("successful_age must be a nonnegative integer")
    policy = {
        "phase": "endpoint_focus" if config.endpoint_focus_probability else "constant",
        "ratio": config.ratio,
        "adaptive_p": config.adaptive_p,
        "error_reduction": "sum",
        "gap_weight_progress": 0.0,
        "gap_weight_lambda": config.curriculum_gap_lambda,
        "focus_probability": config.endpoint_focus_probability,
    }
    if config.train_policy == "constant":
        return policy
    if config.train_policy != "fm_curriculum":
        raise ValueError("Unknown train_policy")
    start = config.curriculum_fm_steps
    ramp_end = start + config.curriculum_ramp_steps
    if successful_age < start:
        policy.update(phase="fm_pretrain", ratio=0.0, adaptive_p=0.0,
                      error_reduction="mean", focus_probability=0.0)
    else:
        policy.update(
            phase="mf_transition" if successful_age < ramp_end else "mf_full",
            ratio=config.ratio * min((successful_age - start) / config.curriculum_ramp_steps, 1.0),
            gap_weight_progress=max(0.0, 1.0 - (successful_age - start) /
                                    (config.curriculum_weight_end - start)),
            focus_probability=0.0,
        )
    return policy


def sample_times(
    batch_size: int,
    device: torch.device | str,
    *,
    time_mu: float = -2.0,
    time_sigma: float = 2.0,
    ratio: float = 0.75,
    time_sampler: str = "v0",
    generator: torch.Generator | None = None,
    focus_probability: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return (r, t) using the original v0 or official fixed-parameter v1 sampler.

    Two independent samples are sorted before replacing r by t with probability
    1-ratio in v0. V1 uses separate logit-normals and clamps only r; its t
    marginal is never sorted upward. time_mu/time_sigma configure v0 only.
    A dedicated generator consumes two normal and four uniform batch vectors
    under every policy, so changing the equal-time ratio or focus does not
    change the subsequent time random stream. Without a generator and focus,
    the legacy global-RNG sampling remains exactly unchanged.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not math.isfinite(time_mu) or not math.isfinite(time_sigma) or time_sigma < 0:
        raise ValueError("time_mu and time_sigma must be finite, with time_sigma nonnegative.")
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must lie in [0, 1].")
    if time_sampler not in ("v0", "v1"):
        raise ValueError("time_sampler must be 'v0' or 'v1'.")
    if not math.isfinite(focus_probability) or not 0 <= focus_probability <= 1:
        raise ValueError("focus_probability must be finite and in [0, 1].")
    first = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    second = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    equal_coin = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    if time_sampler == "v1":
        t = torch.sigmoid(first * 1.6 - 0.6)
        r = torch.sigmoid(second * 1.6 - 4.0)
        r = torch.minimum(torch.where(equal_coin < 1 - ratio, t, r), t)
    else:
        first = torch.sigmoid(first * time_sigma + time_mu)
        second = torch.sigmoid(second * time_sigma + time_mu)
        r, t = torch.minimum(first, second), torch.maximum(first, second)
        r = torch.where(equal_coin < 1 - ratio, t, r)
    if generator is not None or focus_probability:
        focus_coin = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        focus_t = 0.8 + 0.2 * torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        focus_r = 0.2 * torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        # Preserve both the explicit FM rows and additional v1 clamp-equal rows.
        focused = (r < t) & (focus_coin < focus_probability)
        r, t = torch.where(focused, focus_r, r), torch.where(focused, focus_t, t)
    return r, t


def meanflow_loss(
    model: nn.Module,
    unwrapped_model: nn.Module,
    images: Tensor,
    augment_labels: Tensor | None = None,
    *,
    time_mu: float = -2.0,
    time_sigma: float = 2.0,
    ratio: float = 0.75,
    time_sampler: str = "v0",
    adaptive_p: float = 0.75,
    norm_eps: float = 0.001,
    time_generator: torch.Generator | None = None,
    error_reduction: str = "sum",
    gap_weight_progress: float = 0.0,
    gap_weight_lambda: float = 1.61914605,
    focus_probability: float = 0.0,
    noise: Tensor | None = None,
    times: tuple[Tensor, Tensor] | None = None,
    diagnostics: dict | None = None,
    amp_dtype: torch.dtype | None = None,
    jvp_amp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return the scalar adaptive loss and detached mean pixel squared error.

    ``model`` is the trainable module (possibly DDP). ``unwrapped_model`` is its
    underlying module and is used only for the target JVP, before the trainable
    forward. Forward-mode AD remains enabled under ``no_grad``; the target has
    no parameter-gradient graph. ``noise`` and ``times=(r,t)`` allow deterministic
    numerical checks. All model parameters and input images must be FP32.
    ``amp_dtype`` (e.g. ``torch.float16``) runs the trainable forward under autocast,
    and the JVP too when ``jvp_amp``; the target, adaptive weights and loss stay FP32.
    The JVP is evaluated only for samples with r < t: for r = t its term is
    multiplied by t - r = 0, so the target is the velocity itself.
    ``error_reduction='mean', adaptive_p=0`` gives ordinary per-pixel MSE for
    the diagonal FM preparation. Otherwise the original pixel-sum adaptive
    loss is retained. Progressive gap weights multiply after adaptive weighting
    on positive-gap rows only, with a single whole-batch mean and no group
    renormalization. The conditional JVP tangent and target remain unchanged.
    When ``diagnostics`` is a dict it receives detached ``per_image_mse``,
    ``per_image_loss`` (the adaptive-weighted terms whose mean is the loss), ``r``,
    ``t`` and ``z``, plus ``per_image_gap_weight`` and ``actual_mf_fraction``;
    the returned values are unaffected.
    """
    if images.ndim != 4 or images.shape[0] == 0 or images.dtype != torch.float32:
        raise ValueError("images must be a nonempty FP32 BCHW tensor.")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive.")
    if not math.isfinite(adaptive_p) or adaptive_p < 0:
        raise ValueError("adaptive_p must be finite and nonnegative.")
    if error_reduction not in ("sum", "mean"):
        raise ValueError("error_reduction must be sum or mean.")
    if not math.isfinite(gap_weight_progress) or not 0 <= gap_weight_progress <= 1:
        raise ValueError("gap_weight_progress must be finite and in [0, 1].")
    if not math.isfinite(gap_weight_lambda) or gap_weight_lambda <= 0:
        raise ValueError("gap_weight_lambda must be finite and positive.")
    if not math.isfinite(focus_probability) or not 0 <= focus_probability <= 1:
        raise ValueError("focus_probability must be finite and in [0, 1].")
    noise = torch.randn_like(images) if noise is None else noise
    if noise.shape != images.shape or noise.dtype != images.dtype or noise.device != images.device:
        raise ValueError("noise must have the same shape, dtype, and device as images.")
    r, t = times if times is not None else sample_times(
        images.shape[0], images.device, time_mu=time_mu, time_sigma=time_sigma, ratio=ratio,
        time_sampler=time_sampler, generator=time_generator, focus_probability=focus_probability,
    )
    if any(time.shape != (images.shape[0],) or time.device != images.device or time.dtype != torch.float32 for time in (r, t)):
        raise ValueError("r and t must be FP32 batch-length vectors on the images device.")
    time_shape = (-1, 1, 1, 1)
    z = (1 - t.view(time_shape)) * images + t.view(time_shape) * noise
    velocity = noise - images

    device_type = images.device.type

    def autocast(enabled):
        if enabled and amp_dtype is not None:
            return torch.autocast(device_type=device_type, dtype=amp_dtype)
        return torch.autocast(device_type=device_type, enabled=False)

    def time_derivative(rows, reduced_precision):
        labels = None if augment_labels is None else augment_labels[rows]

        def velocity_field(z_value, r_value, t_value):
            return unwrapped_model(z_value, r_value, t_value, augment_labels=labels)

        # Select from original-microbatch masks before entering transformed code.
        # The fallback passes original indices too, so it sees the same function.
        with indexed_dropout_rows(unwrapped_model, rows), torch.no_grad(), autocast(reduced_precision):
            _, partial = torch.func.jvp(
                velocity_field, (z[rows], r[rows], t[rows]),
                (velocity[rows], torch.zeros_like(r[rows]), torch.ones_like(t[rows])),
            )
        return partial.float()

    derivative = torch.zeros_like(images)
    index = (r != t).nonzero(as_tuple=True)[0]
    fallback = 0
    if index.numel():
        partial = time_derivative(index, jvp_amp)
        if jvp_amp and amp_dtype is not None:
            # A reduced-precision tangent can exceed FP16's range on rare samples;
            # recompute exactly those in FP32 rather than training on inf/NaN targets.
            overflow = ~torch.isfinite(partial).flatten(1).all(1)
            if overflow.any():
                fallback = int(overflow.sum())
                partial[overflow] = time_derivative(index[overflow], False)
        derivative[index] = partial
    target = (velocity - (t - r).view(time_shape) * derivative).detach()
    with autocast(True):
        prediction = model(z, r, t, augment_labels=augment_labels)
    with autocast(False):
        squared_error = (prediction.float() - target).square()
        flattened = squared_error.flatten(1)
        per_image_error = flattened.sum(1) if error_reduction == "sum" else flattened.mean(1)
        weights = (per_image_error.detach() + norm_eps).pow(-adaptive_p)
        gap_weights = torch.where(r < t, 1 - gap_weight_progress + gap_weight_lambda *
                                  gap_weight_progress * (1 - (t - r)), torch.ones_like(t))
        per_image_loss = per_image_error * weights * gap_weights
        loss = per_image_loss.mean()
    if diagnostics is not None:
        diagnostics.update(jvp_fp32_fallback=fallback,
                           target_finite=torch.isfinite(target).all(),
                           per_image_mse=squared_error.detach().flatten(1).mean(1),
                           per_image_loss=per_image_loss.detach(),
                           per_image_gap_weight=gap_weights.detach(),
                           actual_mf_fraction=(r < t).float().mean().detach(),
                           r=r.detach(), t=t.detach(), z=z.detach())
    return loss, squared_error.detach().mean()
