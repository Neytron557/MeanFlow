"""Training-health diagnostics and the divergence guard; consumes no RNG."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch


class TrainingDiverged(RuntimeError):
    """Raised on every rank together; the marker file is the contract with the controller."""


@contextmanager
def diagnostic_context(model, device):
    """Restore every module's mode and CPU/device RNG, even when a probe raises."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    modes = [(module, module.training) for module in model.modules()]
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                yield
        finally:
            for module, training in modes:
                module.training = training


@torch.no_grad()
def attention_logit_max(model, z, r, t, augment_labels=None):
    """Return max |q.k|/sqrt(d) per transformer block for one no-grad forward.

    The failed baseline run blew up through attention-logit growth (35-47 while
    still healthy, ~7e4 after divergence), so this tracks the mechanism directly.
    """
    values, hooks = [], []
    for block in model.blocks:
        attention = block.attn

        def hook(module, inputs, output, attention=attention):
            batch, tokens, _ = output.shape
            qkv = output.reshape(batch, tokens, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
            q, k = attention.q_norm(qkv[0]), attention.k_norm(qkv[1])
            values.append(((q * attention.scale) @ k.transpose(-2, -1)).abs().amax().item())

        hooks.append(attention.qkv.register_forward_hook(hook))
    try:
        with diagnostic_context(model, z.device):
            model(z, r, t, augment_labels=augment_labels)
    finally:
        for handle in hooks:
            handle.remove()
    return values


@dataclass(frozen=True)
class FixedProbe:
    """Training images and independent, immutable diagnostic inputs; not a holdout."""

    images: torch.Tensor
    noise: torch.Tensor
    r: torch.Tensor
    t: torch.Tensor
    augment_labels: torch.Tensor
    indices: tuple[int, ...]
    seed: int


def build_fixed_probe(dataset, device, *, count=8, augment_dim=6, seed=271828):
    """Choose fixed training rows and time bins without touching training RNG."""
    if count < 1 or len(dataset) < 1 or augment_dim < 0:
        raise ValueError("probe count and dataset size must be positive; augment_dim must be nonnegative")
    count = min(count, len(dataset))
    indices = tuple(index * len(dataset) // count for index in range(count))
    # Dataset loading should be deterministic, but preserve CPU RNG even if a
    # test dataset or future preprocessing accidentally draws from it.
    with torch.random.fork_rng(devices=[]):
        images = torch.stack([dataset[index][0] for index in indices]).to(device=device, dtype=torch.float32)
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("fixed probes require preprocessed CIFAR-10 RGB32 training images")
    noise = torch.randn(images.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(device)
    r_bins = (0.0, 0.125, 0.25, 0.375, 0.0, 0.25, 0.5, 0.75)
    t_bins = (0.125, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 1.0)
    r = torch.tensor([r_bins[index % len(r_bins)] for index in range(count)], device=device)
    t = torch.tensor([t_bins[index % len(t_bins)] for index in range(count)], device=device)
    labels = torch.zeros(count, augment_dim, dtype=torch.float32, device=device)
    return FixedProbe(images, noise, r, t, labels, indices, seed)


def _summary(values):
    """JSON-safe summaries; quantile population is explicitly finite tensor entries."""
    values = values.detach().float().flatten()
    finite = values[torch.isfinite(values)].double()
    result = {"count": values.numel(), "nonfinite_count": values.numel() - finite.numel()}
    if not finite.numel():
        return {**result, "mean": None, "rms": None, "abs_max": None,
                "abs_p50": None, "abs_p90": None, "abs_p99": None}
    quantiles = torch.quantile(finite.abs(), finite.new_tensor([0.5, 0.9, 0.99])).tolist()
    return {**result, "mean": finite.mean().item(), "rms": finite.square().mean().sqrt().item(),
            "abs_max": finite.abs().max().item(),
            **dict(zip(("abs_p50", "abs_p90", "abs_p99"), quantiles))}


def _attention_and_modulation(model, z, r, t, labels):
    """Observe one FP32 forward; remove hooks before forward-mode transforms."""
    blocks, handles = [{} for _ in model.blocks], []
    for index, block in enumerate(model.blocks):
        attention = block.attn

        def attention_hook(module, inputs, output, *, index=index, attention=attention):
            batch, tokens, _ = output.shape
            qkv = output.float().reshape(batch, tokens, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
            q, k = attention.q_norm(qkv[0]), attention.k_norm(qkv[1])
            logits = (q * attention.scale) @ k.transpose(-2, -1)
            centered = logits - logits.mean(dim=-1, keepdim=True)
            log_probabilities = logits.log_softmax(dim=-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
            width = output.shape[-1] // 3
            weights = module.weight.detach().double()
            blocks[index].update(
                block=index, centered_logits=_summary(centered), attention_entropy=_summary(entropy),
                q_activation=_summary(q), k_activation=_summary(k),
                q_weight_frobenius=weights[:width].norm().item(),
                k_weight_frobenius=weights[width:2 * width].norm().item(),
            )

        def modulation_hook(module, inputs, output, *, index=index):
            names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
            blocks[index]["adaln"] = {name: _summary(value) for name, value in zip(names, output.chunk(6, dim=-1))}

        handles.append(attention.qkv.register_forward_hook(attention_hook))
        handles.append(block.adaLN_modulation.register_forward_hook(modulation_hook))
    try:
        prediction = model(z, r, t, augment_labels=labels).float()
    finally:
        for handle in handles:
            handle.remove()
    return prediction, blocks


@torch.no_grad()
def run_fixed_probe(model, probe: FixedProbe, *, comparison_dtype=torch.float16):
    """Return detached numeric health diagnostics on fixed, unaugmented training rows.

    Reduced-precision comparisons report raw nonfinite derivatives instead of
    applying the production FP32 fallback, so rising precision stress is visible.
    Legacy CPU probes compute FP32; explicit BF16 comparison also works on CPU.
    """
    images, r, t = probe.images, probe.r, probe.t
    velocity = probe.noise - images
    gap = (t - r).view(-1, 1, 1, 1)
    z = (1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * probe.noise

    def derivative(dtype=None):
        def field(z_value, r_value, t_value):
            return model(z_value, r_value, t_value, augment_labels=probe.augment_labels)

        with torch.autocast(device_type=images.device.type, enabled=dtype is not None, dtype=dtype):
            _, partial = torch.func.jvp(
                field, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t))
            )
        return partial.detach().float()

    with diagnostic_context(model, images.device):
        prediction, blocks = _attention_and_modulation(model, z, r, t, probe.augment_labels)
        partial = derivative()
        correction = gap * partial
        target = velocity - correction
        correction_summary = _summary(correction)
        result = {
            "dataset_role": "training_diagnostic_not_holdout", "indices": list(probe.indices),
            "noise_seed": probe.seed, "r": r.tolist(), "t": t.tolist(),
            "augmentation_conditioning": "zero", "model_mode": "eval",
            "percentile_population": "absolute finite tensor entries; per-image RMS where explicitly named",
            "blocks": blocks, "correction": correction_summary,
            "correction_abs_p99": correction_summary["abs_p99"],
            "correction_per_image_rms": _summary(correction.flatten(1).square().mean(1).sqrt()),
            "fm_error": _summary(prediction - velocity),
            "fm_mse_by_image": [(value if math.isfinite(value) else None)
                                for value in (prediction - velocity).flatten(1).square().mean(1).tolist()],
            "fp32_finite": bool(torch.isfinite(prediction).all() & torch.isfinite(partial).all() & torch.isfinite(target).all()),
            "fp16_finite": None, "bf16_finite": None, "precision_target_rms_error": None,
            "precision_derivative_rms_error": None,
            "precision_comparison": "unavailable_on_cpu",
        }
        if comparison_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("comparison_dtype must be fp16, bf16, or None")
        if comparison_dtype is not None and (images.device.type == "cuda" or comparison_dtype == torch.bfloat16):
            name = "bf16" if comparison_dtype == torch.bfloat16 else "fp16"
            partial_mixed = derivative(comparison_dtype)
            target_mixed = velocity - gap * partial_mixed
            derivative_error = _summary(partial_mixed - partial)
            target_error = _summary(target_mixed - target)
            result.update(
                precision_comparison=f"{name}_vs_fp32",
                precision_derivative_error=derivative_error, precision_target_error=target_error,
                precision_derivative_rms_error=derivative_error["rms"],
                precision_target_rms_error=target_error["rms"],
            )
            result[f"{name}_finite"] = bool(torch.isfinite(partial_mixed).all() & torch.isfinite(target_mixed).all())
            target_rms = _summary(target)["rms"]
            result["precision_target_relative_rms_error"] = (target_error["rms"] / target_rms
                if target_error["rms"] is not None and target_rms else None)
        return result


RESEARCH_TIME_PAIRS = (
    (0.0, 0.0), (0.05, 0.05), (0.2, 0.2), (0.5, 0.5),
    (0.8, 0.8), (0.95, 0.95), (0.99, 0.99), (1.0, 1.0),
    (0.0, 0.05), (0.0, 0.2), (0.0, 0.5), (0.25, 0.5),
    (0.0, 0.8), (0.5, 0.8), (0.0, 0.95), (0.0, 0.99), (0.0, 1.0),
)


def build_time_bin_probes(dataset, device, *, count=32, seed=271828, augment_dim=6,
                          pairs=RESEARCH_TIME_PAIRS):
    """Shared fixed training rows/noise across diagonal and long-interval bins."""
    base = build_fixed_probe(dataset, device, count=count, augment_dim=augment_dim, seed=seed)
    probes = []
    for r_value, t_value in pairs:
        if not (math.isfinite(r_value) and math.isfinite(t_value) and 0 <= r_value <= t_value <= 1):
            raise ValueError("diagnostic time pairs require 0 <= r <= t <= 1")
        probes.append(FixedProbe(
            base.images, base.noise, torch.full_like(base.r, r_value),
            torch.full_like(base.t, t_value), base.augment_labels, base.indices, base.seed,
        ))
    return tuple(probes)


def training_mean_image(dataset, *, indices=None, horizontal_flip=True):
    """Mean only the declared training rows; never silently use a held-out split.

    CIFAR's uint8 array is averaged directly in float64. A Subset's indices are
    resolved before averaging. The symmetric mean accounts for the unconditional
    base horizontal flip; it does not estimate an augmented nonzero-label field.
    """
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if not selected or any(isinstance(i, bool) or not isinstance(i, int) or
                           not 0 <= i < len(dataset) for i in selected):
        raise ValueError("training mean requires valid, nonempty training indices")
    if len(set(selected)) != len(selected):
        raise ValueError("training mean indices must be distinct")
    base = dataset
    while isinstance(base, torch.utils.data.Subset):
        selected = [int(base.indices[i]) for i in selected]
        base = base.dataset
    array = getattr(base, "data", None)
    if array is not None and getattr(array, "ndim", 0) == 4:
        import numpy as np

        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape[1:] != (32, 32, 3):
            raise ValueError("raw CIFAR training mean expects uint8 NHWC RGB32 data")
        values = array if selected == list(range(len(base))) else array[selected]
        mean = torch.from_numpy(values.mean(axis=0, dtype=np.float64)).permute(2, 0, 1)
        mean = (mean / 127.5 - 1).float()
    else:
        with torch.random.fork_rng(devices=[]):
            total = torch.zeros(3, 32, 32, dtype=torch.float64)
            for index in selected:
                image = base[index][0]
                if image.dtype != torch.float32 or image.shape != (3, 32, 32):
                    raise ValueError("training mean expects preprocessed FP32 RGB32 rows")
                total += image.double().cpu()
            mean = (total / len(selected)).float()
    if horizontal_flip:
        mean = (mean + mean.flip(-1)) * 0.5
    return mean


def _safe_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


def adaptive_error_summary(per_image_error, powers=(0.0, 0.5, 0.75, 1.0), *, eps=0.001):
    """Original SUM-pixel adaptive weights and output-gradient signal concentration."""
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")
    error = per_image_error.detach().double().flatten()
    finite = error[torch.isfinite(error) & (error >= 0)]
    summaries = []
    for power in powers:
        if not math.isfinite(power) or power < 0:
            raise ValueError("adaptive powers must be finite and nonnegative")
        weights = (finite + eps).pow(-power)
        total = weights.sum()
        valid = weights.numel() > 0 and bool(total > 0) and bool(torch.isfinite(total))
        summaries.append({
            "p": power, "finite_rows": finite.numel(), "excluded_rows": error.numel() - finite.numel(),
            "effective_sample_size": _safe_number(total.square() / weights.square().sum()) if valid else None,
            "max_normalized_weight": _safe_number(weights.max() / total) if valid else None,
            "weight": _summary(weights),
            "adaptive_loss": _safe_number((finite * weights).mean()) if valid else None,
            "per_row_output_gradient_norm": _summary(2 * finite.sqrt() * weights / max(1, error.numel())),
        })
    return summaries


@torch.no_grad()
def run_time_bin_probe(model, probe, *, check_fp16=True, powers=(0.0, 0.5, 0.75, 1.0),
                       comparison_dtype=torch.float16):
    """Conditional-target diagnostics, not an estimate of exact marginal error."""
    images, r, t = probe.images, probe.r, probe.t
    velocity = probe.noise - images
    gap = (t - r).view(-1, 1, 1, 1)
    z = (1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * probe.noise

    def evaluate(dtype=None):
        def field(z_value, r_value, t_value):
            return model(z_value, r_value, t_value, augment_labels=probe.augment_labels)

        with torch.autocast(device_type=images.device.type, enabled=dtype is not None, dtype=dtype):
            if bool((r == t).all()):
                prediction = field(z, r, t)
                partial = torch.zeros_like(prediction)
            else:
                prediction, partial = torch.func.jvp(
                    field, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)),
                )
        return prediction.float(), partial.float()

    with diagnostic_context(model, images.device):
        prediction, partial = evaluate()
        correction = gap * partial
        target = velocity - correction
        residual = prediction - target
        error = residual.square().flatten(1).sum(1)
        result = {
            "r": r.tolist(), "t": t.tolist(), "count": len(images), "indices": list(probe.indices),
            "noise_seed": probe.seed, "model_mode": "eval", "augmentation_conditioning": "zero",
            "target_role": "original_conditional_velocity_and_model_derivative_not_exact_marginal_truth",
            "prediction": _summary(prediction), "target": _summary(target),
            "derivative": _summary(partial), "correction": _summary(correction),
            "residual": _summary(residual), "per_image_mse": [
                _safe_number(value) for value in error / residual[0].numel()],
            "adaptive_weights": adaptive_error_summary(error, powers),
            "fp32_finite": bool(torch.isfinite(prediction).all() & torch.isfinite(partial).all()
                                & torch.isfinite(target).all() & torch.isfinite(residual).all()),
            "fp16_finite": None, "precision_comparison": "unavailable_on_cpu",
        }
        if comparison_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("comparison_dtype must be fp16, bf16, or None")
        if check_fp16 and comparison_dtype is not None and (images.device.type == "cuda" or comparison_dtype == torch.bfloat16):
            name = "bf16" if comparison_dtype == torch.bfloat16 else "fp16"
            prediction16, partial16 = evaluate(comparison_dtype)
            target16 = velocity - gap * partial16
            target_error = _summary(target16 - target)
            prediction_error = _summary(prediction16 - prediction)
            derivative_error = _summary(partial16 - partial)
            target_rms = result["target"]["rms"]
            result.update(
                precision_comparison=f"raw_{name}_vs_fp32_no_fallback",
                precision_target_error=target_error, precision_prediction_error=prediction_error,
                precision_derivative_error=derivative_error,
                precision_target_relative_rms_error=(target_error["rms"] / target_rms
                    if target_error["rms"] is not None and target_rms else None),
            )
            result[f"{name}_finite"] = bool(torch.isfinite(prediction16).all() & torch.isfinite(partial16).all()
                                           & torch.isfinite(target16).all())
        return result


@torch.no_grad()
def run_endpoint_mean_probe(model, noise, training_mean, *, conditional_images=None,
                            powers=(0.0, 0.5, 0.75, 1.0)):
    """Exact t=1 mean velocity; mean tangent is a diagnostic, never a training change.

    At t=1, z=epsilon is independent of x, so E[v|z]=epsilon-E[x]. JVP is linear
    in its tangent. Error-weighted conditional expectations need not equal the
    unweighted mean identity. Optional repeated-input rows quantify that issue.
    """
    if noise.ndim != 4 or not len(noise) or noise.dtype != torch.float32 or noise.shape[1:] != (3, 32, 32):
        raise ValueError("endpoint probe requires FP32 BCHW RGB32 noises")
    if training_mean.shape != noise.shape[1:] or not torch.isfinite(training_mean).all():
        raise ValueError("endpoint probe requires a finite RGB32 declared-training mean")
    mean = training_mean.to(noise.device, dtype=torch.float32)
    velocity = noise - mean
    r, t = noise.new_zeros(len(noise)), noise.new_ones(len(noise))

    def field(z_value, r_value, t_value):
        return model(z_value, r_value, t_value, augment_labels=None)

    with diagnostic_context(model, noise.device):
        diagonal = field(noise, t, t).float() - velocity
        prediction, partial = torch.func.jvp(field, (noise, r, t),
                                            (velocity, torch.zeros_like(t), torch.ones_like(t)))
        residual = prediction.float() - (velocity - partial.float())
        result = {
            "time": 1.0, "augmentation_conditioning": "zero", "diagnostic_only": True,
            "truth": "epsilon_minus_declared_training_mean_at_exact_noise_endpoint",
            "diagonal_marginal_error": _summary(diagonal), "mean_identity_residual": _summary(residual),
            "mean_derivative": _summary(partial),
            "fp32_finite": bool(torch.isfinite(diagonal).all() & torch.isfinite(residual).all()
                                & torch.isfinite(partial).all()),
        }
        if conditional_images is not None:
            images = conditional_images.to(noise.device, dtype=torch.float32)
            if images.ndim != 4 or images.shape[1:] != noise.shape[1:] or not len(images):
                raise ValueError("conditional endpoint images must be nonempty RGB32")
            count = len(images)
            repeated = noise.repeat_interleave(count, dim=0)
            conditional_velocity = repeated - images.repeat((len(noise), 1, 1, 1))
            cr, ct = r.repeat_interleave(count), t.repeat_interleave(count)
            cp, cd = torch.func.jvp(field, (repeated, cr, ct),
                                   (conditional_velocity, torch.zeros_like(ct), torch.ones_like(ct)))
            rows = (cp.float() - (conditional_velocity - cd.float())).reshape(len(noise), count, -1)
            analytic = residual.flatten(1)
            sample_mean = rows.mean(1)
            errors = rows.double().square().sum(-1)
            finite = bool(torch.isfinite(rows).all() & torch.isfinite(analytic).all())
            weight_results = []
            if finite:
                for power in powers:
                    weights = (errors + 0.001).pow(-power)
                    normalized = weights / weights.sum(1, keepdim=True)
                    weighted = (normalized[:, :, None] * rows.double()).sum(1)
                    weight_results.append({
                        "p": power, "mean_residual": _summary(weighted),
                        "error_vs_analytic_mean": _summary(weighted - analytic),
                        "cosine_vs_analytic_per_noise": [
                            _safe_number(value) for value in torch.nn.functional.cosine_similarity(
                                weighted, analytic.double(), dim=1)],
                        "cosine_vs_unweighted_sample_per_noise": [
                            _safe_number(value) for value in torch.nn.functional.cosine_similarity(
                                weighted, sample_mean.double(), dim=1)],
                        "ess_per_noise": (weights.sum(1).square() / weights.square().sum(1)).tolist(),
                        "max_normalized_weight_per_noise": normalized.max(1).values.tolist(),
                    })
            result["conditional_weighting"] = {
                "images_per_noise": count, "noise_count": len(noise), "finite": finite,
                "nonfinite_counts": {"prediction": int((~torch.isfinite(cp)).sum()),
                                     "derivative": int((~torch.isfinite(cd)).sum()),
                                     "residual": int((~torch.isfinite(rows)).sum())},
                "conditional_residual": _summary(rows),
                "unweighted_sample_mean_error_vs_analytic": _summary(sample_mean - analytic),
                "powers": weight_results,
                "interpretation": "finite_sample_output_gradient_direction_not_parameter_gradient_or_population_bias",
            }
            result["fp32_finite"] = result["fp32_finite"] and finite
        return result


def flow_gradient_alignment(model, probe, *, count=4, adaptive_p=0.75):
    """Bounded optional FM/MF parameter-gradient comparison; optimizer/gradients untouched."""
    from .objective import meanflow_loss

    if not 1 <= count <= 8:
        raise ValueError("gradient alignment count must lie in [1,8]")
    parameters = [(name, p) for name, p in model.named_parameters() if name != "pos_embed"]
    if not parameters:
        return {"available": False, "reason": "model has no trainable diagnostic parameters"}
    flags = [p.requires_grad for _, p in parameters]
    selected = slice(0, min(count, len(probe.images)))
    try:
        for _, parameter in parameters:
            parameter.requires_grad_(True)
        with diagnostic_context(model, probe.images.device), torch.enable_grad():
            gradients = []
            for r in (probe.t[selected], probe.r[selected]):
                loss, _ = meanflow_loss(
                    model, model, probe.images[selected], probe.augment_labels[selected],
                    noise=probe.noise[selected], times=(r, probe.t[selected]), adaptive_p=adaptive_p,
                )
                gradients.append(torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True))
            fm_norm, mf_norm, dot = 0.0, 0.0, 0.0
            for fm, mf in zip(*gradients):
                if fm is not None:
                    fm_norm += float(fm.float().square().sum(dtype=torch.float64))
                if mf is not None:
                    mf_norm += float(mf.float().square().sum(dtype=torch.float64))
                if fm is not None and mf is not None:
                    dot += float((fm.float() * mf.float()).sum(dtype=torch.float64))
            denominator = math.sqrt(fm_norm * mf_norm) if fm_norm >= 0 and mf_norm >= 0 else 0
            return {"available": True, "count": min(count, len(probe.images)), "adaptive_p": adaptive_p,
                    "fm_gradient_norm": _safe_number(math.sqrt(fm_norm)),
                    "mf_gradient_norm": _safe_number(math.sqrt(mf_norm)),
                    "parameter_gradient_cosine": _safe_number(dot / denominator) if denominator else None,
                    "parameter_count": len(parameters), "dropout_mode": "eval_identity"}
    finally:
        for (_, parameter), flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def run_research_diagnostics(model, probes, training_mean, *, check_fp16=True,
                             endpoint_images=16, endpoint_noises=4, gradient_alignment=False,
                             deadline=None):
    """Bounded collection; the caller owns wall-clock/process supervision."""
    if not probes or min(endpoint_images, endpoint_noises) < 1:
        raise ValueError("research diagnostics require probes and positive endpoint counts")
    rows = []
    for probe in probes:
        if deadline is not None and time.monotonic() >= deadline:
            return {"status": "budget_limited", "time_bins": rows}
        rows.append(run_time_bin_probe(model, probe, check_fp16=check_fp16))
    if deadline is not None and time.monotonic() >= deadline:
        return {"status": "budget_limited", "time_bins": rows}
    base = probes[0]
    endpoint = run_endpoint_mean_probe(
        model, base.noise[:endpoint_noises], training_mean,
        conditional_images=base.images[:endpoint_images],
    )
    result = {"status": "completed", "time_bins": rows, "exact_endpoint": endpoint}
    if gradient_alignment and (deadline is None or time.monotonic() < deadline):
        interval = next((p for p in probes if bool((p.r < p.t).all())), None)
        if interval is not None:
            result["gradient_alignment"] = flow_gradient_alignment(model, interval)
    result["fp32_finite"] = all(row["fp32_finite"] for row in rows) and endpoint["fp32_finite"]
    return result


class LossGuard:
    """Trip when the rolling mean loss approaches the zero-output level.

    A zero-output model scores loss ~7.87 on this objective and a healthy run
    ~5.5, so an absolute threshold needs no run-specific state. The window is
    rebuilt after each restart; inputs are all-reduced, so every rank decides
    identically and no collective can hang.
    """

    def __init__(self, threshold, window=50, after_step=1000):
        self.threshold, self.after_step = threshold, after_step
        self.losses = deque(maxlen=window)

    def update(self, step, loss):
        # Only losses after the exemption period enter the window.
        if self.threshold <= 0 or step <= self.after_step:
            return None
        self.losses.append(loss)
        if len(self.losses) < self.losses.maxlen:
            return None
        mean = sum(self.losses) / len(self.losses)
        if mean > self.threshold:
            return f"rolling-{self.losses.maxlen} loss {mean:.4f} exceeds divergence_loss {self.threshold}"
        return None


def write_divergence_marker(output_dir, step, reason, last_checkpoint_step):
    path = Path(output_dir) / "divergence.json"
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps({
        "step": step, "reason": reason, "last_checkpoint_step": last_checkpoint_step,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    staging.replace(path)
    return path
