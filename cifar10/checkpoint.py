"""Complete, atomic checkpoints at optimizer-update boundaries."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from .training_state import capture_rng, restore_rng
from .optim import load_optimizer_state, optimizer_metadata


def checkpoint_path(path, output_dir):
    if path == "latest":
        candidates = sorted((Path(output_dir) / "checkpoints").glob("step-[0-9]*"))
        candidates = [p for p in candidates if (p / "complete").is_file()]
        if not candidates:
            raise FileNotFoundError("No complete checkpoints found")
        return candidates[-1]
    result = Path(path)
    if not (result / "complete").is_file():
        raise ValueError(f"Not a complete training checkpoint: {result}")
    return result


def save_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config, step, *,
                    scaler=None, extra_emas=None, runtime_state=None, dropout_generator=None,
                    time_generator=None, attempt_revision=0):
    if type(attempt_revision) is not int or attempt_revision < 0:
        raise ValueError("attempt_revision must be a nonnegative integer")
    checked_runtime = _runtime_state(runtime_state, step)
    if config.precision == "fp16" and (scaler is None or not scaler.is_enabled()):
        raise ValueError("FP16 checkpoints require an enabled scaler")
    if (config.attn_projection_dropout or config.mlp_output_dropout) and dropout_generator is None:
        raise ValueError("Dropout checkpoints require a per-rank dropout generator")
    parent = Path(config.output_dir) / "checkpoints"
    name = f"step-{step:08d}" + (f"-attempt-{attempt_revision:012d}" if attempt_revision else "")
    final = parent / name
    staging = parent / f".{name}.incomplete"
    if accelerator.is_main_process:
        parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"Refusing to replace checkpoint {final}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
    accelerator.wait_for_everyone()
    rank_state = {"rng": capture_rng(accelerator.device), "stream": stream.state_dict()}
    if dropout_generator is not None:
        rank_state["dropout_rng"] = dropout_generator.get_state()
    if getattr(config, "dedicated_time_rng", False):
        if time_generator is None:
            raise ValueError("Dedicated time sampling requires its checkpointed generator")
        rank_state["time_rng"] = time_generator.get_state()
    torch.save(rank_state, staging / f"rank-{accelerator.process_index}.pt")
    if accelerator.is_main_process:
        weights = {"model": accelerator.unwrap_model(model).state_dict(),
                   "ema": ema.state_dict(), "config": asdict(config), "step": step}
        weights.update({name: extra.state_dict() for name, extra in (extra_emas or {}).items()})
        torch.save(weights, staging / "weights.pt")
        training = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "world_size": accelerator.num_processes, "step": step, "config": asdict(config)}
        training["optimizer_metadata"] = optimizer_metadata(optimizer, accelerator.unwrap_model(model))
        training["runtime_state"] = checked_runtime
        if scaler is not None and scaler.is_enabled():
            training["scaler"] = scaler.state_dict()
        torch.save(training, staging / "training.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (staging / "complete").write_text(f"{step}\n", encoding="utf-8")
        staging.rename(final)
        complete = sorted(p for p in parent.glob("step-[0-9]*") if (p / "complete").is_file())
        rolling = [p for p in complete if not (p / "protected.json").is_file()]
        for old in rolling[:-config.checkpoints_to_keep]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()
    return final


# Settings added after checkpoints already existed; a missing key means this value.
LEGACY_DEFAULTS = {"cooldown_start": 0, "cooldown_end": 0, "precision": "fp32", "jvp_fp16": False,
                   "extra_emas": False, "init_from": "", "time_sampler": "v0",
                   "attn_projection_dropout": 0.0, "mlp_output_dropout": 0.0,
                   "dropout_mask_mode": "indexed", "optimizer": "adam", "weight_decay": 0.0,
                   "weight_decay_policy": "matrix_only"}
RESEARCH_DEFAULTS = {"train_policy": "constant", "curriculum_fm_steps": 20000,
    "curriculum_ramp_steps": 20000, "curriculum_weight_end": 80000,
    "curriculum_gap_lambda": 1.61914605, "endpoint_focus_probability": 0.0,
    "dedicated_time_rng": False, "time_rng_seed": 314159, "data_split": "full",
    "data_split_manifest": "", "data_split_sha256": "", "fork_from": "",
    "fork_reset_data": False, "fork_reason": "", "muon_momentum": 0.95,
    "muon_ns_steps": 5, "muon_ns_dtype": "fp16", "muon_rms_scale": 0.2}
LEGACY_DEFAULTS.update(RESEARCH_DEFAULTS)
LEGACY_DEFAULTS.update(optimizer_kernel="reference", metrics_every=1, memory_every=1,
                       deadline_utc="", stop_file="", gpu_memory_limit_gib=0.0, gpu_free_margin_gib=0.0)
COOLDOWN_KEYS = ("cooldown_start", "cooldown_end")


def _runtime_state(state, step):
    values = {"successful_updates": step, "attempted_updates": step,
              "skipped_attempts": 0, "consecutive_skips": 0, **(state or {})}
    if values["successful_updates"] != step:
        raise ValueError("Checkpoint runtime successful_updates disagrees with step")
    for key in ("successful_updates", "attempted_updates", "skipped_attempts", "consecutive_skips"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError(f"Invalid checkpoint runtime counter: {key}")
    if values["attempted_updates"] < values["skipped_attempts"]:
        raise ValueError("Checkpoint skips exceed attempted updates")
    if values["consecutive_skips"] > values["skipped_attempts"]:
        raise ValueError("Checkpoint consecutive skips exceed total skips")
    return values


def _validate_complete(path, weights, training):
    """Reject mixed/corrupt payloads before mutating model or optimizer state."""
    try:
        marker_step = int((path / "complete").read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint completion marker: {path}") from error
    if type(training.get("step")) is not int or training["step"] < 0:
        raise ValueError("Invalid training checkpoint step")
    if weights.get("step") != training["step"] or marker_step != training["step"]:
        raise ValueError("Checkpoint weights/training/complete steps disagree")
    if {**LEGACY_DEFAULTS, **weights["config"]} != {**LEGACY_DEFAULTS, **training["config"]}:
        raise ValueError("Checkpoint weights/training configurations disagree")
    world_size = training.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid checkpoint world size")
    for rank in range(world_size):
        if not (path / f"rank-{rank}.pt").is_file():
            raise ValueError(f"Checkpoint is missing rank-{rank}.pt")


def protect_checkpoint(path, reason="selected best"):
    """Protect a complete training checkpoint in addition to rolling retention."""
    path = checkpoint_path(path, None)
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True, mmap=True)
    training = torch.load(path / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    _validate_complete(path, weights, training)
    marker = path / "protected.json"
    if marker.exists():
        return path
    staging = path / ".protected.json.incomplete"
    staging.write_text(json.dumps({"step": training["step"], "reason": str(reason),
                                   "protected_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
                       encoding="utf-8")
    staging.replace(marker)
    return path


def load_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config, *,
                    scaler=None, extra_emas=None, runtime_state=None, dropout_generator=None,
                    time_generator=None, scientific_fork=False):
    path = checkpoint_path(config.resume, config.output_dir)
    training = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    if training["world_size"] != accelerator.num_processes:
        raise ValueError("Resume requires the same world size")
    mutable = {"resume", "max_train_steps", "output_dir", "num_workers", "log_every",
               "checkpointing_steps", "checkpoints_to_keep", "download",
               "diag_every", "plot_every", "divergence_loss", "init_from", "metrics_every", "memory_every",
               "deadline_utc", "stop_file", "gpu_memory_limit_gib", "gpu_free_margin_gib"}
    saved = {**LEGACY_DEFAULTS, **training["config"]}
    fork_mutable = {"endpoint_focus_probability", "data_split", "fork_from", "fork_reset_data",
                    "fork_reason"} if scientific_fork else set()
    if (scientific_fork and saved["train_policy"] == "fm_curriculum"
            and config.train_policy == "constant"
            and training["step"] >= saved["curriculum_weight_end"]):
        from types import SimpleNamespace
        from .objective import training_policy
        before = training_policy(SimpleNamespace(**saved), training["step"])
        after = training_policy(config, training["step"])
        equivalent = ("ratio", "adaptive_p", "error_reduction", "gap_weight_progress")
        if any(before[key] != after[key] for key in equivalent):
            raise ValueError("Completed-curriculum fork changed the MeanFlow loss policy")
        fork_mutable.add("train_policy")
    for key, value in asdict(config).items():
        if key not in mutable | fork_mutable and key not in COOLDOWN_KEYS and saved.get(key) != value:
            raise ValueError(f"Resume configuration mismatch: {key}")
    if scientific_fork:
        if not config.fork_reason:
            raise ValueError("Scientific forks require an explicit reason")
        if saved["data_split"] != config.data_split and not config.fork_reset_data:
            raise ValueError("A changed dataset partition requires an explicit stream reset")
    old = tuple(saved[key] for key in COOLDOWN_KEYS)
    new = (config.cooldown_start, config.cooldown_end)
    # A cooldown may be added or moved only while no update has used it, so the
    # resumed learning-rate history is identical to an uninterrupted run.
    started = [start for start, end in (old, new) if end and training["step"] > start]
    if old != new and started:
        raise ValueError(f"Cooldown settings cannot change after the cooldown started (checkpoint at update "
                         f"{training['step']}, saved cooldown {old[0]}-{old[1]}, requested {new[0]}-{new[1]})")
    if config.max_train_steps <= training["step"]:
        raise ValueError("max_train_steps must exceed the saved update count")
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    _validate_complete(path, weights, training)
    restored_runtime = _runtime_state(training.get("runtime_state"), training["step"])
    rank = torch.load(path / f"rank-{accelerator.process_index}.pt",
                      map_location="cpu", weights_only=True)
    if saved["precision"] == "fp16":
        if "scaler" not in training or not training["scaler"]:
            raise ValueError("FP16 checkpoint has no saved scaler state")
        if scaler is None or not scaler.is_enabled():
            raise ValueError("FP16 resume requires an enabled scaler")
    if saved["attn_projection_dropout"] or saved["mlp_output_dropout"]:
        if "dropout_rng" not in rank:
            raise ValueError("Dropout checkpoint has no per-rank dropout RNG state")
        if dropout_generator is None:
            raise ValueError("Dropout resume requires a per-rank dropout generator")
    if saved["dedicated_time_rng"]:
        if time_generator is None or "time_rng" not in rank:
            raise ValueError("Dedicated time sampling checkpoint has no generator state")
    accelerator.unwrap_model(model).load_state_dict(weights["model"], strict=True)
    ema.load_state_dict(weights["ema"], strict=True)
    for name, extra in (extra_emas or {}).items():
        extra.load_state_dict(weights.get(name, weights["ema"]), strict=True)
    load_optimizer_state(optimizer, training["optimizer"], accelerator.unwrap_model(model), config,
                         training.get("optimizer_metadata"))
    if len(training["scheduler"]["base_lrs"]) != len(optimizer.param_groups):
        raise ValueError("Checkpoint scheduler group count does not match optimizer")
    scheduler.load_state_dict(training["scheduler"])
    if scaler is not None and scaler.is_enabled() and "scaler" in training:
        scaler.load_state_dict(training["scaler"])
    # Recompute the next update's rate with this run's schedule (identical unless a
    # not-yet-started cooldown was just added).
    for group, base, factor in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base * factor(scheduler.last_epoch)
    if dropout_generator is not None:
        if "dropout_rng" in rank:
            dropout_generator.set_state(rank["dropout_rng"])
    if time_generator is not None and "time_rng" in rank:
        time_generator.set_state(rank["time_rng"])
    if runtime_state is not None:
        runtime_state.clear()
        runtime_state.update(restored_runtime)
    if not (scientific_fork and config.fork_reset_data):
        stream.load_state_dict(rank["stream"])
    restore_rng(rank["rng"], accelerator.device)
    # Checkpoints contain no partial gradients; resume at an accumulation boundary.
    accelerator.step = training["step"] * config.gradient_accumulation_steps
    return training["step"]


def fork_from_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config, **kwargs):
    """Explicit policy/data fork; retain complete states, never call it exact resume."""
    from dataclasses import replace
    source_config = replace(config, resume=config.fork_from)
    return load_checkpoint(accelerator, model, ema, optimizer, scheduler, stream,
                           source_config, scientific_fork=True, **kwargs)


def init_from_checkpoint(accelerator, model, ema, optimizer, scheduler, config, *, extra_emas=None):
    """Warm-start a new run from another run's complete checkpoint.

    Loads the online weights, EMA (which also seeds any extra EMAs), Adam moments,
    and the update count, so the learning-rate schedule and EMA rampup continue by
    absolute update. This run's own optimizer settings, schedule, data stream, and
    RNG are used, so its recipe may differ (e.g. precision). Only the architecture
    must match.
    """
    path = checkpoint_path(config.init_from, None)
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    training = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    _validate_complete(path, weights, training)
    for key in ("model", "augment_dim"):
        if weights["config"].get(key) != getattr(config, key):
            raise ValueError(f"init_from checkpoint has a different {key}: {weights['config'].get(key)!r}")
    step = int(training["step"])
    if config.max_train_steps <= step:
        raise ValueError(f"max_train_steps must exceed the init_from update count {step}")
    accelerator.unwrap_model(model).load_state_dict(weights["model"], strict=True)
    ema.load_state_dict(weights["ema"], strict=True)
    for name, extra in (extra_emas or {}).items():
        extra.load_state_dict(weights.get(name, weights["ema"]), strict=True)
    load_optimizer_state(optimizer, training["optimizer"], accelerator.unwrap_model(model), config,
                         training.get("optimizer_metadata"), warm_start=True)
    scheduler.base_lrs = [config.learning_rate for _ in optimizer.param_groups]
    scheduler.last_epoch = step
    for group, base, factor in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base * factor(step)
    accelerator.step = step * config.gradient_accumulation_steps
    return step
