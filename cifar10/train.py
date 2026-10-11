"""MeanFlow training with explicit precision, complete checkpoints, and runtime bounds."""

from copy import deepcopy
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, set_seed

from .augmentation import augment_batch
from .amp_update import optimizer_attempt
from .checkpoint import (checkpoint_path, fork_from_checkpoint, init_from_checkpoint,
                         load_checkpoint, save_checkpoint)
from .config import parse_config
from .data import SyntheticCIFAR10, load_cifar10, load_cifar10_partition
from .diagnostics import (LossGuard, TrainingDiverged, attention_logit_max, write_divergence_marker,
                          build_fixed_probe, run_fixed_probe)
from .model import create_model
from .dropout import indexed_dropout_context, make_dropout_generator
from .optim import create_optimizer
from .objective import meanflow_loss, training_policy
from .plots import LossHistory, save_loss_plots
from .records import recover_attempt_tail
from .runtime import TrainingControl, configure_gpu_memory
from .training_state import (EXTRA_EMAS, BatchStream, ema_decay_at, lr_factor, update_ema,
                             make_time_generator)


logger = logging.getLogger("cifar10.train")
LOG_FILE = "train.log"


def configure_logging(output_dir, rank):
    """Mirror this process's log to ``<output_dir>/train.log`` and the console.

    The file is appended across resumes. Rank 0 records progress (INFO); other
    ranks record only warnings and errors. Library warnings are captured too.
    """
    path = Path(output_dir) / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(f"%(asctime)s UTC | rank {rank} | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime  # Same clock as the JSONL timestamps.
    handlers = [logging.FileHandler(path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout if rank == 0 else sys.stderr)]
    root = logging.getLogger()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.getLogger("cifar10").setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logging.captureWarnings(True)
    return handlers


def close_logging(handlers):
    logging.captureWarnings(False)
    root = logging.getLogger()
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()


def format_progress(record, target):
    parts = [f"update {record['step']:,}/{target:,}", f"total loss {record['loss']:.4f}"]
    if record.get("loss_fm") is not None:
        parts.append(f"loss 1 (flow matching) {record['loss_fm']:.4f}")
    if record.get("loss_mf") is not None:
        parts.append(f"loss 2 (MeanFlow) {record['loss_mf']:.4f}")
    parts += [f"mse {record['mse']:.4f}", f"grad norm {record['grad_norm']:.3f}",
              f"lr {record['lr']:.2e}", f"EMA decay {record['ema_decay']:.6f}",
              f"{record['seconds_per_update']:.2f} s/update"]
    if record.get("attn_logit_max") is not None:
        parts.append(f"max attention logit {record['attn_logit_max']:.2f}")
    if record.get("peak_reserved_gib") is not None:
        parts.append(f"peak reserved {record['peak_reserved_gib']:.2f} GiB")
    return " | ".join(parts)


def make_accelerator(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu --smoke for local verification.")
    if config.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # FP16 matmuls accumulate in FP32 (no reduced-precision split-K reductions).
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return Accelerator(
        cpu=config.device == "cpu", mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps, sync_with_dataloader=False),
        # Fixed registration-order buckets are needed for bitwise CUDA resume.
        # With find_unused_parameters=False, DDP rebuilds buckets after startup;
        # a restart changes reduction grouping and introduces last-bit drift.
        # In PyTorch 2.7.1 this public option disables rebuilding (static_graph
        # must remain False). All trainable parameters are still used normally.
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True,
                                                       find_unused_parameters=True,
                                                       static_graph=False)],
    )


def validate_output(config):
    """Never append to or prune an unrelated run, or roll an existing run back."""
    output = Path(config.output_dir).resolve()
    # A log left by a launch that failed before training started does not make a run.
    if not output.exists() or not any(path.name != LOG_FILE for path in output.iterdir()):
        return
    if not config.resume:
        raise FileExistsError(f"Output directory is nonempty: {output}; use --resume or a new directory")
    selected = checkpoint_path(config.resume, config.output_dir).resolve()
    if selected.parent != output / "checkpoints" or not (output / "config.json").is_file():
        raise ValueError("Resume into a different run requires a new, empty output directory")
    latest = checkpoint_path("latest", config.output_dir).resolve()
    if selected != latest:
        raise ValueError("Resuming an older checkpoint requires a new, empty output directory")


def prepare_run(config, accelerator, step):
    output = Path(config.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        log = output / "train.jsonl"
        if config.resume and log.is_file():
            # A crash may leave logged updates after the last complete checkpoint.
            recover_log(log, step)
        if config.resume:
            recover_attempt_tail(output / "attempts.jsonl")
        resolved = asdict(config)
        resolved.update(world_size=accelerator.num_processes,
                        effective_batch_size=config.effective_batch_size(accelerator.num_processes),
                        torch_version=str(torch.__version__), python_version=platform.python_version(),
                        dataset="synthetic" if config.smoke else "cifar10")
        (output / "config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
        logger.info("%s with %d process(es); command: %s",
                    f"Resuming from update {step:,}" if config.resume else "Starting a new run",
                    accelerator.num_processes, " ".join(sys.argv))
        logger.info("Resolved configuration:\n%s", json.dumps(resolved, indent=2))
    accelerator.wait_for_everyone()


def recover_log(path, step):
    """Discard updates beyond a checkpoint and a crash-truncated trailing line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"Malformed interior training log line {index + 1}") from None
        if record["step"] <= step:
            retained.append(line)
    staging = path.with_suffix(".recovering")
    staging.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    staging.replace(path)


def build_training(config, accelerator):
    set_seed(config.seed)
    model = create_model(config.model, augment_dim=config.augment_dim,
                         attn_projection_dropout=config.attn_projection_dropout,
                         mlp_output_dropout=config.mlp_output_dropout,
                         dropout_mask_mode=config.dropout_mask_mode)
    optimizer = create_optimizer(model, config)
    # LambdaLR passes the number of completed updates; the factor is for the next one.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: lr_factor(completed + 1, config.warmup_steps,
                                               config.cooldown_start, config.cooldown_end))
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = deepcopy(accelerator.unwrap_model(model)).eval().requires_grad_(False)
    # Extra EMA widths live on rank 0 only; they are saved in weights.pt and chosen at evaluation.
    extra_emas = ({name: deepcopy(ema) for name in EXTRA_EMAS}
                  if config.extra_emas and accelerator.is_main_process else {})
    # Initialization is common to all ranks; training randomness is independent.
    set_seed(config.seed + accelerator.process_index)
    if config.smoke:
        dataset = SyntheticCIFAR10(size=max(128, config.per_device_batch_size * accelerator.num_processes * 3),
                                   seed=config.seed)
    else:
        if config.download:
            with accelerator.main_process_first():
                dataset = load_cifar10(config.data_dir, download=accelerator.is_main_process)
        else:
            dataset = load_cifar10(config.data_dir, download=False)
        if config.data_split_manifest:
            dataset = load_cifar10_partition(dataset, config.data_split_manifest,
                                             partition=config.data_split,
                                             expected_sha256=config.data_split_sha256)
    stream = BatchStream(dataset, config.per_device_batch_size,
                         rank=accelerator.process_index, world_size=accelerator.num_processes,
                         seed=config.seed, num_workers=config.num_workers)
    return model, ema, extra_emas, optimizer, scheduler, stream


AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def train(config):
    validate_output(config)
    rank = int(os.environ.get("RANK", "0"))
    handlers = configure_logging(config.output_dir, rank)
    try:
        with TrainingControl(config) as control:
            return run_training(config, control=control)
    except TrainingDiverged as error:
        if rank == 0:
            logger.error("Training diverged: %s. See divergence.json in the output directory.", error)
        raise
    except BaseException:
        logger.exception("Training failed")
        raise
    finally:
        close_logging(handlers)


def run_training(config, *, control=None):
    accelerator = make_accelerator(config)
    config.validate(accelerator.num_processes)
    control = control or TrainingControl(config)
    memory_error = None
    try:
        memory_limits = configure_gpu_memory(config, accelerator.device)
    except RuntimeError as error:
        memory_limits, memory_error = None, str(error)
    if (config.gpu_memory_limit_gib or config.gpu_free_margin_gib):
        valid = torch.tensor(int(memory_error is None), device=accelerator.device)
        if accelerator.reduce(valid, reduction="sum").item() != accelerator.num_processes:
            raise RuntimeError(memory_error or "A peer GPU failed the initial free-memory safety check")
    if memory_limits:
        logger.info("GPU allocator bounds: %s", json.dumps(memory_limits))
    model, ema, extra_emas, optimizer, scheduler, stream = build_training(config, accelerator)
    amp_dtype = AMP_DTYPES[config.precision]
    # FP16 gradients need loss scaling; the scaler is a no-op for fp32/bf16.
    scaler = torch.amp.GradScaler(accelerator.device.type, enabled=config.precision == "fp16")
    dropout_generator = make_dropout_generator(accelerator.device, config.seed, accelerator.process_index)
    time_generator = (make_time_generator(accelerator.device, config.time_rng_seed,
                                         accelerator.process_index)
                      if config.dedicated_time_rng else None)
    runtime_state = {"attempted_updates": 0, "skipped_attempts": 0,
                     "consecutive_skips": 0, "successful_updates": 0}
    if config.resume:
        step = load_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config,
                               scaler=scaler, extra_emas=extra_emas,
                               runtime_state=runtime_state, dropout_generator=dropout_generator,
                               time_generator=time_generator)
    elif config.fork_from:
        step = fork_from_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config,
                                    scaler=scaler, extra_emas=extra_emas,
                                    runtime_state=runtime_state, dropout_generator=dropout_generator,
                                    time_generator=time_generator)
    elif config.init_from:
        step = init_from_checkpoint(accelerator, model, ema, optimizer, scheduler, config,
                                    extra_emas=extra_emas)
    else:
        step = 0
    runtime_state["successful_updates"] = step
    prepare_run(config, accelerator, step)
    if config.init_from and not config.resume:
        logger.info("Warm start from %s at update %s", config.init_from, f"{step:,}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    last_saved = step if config.resume else -1
    last_saved_attempts = runtime_state["attempted_updates"] if config.resume else None
    # The initial zero-output loss (~7.87) is exempt until training has had time to leave it.
    guard = LossGuard(config.divergence_loss, after_step=min(1000, config.warmup_steps))
    guard.losses.extend(runtime_state.get("loss_guard_history", []))
    # Rank 0 keeps the loss history for the loss 1 / loss 2 / total plots.
    history = (LossHistory.from_jsonl(Path(config.output_dir) / "train.jsonl", max_step=step)
               if accelerator.is_main_process else None)
    fixed_probe = (build_fixed_probe(stream.loader.dataset, accelerator.device)
                   if accelerator.is_main_process else None)
    plotting = [accelerator.is_main_process and config.plot_every > 0]

    def write_plots():
        if not plotting[0] or not len(history):
            return
        try:
            save_loss_plots(config.output_dir, history, divergence_loss=config.divergence_loss)
        except ImportError as error:
            plotting[0] = False
            logger.warning("Loss plots disabled; install matplotlib (requirements-cifar10.txt): %s", error)
        except Exception as error:  # A plot must never stop training.
            logger.warning("Loss plots failed at update %d: %s", step, error)

    def diverge(reason):
        # Called on every rank with identical reduced values, so all ranks stop together.
        if accelerator.is_main_process:
            if last_record is not None and last_logged != step:
                with (Path(config.output_dir) / "train.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(last_record) + "\n")
                history.append(last_record)
            write_divergence_marker(config.output_dir, step, reason, last_saved if last_saved >= 0 else None)
            write_plots()
        raise TrainingDiverged(reason)

    window = torch.zeros(2, device=accelerator.device)
    # Per-image sums for r == t (loss 1, flow matching) and r < t (loss 2, MeanFlow):
    # [mse_fm, count_fm, mse_mf, count_mf, loss_fm, loss_mf, FP32 JVP fallbacks].
    parts = torch.zeros(7, device=accelerator.device)
    # Mixed precision only: a nonfinite loss skips its update instead of stopping the run.
    nonfinite_update = False
    max_norm = config.max_grad_norm if config.max_grad_norm > 0 else float("inf")
    grad_norm = torch.zeros((), device=accelerator.device)
    skipped_in_a_row = runtime_state["consecutive_skips"]
    attempts_path = Path(config.output_dir) / "attempts.jsonl"
    recent_skips = deque(runtime_state.get("recent_skips", []), maxlen=2000)
    started = time.perf_counter()
    microbatches = 0
    stop_reason = None
    last_record = None
    last_logged = None
    while step < config.max_train_steps:
        if microbatches == 0:
            stop_reason = control.reason(accelerator)
            if stop_reason:
                break
        next_step = step + 1
        record_due = (next_step == 1 or next_step == config.max_train_steps or
                      any(every and next_step % every == 0 for every in
                          (config.metrics_every, config.log_every, config.diag_every,
                           config.plot_every, config.checkpointing_steps,
                           config.memory_every if accelerator.device.type == "cuda" else 0)))
        policy = training_policy(config, step)
        if runtime_state.get("loss_guard_phase") != policy["phase"]:
            guard.losses.clear()
            runtime_state["loss_guard_phase"] = policy["phase"]
        images = stream.next()[0].to(accelerator.device, non_blocking=True)
        images, labels = augment_batch(images, enabled=config.augmentation)
        with accelerator.accumulate(model), indexed_dropout_context(
                accelerator.unwrap_model(model), batch_size=images.shape[0],
                device=images.device, generator=dropout_generator):
            stats = {}
            loss, mse = meanflow_loss(model, accelerator.unwrap_model(model), images, labels,
                                      time_mu=config.time_mu, time_sigma=config.time_sigma,
                                      ratio=policy["ratio"], adaptive_p=policy["adaptive_p"],
                                      norm_eps=config.norm_eps, diagnostics=stats,
                                      amp_dtype=amp_dtype, jvp_amp=config.jvp_fp16,
                                      time_sampler=config.time_sampler,
                                      time_generator=time_generator,
                                      error_reduction=policy["error_reduction"],
                                      gap_weight_progress=policy["gap_weight_progress"],
                                      gap_weight_lambda=policy["gap_weight_lambda"],
                                      focus_probability=policy["focus_probability"])
            target_finite = stats.get("target_finite", torch.tensor(True, device=images.device))
            target_ranks = accelerator.reduce(target_finite.to(torch.int32), reduction="sum")
            if target_ranks.item() != accelerator.num_processes:
                diverge(f"Nonfinite FP32 MeanFlow reference target after update {step}")
            finite = accelerator.reduce(torch.isfinite(loss).to(torch.int32), reduction="sum")
            loss_finite = finite.item() == accelerator.num_processes
            if not loss_finite:
                if amp_dtype is None:
                    diverge(f"Nonfinite MeanFlow loss after update {step}")
                # A reduced-precision forward overflowed on this batch. The backward still runs
                # (DDP must synchronize), and the whole update is skipped below on every rank.
                nonfinite_update = True
                if accelerator.is_main_process:
                    logger.warning("Nonfinite %s loss after update %d; skipping this update",
                                   config.precision, step)
            accelerator.backward(scaler.scale(loss))
            parts[6] += stats["jvp_fp32_fallback"]
            if loss_finite:
                window += torch.stack([loss.detach(), mse]) / config.gradient_accumulation_steps
                if record_due:
                    flow = stats["r"] == stats["t"]
                    per_mse, per_loss = stats["per_image_mse"], stats["per_image_loss"]
                    parts[:6] += torch.stack([per_mse[flow].sum(), flow.sum().float(), per_mse[~flow].sum(),
                                              (~flow).sum().float(), per_loss[flow].sum(), per_loss[~flow].sum()])
            amp_skipped = False
            if accelerator.sync_gradients:
                committed, grad_norm = optimizer_attempt(
                    model, optimizer, scaler, accelerator,
                    nonfinite_loss=nonfinite_update, max_norm=max_norm)
                amp_skipped = not committed
            optimizer.zero_grad(set_to_none=True)
        microbatches += 1
        if not accelerator.sync_gradients:
            continue
        microbatches = 0
        skipped_in_a_row = skipped_in_a_row + 1 if amp_skipped else 0
        runtime_state["attempted_updates"] += 1
        runtime_state["skipped_attempts"] += int(amp_skipped)
        runtime_state["consecutive_skips"] = skipped_in_a_row
        runtime_state["successful_updates"] = step + int(not amp_skipped)
        recent_skips.append(int(amp_skipped))
        runtime_state["recent_skips"] = list(recent_skips)
        skipped_parts = accelerator.reduce(parts, reduction="sum").tolist() if amp_skipped else None
        if accelerator.is_main_process:
            with attempts_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({**{k: v for k, v in runtime_state.items()
                                        if k not in ("recent_skips", "loss_guard_history")},
                    "step": step + int(not amp_skipped),
                    "committed": not amp_skipped, "nonfinite_loss": nonfinite_update,
                    "jvp_fp32_fallback": int(skipped_parts[6]) if skipped_parts else 0,
                    "amp_scale": scaler.get_scale(),
                    "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                    "seconds": time.perf_counter() - started,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        if amp_skipped and amp_dtype is None:
            diverge(f"Nonfinite gradients or norm after update {step}")
        if skipped_in_a_row >= 50:
            diverge(f"{config.precision.upper()} gradients overflowed on 50 consecutive updates (scale {scaler.get_scale():g})")
        if len(recent_skips) == 2000 and sum(recent_skips) >= 20:
            diverge("At least 1% skipped attempts in the last 2000; fixed-input FP32 diagnosis required")
        if amp_skipped:
            window.zero_()
            parts.zero_()
            nonfinite_update = False
            started = time.perf_counter()
            continue
        used_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        ema_decay = ema_decay_at(step + 1, config.ema_decay, config.ema_rampup_ratio)
        update_ema(ema, accelerator.unwrap_model(model), ema_decay)
        for name, extra in extra_emas.items():
            ratio, cap = EXTRA_EMAS[name]
            update_ema(extra, accelerator.unwrap_model(model), ema_decay_at(step + 1, cap, ratio))
        step += 1
        values = accelerator.reduce(window, reduction="mean")
        window.zero_()
        mse_fm, count_fm, mse_mf, count_mf, loss_fm, loss_mf, fallbacks = (
            accelerator.reduce(parts, reduction="sum").tolist() if record_due else [0] * 7)
        parts.zero_()
        skipped_nonfinite, nonfinite_update = nonfinite_update, False
        # total loss == fm_fraction * loss_fm + (1 - fm_fraction) * loss_mf
        record = {"step": step, **{k: v for k, v in runtime_state.items()
                                   if k not in ("recent_skips", "loss_guard_history")},
                  "loss": values[0].item(),
                  "training_phase": policy["phase"], "nominal_mf_probability": policy["ratio"],
                  "adaptive_p": policy["adaptive_p"], "error_reduction": policy["error_reduction"],
                  "gap_weight_progress": policy["gap_weight_progress"],
                  "data_partition": config.data_split,
                  "loss_fm": loss_fm / count_fm if count_fm else None,
                  "loss_mf": loss_mf / count_mf if count_mf else None,
                  "fm_fraction": count_fm / (count_fm + count_mf) if count_fm + count_mf else None,
                  "mse": values[1].item(),
                  "mse_fm": mse_fm / count_fm if count_fm else None,
                  "mse_mf": mse_mf / count_mf if count_mf else None,
                  "grad_norm": float(grad_norm), "ema_decay": ema_decay,
                  **({"amp_scale": scaler.get_scale(), "amp_skipped": amp_skipped} if scaler.is_enabled() else {}),
                  **({"nonfinite_loss_skipped": True} if skipped_nonfinite else {}),
                  **({"jvp_fp32_fallback": int(fallbacks)} if fallbacks else {}),
                  "lr": used_lr, "epoch": stream.epoch,
                  "metrics_sampled": record_due,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "seconds_per_update": time.perf_counter() - started}
        if accelerator.is_main_process and config.diag_every and step % config.diag_every == 0:
            # Rank-local forward of the last microbatch: no RNG, no collectives.
            probe = slice(0, 64)
            logits = attention_logit_max(accelerator.unwrap_model(model), stats["z"][probe],
                                         stats["r"][probe], stats["t"][probe], labels[probe])
            record.update(attn_logit_max=max(logits), attn_logit_max_by_block=logits)
        probe_ok = torch.ones((), dtype=torch.int32, device=accelerator.device)
        if step % config.checkpointing_steps == 0 or step == config.max_train_steps:
            if accelerator.is_main_process:
                probe_values = run_fixed_probe(accelerator.unwrap_model(model), fixed_probe,
                                               comparison_dtype=(torch.bfloat16 if config.precision == "bf16"
                                                                 else torch.float16))
                record["fixed_probe"] = probe_values
                if not probe_values["fp32_finite"]:
                    probe_ok.zero_()
            if accelerator.reduce(probe_ok, reduction="sum").item() != accelerator.num_processes:
                diverge(f"Nonfinite FP32 fixed-probe target at update {step}")
        if accelerator.device.type == "cuda" and (step == 1 or step % config.memory_every == 0
                                                     or step == config.max_train_steps):
            # CUDA allocator peaks do not include NCCL/context memory. Sample total
            # device usage as well, and retain all ranks rather than rank zero only.
            free, total = torch.cuda.mem_get_info(accelerator.device)
            memory = torch.tensor([[torch.cuda.max_memory_allocated() / 2**30,
                                    torch.cuda.max_memory_reserved() / 2**30,
                                    (total - free) / 2**30, total / 2**30]],
                                  device=accelerator.device, dtype=torch.float64)
            per_rank = accelerator.gather(memory).cpu().tolist()
            record.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          memory_by_rank=[dict(rank=rank, peak_allocated_gib=row[0],
                                               peak_reserved_gib=row[1], device_used_gib=row[2],
                                               device_total_gib=row[3])
                                          for rank, row in enumerate(per_rank)])
        last_record = record
        if accelerator.is_main_process and record_due:
            with (Path(config.output_dir) / "train.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            history.append(record)
            last_logged = step
            if step % config.log_every == 0 or step == 1:
                logger.info(format_progress(record, config.max_train_steps))
            if config.plot_every and (step % config.plot_every == 0 or step == config.max_train_steps):
                write_plots()
        if fallbacks and accelerator.is_main_process:
            logger.info("%s JVP overflowed on %d sample(s) at update %d; recomputed in FP32",
                        config.precision.upper(), int(fallbacks), step)
        # Checked after logging so the tripping update appears in train.jsonl and the plots.
        # A skipped update's statistics cover only its finite microbatches, so it is not judged.
        reason = None if skipped_nonfinite else guard.update(step, record["loss"])
        runtime_state["loss_guard_history"] = list(guard.losses)
        if reason is not None:
            diverge(f"{reason} at update {step}")
        if step % config.checkpointing_steps == 0:
            checkpoint_started = time.perf_counter()
            save_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config, step,
                            scaler=scaler, extra_emas=extra_emas,
                            runtime_state=runtime_state, dropout_generator=dropout_generator,
                            time_generator=time_generator)
            last_saved = step
            last_saved_attempts = runtime_state["attempted_updates"]
            if accelerator.is_main_process:
                event = {"step": step, "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                         "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                with (Path(config.output_dir) / "checkpoints.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(event) + "\n")
                logger.info("Saved checkpoint step-%08d in %.1f s", step, event["checkpoint_seconds"])
        started = time.perf_counter()
    final_checkpoint = checkpoint_path("latest", config.output_dir) if last_saved >= 0 else None
    if step != last_saved or runtime_state["attempted_updates"] != last_saved_attempts:
        # Skips advance RNG/data without advancing the successful step. Preserve
        # their state in a new immutable revision rather than replacing a checkpoint.
        revision = runtime_state["attempted_updates"] if step == last_saved else 0
        final_checkpoint = save_checkpoint(accelerator, model, ema, optimizer, scheduler, stream, config, step,
                        scaler=scaler, extra_emas=extra_emas,
                        runtime_state=runtime_state, dropout_generator=dropout_generator,
                        time_generator=time_generator, attempt_revision=revision)
        logger.info("Saved checkpoint step-%08d", step)
    if accelerator.is_main_process:
        if last_record is not None and last_logged != step:
            with (Path(config.output_dir) / "train.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(last_record) + "\n")
            history.append(last_record)
        write_plots()
        receipt = {"step": step, "target": config.max_train_steps,
                   "reason": stop_reason or "target_reached", "checkpoint_step": step,
                   "checkpoint": str(final_checkpoint.resolve()),
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        path = Path(config.output_dir) / "termination.json"
        staging = path.with_suffix(".tmp")
        staging.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        staging.replace(path)
        logger.info("Termination receipt: %s", json.dumps(receipt))
    logger.info("Training stopped at update %s (target %s).", f"{step:,}", f"{config.max_train_steps:,}")
    accelerator.end_training()
    return step


def main(argv=None):
    try:
        train(parse_config(argv))
    except TrainingDiverged:
        sys.exit(3)  # Already logged; divergence.json is the contract with the controller.


if __name__ == "__main__":
    main()
