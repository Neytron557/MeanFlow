"""Explicit, serializable configuration for the CIFAR-10 experiment."""

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math


@dataclass
class TrainConfig:
    model: str = "DiT-B/4"
    augment_dim: int = 6
    data_dir: str = "data/cifar10"
    output_dir: str = "work_dir/cifar10_dit_b4"
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_train_steps: int = 800000
    warmup_steps: int = 5000
    cooldown_start: int = 0
    cooldown_end: int = 0
    learning_rate: float = 0.0002
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    optimizer: str = "adam"
    optimizer_kernel: str = "reference"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_ns_dtype: str = "fp16"
    muon_rms_scale: float = 0.2
    weight_decay: float = 0.0
    weight_decay_policy: str = "matrix_only"
    max_grad_norm: float = 1.0
    ema_decay: float = 0.99995
    ema_rampup_ratio: float = 0.05
    # Two extra EMA widths on rank 0 (half-life 2.5% and 10% of updates), chosen at evaluation.
    extra_emas: bool = False
    time_mu: float = -2.0
    time_sigma: float = 2.0
    time_sampler: str = "v0"
    # Replacement research policies are opt-in; old checkpoints retain their recipe.
    train_policy: str = "constant"
    curriculum_fm_steps: int = 20000
    curriculum_ramp_steps: int = 20000
    curriculum_weight_end: int = 80000
    curriculum_gap_lambda: float = 1.61914605
    endpoint_focus_probability: float = 0.0
    dedicated_time_rng: bool = False
    time_rng_seed: int = 314159
    data_split: str = "full"
    data_split_manifest: str = ""
    data_split_sha256: str = ""
    attn_projection_dropout: float = 0.0
    mlp_output_dropout: float = 0.0
    dropout_mask_mode: str = "indexed"
    ratio: float = 0.75
    adaptive_p: float = 0.75
    norm_eps: float = 0.001
    checkpointing_steps: int = 10000
    checkpoints_to_keep: int = 3
    log_every: int = 10
    # Ordinary metrics and allocator telemetry may be sampled on fast hardware.
    metrics_every: int = 1
    memory_every: int = 1
    diag_every: int = 100
    plot_every: int = 500
    divergence_loss: float = 7.0
    num_workers: int = 4
    seed: int = 0
    # FP16/BF16 autocast; only FP16 needs a scaler. The legacy JVP flag uses this dtype.
    precision: str = "fp32"
    jvp_fp16: bool = False
    # Scientific forks preserve complete state unless a declared data transition
    # intentionally restarts the deterministic BatchStream on a new partition.
    fork_from: str = ""
    fork_reset_data: bool = False
    fork_reason: str = ""
    # Warm start (weights, EMA, optimizer, update count) from a checkpoint of another run;
    # ignored when resuming this run's own checkpoints.
    init_from: str = ""
    device: str = "cuda"
    download: bool = False
    smoke: bool = False
    augmentation: bool = True
    resume: str = ""
    # Operational bounds; stopping happens only at a complete optimizer boundary.
    deadline_utc: str = ""
    stop_file: str = ""
    gpu_memory_limit_gib: float = 0.0
    gpu_free_margin_gib: float = 0.0

    def effective_batch_size(self, world_size):
        return self.per_device_batch_size * world_size * self.gradient_accumulation_steps

    def validate(self, world_size):
        for name in ("per_device_batch_size", "gradient_accumulation_steps", "max_train_steps",
                     "checkpointing_steps", "checkpoints_to_keep", "log_every", "metrics_every", "memory_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.warmup_steps < 0 or world_size < 1:
            raise ValueError("Workers/warmup must be nonnegative and world size positive")
        if not 0 <= self.seed <= 2**32 - world_size:
            raise ValueError("seed plus each process rank must fit in NumPy's uint32 range")
        if self.model not in ("DiT-B/4", "DiT-Tiny/4"):
            raise ValueError("CIFAR-10 supports DiT-B/4 (or DiT-Tiny/4 for smoke tests)")
        if self.model != "DiT-B/4" and not self.smoke:
            raise ValueError("DiT-Tiny/4 requires --smoke; it is not the experiment model")
        if self.device not in ("cpu", "cuda") or self.augment_dim != 6:
            raise ValueError("Use device cpu/cuda and augment_dim=6")
        for name in ("learning_rate", "adam_epsilon", "norm_eps", "time_sigma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("adam_beta1", "adam_beta2", "ema_decay"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 <= self.ratio <= 1 or not 0 <= self.adaptive_p <= 1:
            raise ValueError("ratio and adaptive_p must be in [0, 1]")
        if not math.isfinite(self.time_mu):
            raise ValueError("time_mu must be finite")
        if self.time_sampler not in ("v0", "v1"):
            raise ValueError("time_sampler must be v0 or v1")
        if self.train_policy not in ("constant", "fm_curriculum"):
            raise ValueError("train_policy must be constant or fm_curriculum")
        if any(type(getattr(self, name)) is not int for name in
               ("curriculum_fm_steps", "curriculum_ramp_steps", "curriculum_weight_end")):
            raise ValueError("Curriculum phase lengths must be integers")
        if self.curriculum_fm_steps < 0 or self.curriculum_ramp_steps <= 0:
            raise ValueError("Curriculum FM steps must be nonnegative and ramp steps positive")
        if self.curriculum_weight_end <= self.curriculum_fm_steps + self.curriculum_ramp_steps:
            raise ValueError("curriculum_weight_end must follow the FM and ramp phases")
        if not math.isfinite(self.curriculum_gap_lambda) or self.curriculum_gap_lambda <= 0:
            raise ValueError("curriculum_gap_lambda must be finite and positive")
        if not math.isfinite(self.endpoint_focus_probability) or not 0 <= self.endpoint_focus_probability <= 1:
            raise ValueError("endpoint_focus_probability must be finite and in [0, 1]")
        if self.train_policy == "fm_curriculum" and self.endpoint_focus_probability:
            raise ValueError("Endpoint focus is a separate constant-policy experiment")
        if (self.train_policy != "constant" or self.endpoint_focus_probability) and not self.dedicated_time_rng:
            raise ValueError("Curriculum and endpoint focus require dedicated_time_rng")
        if type(self.time_rng_seed) is not int or not 0 <= self.time_rng_seed <= 2**63 - world_size:
            raise ValueError("time_rng_seed plus each process rank must fit in a signed int64")
        if self.data_split not in ("full", "train"):
            raise ValueError("data_split must be full or train; holdout images cannot train")
        if self.data_split == "train" and not self.data_split_manifest:
            raise ValueError("data_split=train requires an immutable data_split_manifest")
        if self.data_split_sha256 and (len(self.data_split_sha256) != 64 or
                                     any(c not in "0123456789abcdef" for c in self.data_split_sha256)):
            raise ValueError("data_split_sha256 must be a lowercase SHA256 digest")
        if self.data_split_sha256 and not self.data_split_manifest:
            raise ValueError("data_split_sha256 requires data_split_manifest")
        for name in ("attn_projection_dropout", "mlp_output_dropout"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if self.dropout_mask_mode != "indexed":
            raise ValueError("dropout_mask_mode must be indexed")
        if self.optimizer not in ("adam", "adamw", "muon"):
            raise ValueError("optimizer must be adam, adamw, or muon")
        if self.optimizer_kernel not in ("reference", "foreach", "fused"):
            raise ValueError("optimizer_kernel must be reference, foreach, or fused")
        if self.optimizer == "muon" and self.optimizer_kernel != "reference":
            raise ValueError("Muon requires optimizer_kernel=reference")
        if not math.isfinite(self.muon_momentum) or not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be finite and in [0, 1)")
        if type(self.muon_ns_steps) is not int or self.muon_ns_steps <= 0 or self.muon_ns_dtype not in ("fp16", "fp32"):
            raise ValueError("Muon requires positive muon_ns_steps and muon_ns_dtype fp16/fp32")
        if not math.isfinite(self.muon_rms_scale) or self.muon_rms_scale <= 0:
            raise ValueError("muon_rms_scale must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.optimizer in ("adam", "muon") and self.weight_decay != 0:
            raise ValueError("Nonzero weight_decay requires optimizer=adamw")
        if self.weight_decay_policy != "matrix_only":
            raise ValueError("weight_decay_policy must be matrix_only")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and nonnegative (0 disables clipping)")
        if not 0 <= self.ema_rampup_ratio <= 1:
            raise ValueError("ema_rampup_ratio must be in [0, 1] (0 disables the rampup)")
        if not math.isfinite(self.divergence_loss) or self.divergence_loss < 0:
            raise ValueError("divergence_loss must be finite and nonnegative (0 disables the guard)")
        if self.diag_every < 0:
            raise ValueError("diag_every must be nonnegative (0 disables attention diagnostics)")
        if self.cooldown_end or self.cooldown_start:
            if not self.warmup_steps <= self.cooldown_start < self.cooldown_end:
                raise ValueError("Cooldown needs warmup_steps <= cooldown_start < cooldown_end (0/0 disables it)")
            if self.max_train_steps > self.cooldown_end:
                raise ValueError("max_train_steps must not exceed cooldown_end; the learning rate reaches 0 there")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.jvp_fp16 and self.precision == "fp32":
            raise ValueError("jvp_fp16 requires precision fp16 or bf16")
        if self.precision == "fp16" and self.device != "cuda":
            raise ValueError("precision fp16 needs CUDA; use bf16 for CPU checks")
        if self.plot_every < 0:
            raise ValueError("plot_every must be nonnegative (0 disables loss plots)")
        if self.fork_from and not self.fork_reason.strip():
            raise ValueError("fork_from requires a scientific fork_reason")
        if self.fork_reset_data and not self.fork_from:
            raise ValueError("fork_reset_data requires fork_from")
        if self.fork_from and self.init_from:
            raise ValueError("fork_from and legacy init_from are mutually exclusive")
        from .runtime import parse_deadline
        parse_deadline(self.deadline_utc)
        for name in ("gpu_memory_limit_gib", "gpu_free_margin_gib"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Unconditional CIFAR-10 MeanFlow with DiT-B/4")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--config", help="JSON defaults; explicit CLI arguments override these")
    for field in fields(TrainConfig):
        option = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(option, type=field.type, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    args.pop("dataset")
    values = asdict(TrainConfig())
    if path:
        with open(path, encoding="utf-8") as source:
            loaded = json.load(source)
        if not isinstance(loaded, dict):
            parser.error("Configuration JSON must be an object")
        unknown = set(loaded) - set(values)
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        values.update(loaded)
    values.update({key: value for key, value in args.items() if value is not None})
    for field in fields(TrainConfig):
        value = values[field.name]
        if field.type is float and isinstance(value, int) and not isinstance(value, bool):
            values[field.name] = float(value)
        elif type(value) is not field.type:
            parser.error(f"{field.name} must be {field.type.__name__}")
    return TrainConfig(**values)
