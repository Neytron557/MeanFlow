"""Replicated hybrid Muon with transactional updates for manually managed AMP.

The five-step polynomial follows https://github.com/KellerJordan/Muon.
RMS scaling follows https://arxiv.org/abs/2502.16982. This implementation keeps
optimizer state in FP32 and normalizes before casting Newton--Schulz to FP16.
It does not shard optimizer state or alter the network's fused QKV parameters.
"""

from dataclasses import dataclass
import math

import torch


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def unwrap_optimizer(optimizer):
    """Access the replicated optimizer under Accelerate's accumulation wrapper."""
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    return optimizer


def newton_schulz(matrix, *, steps=5, dtype=torch.float16):
    """Approximate a polar factor; exact orthogonality is not an invariant.

    FP32 normalization precedes the half cast, avoiding overflow/underflow of
    unnormalized gradients. Momentum, normalization and returned FP32 fallback
    computations are never controlled by the caller's autocast context.
    """
    if matrix.ndim != 2 or matrix.dtype != torch.float32:
        raise ValueError("Newton--Schulz needs an FP32 matrix")
    if dtype not in (torch.float16, torch.float32) or steps < 1:
        raise ValueError("Use positive NS steps and FP16 or FP32 arithmetic")
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        x = matrix / (torch.linalg.vector_norm(matrix) + 1e-7)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.mT
        x = x.to(dtype=dtype)
        a, b, c = NS_COEFFICIENTS
        for _ in range(steps):
            gram = x @ x.mT
            polynomial = b * gram + c * (gram @ gram)
            x = a * x + polynomial @ x
        return x.mT if transposed else x


@dataclass
class _Candidate:
    parameter: torch.Tensor
    group: dict
    gradient: torch.Tensor
    gradient_version: int
    parameter_version: int
    state: dict
    update: torch.Tensor
    # Muon keeps the unscaled direction; Adam keeps its final candidate weight.
    muon: bool
    group_signature: tuple


class HybridMuon(torch.optim.Optimizer):
    """Hidden matrix Muon plus Adam, with all state replicated on every DDP rank.

    ``stage_updates`` and ``reconcile_staged`` do not change parameters, gradients
    or optimizer state. ``step`` consumes an approved transaction. In a local
    process it may prepare its own transaction; distributed callers must use the
    collective AMP integration rather than calling a local optimizer step.
    """

    def __init__(self, groups, *, lr=2e-4, betas=(0.9, 0.95), eps=1e-8,
                 momentum=0.95, ns_steps=5, ns_dtype="fp16", rms_scale=0.2):
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Muon LR and auxiliary Adam epsilon must be positive")
        if any(not 0 <= beta < 1 for beta in betas) or not 0 <= momentum < 1:
            raise ValueError("Adam betas and Muon momentum must lie in [0, 1)")
        if not isinstance(ns_steps, int) or isinstance(ns_steps, bool) or ns_steps < 1:
            raise ValueError("Muon NS steps must be a positive integer")
        if ns_dtype not in ("fp16", "fp32") or not math.isfinite(rms_scale) or rms_scale <= 0:
            raise ValueError("Use FP16/FP32 NS and a positive RMS scale")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0,
                        momentum=momentum, ns_steps=ns_steps, ns_dtype=ns_dtype,
                        rms_scale=rms_scale, nesterov=True)
        super().__init__(groups, defaults)
        seen = set()
        for group in self.param_groups:
            if group.get("kind") not in ("muon", "adam") or group["weight_decay"] != 0:
                raise ValueError("Hybrid groups must be Muon/Adam with zero weight decay")
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.dtype != torch.float32:
                    raise ValueError("Hybrid Muon requires trainable FP32 parameters")
                if id(parameter) in seen:
                    raise ValueError("Duplicate hybrid optimizer parameter")
                seen.add(id(parameter))
                if group["kind"] == "muon" and parameter.ndim != 2:
                    raise ValueError("Muon parameters must be hidden matrices")
        if not any(group["kind"] == "muon" for group in self.param_groups):
            raise ValueError("Hybrid Muon requires at least one hidden matrix")
        self._staged = None
        self._stage_approved = False
        self.last_step_diagnostics = {"ns_fp32_fallback": 0, "committed": False}

    @staticmethod
    def _step_number(state):
        value = state.get("step", 0)
        return int(value.item() if isinstance(value, torch.Tensor) else value)

    def recipe(self):
        group = next(group for group in self.param_groups if group["kind"] == "muon")
        return dict(version=1, scaling="rms", coefficients=list(NS_COEFFICIENTS),
                    momentum=group["momentum"], ns_steps=group["ns_steps"],
                    ns_dtype=group["ns_dtype"], rms_scale=group["rms_scale"], nesterov=True)

    @staticmethod
    def _group_signature(group):
        return tuple(group.get(key) for key in ("kind", "lr", "betas", "eps", "weight_decay",
                                                "momentum", "ns_steps", "ns_dtype", "rms_scale", "nesterov"))

    @torch.no_grad()
    def stage_updates(self):
        if self._staged is not None:
            raise RuntimeError("A hybrid update is already staged")
        self._stage_approved = False
        self.last_step_diagnostics = {"ns_fp32_fallback": 0, "committed": False}
        candidates, presence, ns_bad = [], [], []
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                presence.append(gradient is not None)
                if group["kind"] == "muon":
                    ns_bad.append(torch.zeros((), dtype=torch.int32, device=parameter.device))
                if gradient is None:
                    candidates.append(None)
                    continue
                if gradient.is_sparse or gradient.dtype != torch.float32:
                    raise ValueError("Hybrid Muon needs dense FP32 gradients")
                old = self.state.get(parameter, {})
                step = self._step_number(old) + 1
                next_state = {"step": torch.tensor(float(step), dtype=torch.float32)}
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    if group["kind"] == "muon":
                        old_momentum = old.get("momentum_buffer")
                        if old_momentum is None:
                            old_momentum = torch.zeros_like(parameter)
                        momentum = torch.lerp(old_momentum, gradient, 1 - group["momentum"])
                        next_state["momentum_buffer"] = momentum
                        nesterov = torch.lerp(gradient, momentum, group["momentum"])
                        dtype = torch.float16 if group["ns_dtype"] == "fp16" else torch.float32
                        update = newton_schulz(nesterov, steps=group["ns_steps"], dtype=dtype)
                        ns_bad[-1] = (~torch.isfinite(update).all()).to(torch.int32)
                    else:
                        first = old["exp_avg"].clone() if old else torch.zeros_like(parameter)
                        second = old["exp_avg_sq"].clone() if old else torch.zeros_like(parameter)
                        beta1, beta2 = group["betas"]
                        first.lerp_(gradient, 1 - beta1)
                        second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                        next_state.update(exp_avg=first, exp_avg_sq=second)
                        denominator = second.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                        update = torch.addcdiv(parameter, first, denominator,
                                              value=-group["lr"] / (1 - beta1 ** step))
                candidates.append(_Candidate(parameter, group, gradient, gradient._version,
                                             parameter._version, next_state, update,
                                             group["kind"] == "muon", self._group_signature(group)))
        device = self.param_groups[0]["params"][0].device
        self._staged = dict(candidates=candidates,
                            presence=torch.tensor(presence, dtype=torch.int32, device=device),
                            ns_bad=torch.stack(ns_bad))

    @torch.no_grad()
    def reconcile_staged(self, accelerator=None):
        """Agree on fallback matrices and finiteness before approving any mutation."""
        if self._staged is None:
            raise RuntimeError("No hybrid update is staged")
        world = 1 if accelerator is None else accelerator.num_processes

        def reduce(value):
            return value if accelerator is None else accelerator.reduce(value, reduction="sum")

        presence = reduce(self._staged["presence"])
        consistent = ((presence == 0) | (presence == world)).all()
        fallback_mask = reduce(self._staged["ns_bad"]).gt(0).tolist()
        self.last_step_diagnostics["ns_fp32_fallback"] = sum(fallback_mask)
        muon_index = 0
        finite = [consistent]
        for candidate, group, parameter in self._iter_staged():
            if group["kind"] == "muon":
                fallback = fallback_mask[muon_index]
                muon_index += 1
            else:
                fallback = False
            if candidate is None:
                continue
            if fallback:
                nesterov = torch.lerp(candidate.gradient, candidate.state["momentum_buffer"],
                                      group["momentum"])
                candidate.update = newton_schulz(nesterov, steps=group["ns_steps"], dtype=torch.float32)
            finite.append(torch.isfinite(candidate.update).all())
            finite.extend(torch.isfinite(value).all() for key, value in candidate.state.items()
                          if key != "step")
            if candidate.muon:
                scale = group["rms_scale"] * math.sqrt(max(parameter.shape))
                projected = parameter.add(candidate.update.float(), alpha=-group["lr"] * scale)
                finite.append(torch.isfinite(projected).all())
        valid = torch.stack(finite).all().to(torch.int32)
        self._stage_approved = reduce(valid).item() == world
        return self._stage_approved

    def _iter_staged(self):
        index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                yield self._staged["candidates"][index], group, parameter
                index += 1

    def discard_staged(self):
        self._staged = None
        self._stage_approved = False

    @torch.no_grad()
    def commit_staged(self):
        if self._staged is None or not self._stage_approved:
            raise RuntimeError("Hybrid update has not passed the finite collective")
        # Validate the whole transaction before changing even the first parameter.
        candidates = [item for item in self._staged["candidates"] if item is not None]
        for candidate in candidates:
            if (candidate.parameter.grad is not candidate.gradient
                    or candidate.gradient._version != candidate.gradient_version
                    or candidate.parameter._version != candidate.parameter_version
                    or self._group_signature(candidate.group) != candidate.group_signature):
                raise RuntimeError("Parameter or gradient changed after staging")
        for candidate in candidates:
            parameter, group = candidate.parameter, candidate.group
            if candidate.muon:
                scale = group["rms_scale"] * math.sqrt(max(parameter.shape))
                parameter.add_(candidate.update.float(), alpha=-group["lr"] * scale)
            else:
                parameter.copy_(candidate.update)
            self.state[parameter] = candidate.state
        self.discard_staged()
        self.last_step_diagnostics["committed"] = True

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Hybrid Muon does not support optimizer closures")
        if self._staged is None:
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                raise RuntimeError("Distributed HybridMuon requires optimizer_attempt")
            self.stage_updates()
            if not self.reconcile_staged():
                self.discard_staged()
                raise FloatingPointError("Nonfinite hybrid optimizer transaction")
        self.commit_staged()
