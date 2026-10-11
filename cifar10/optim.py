"""Named optimizer state loading, including the legacy frozen-parameter offset."""

from copy import deepcopy
import math
import re

import torch

from .muon import HybridMuon, NS_COEFFICIENTS, unwrap_optimizer


_HIDDEN_MATRIX = re.compile(r"blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))\.weight$")


def create_optimizer(model, config):
    """Adam preserves historical order; AdamW decays only trainable matrices."""
    named = list(model.named_parameters())
    kernel = getattr(config, "optimizer_kernel", "reference")
    common = dict(lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
                  eps=config.adam_epsilon, foreach=kernel == "foreach", fused=kernel == "fused")
    if config.optimizer == "muon":
        if config.weight_decay != 0:
            raise ValueError("The hybrid Muon recipe uses zero weight decay")
        groups = []
        for kind in ("muon", "adam"):
            selected = [(name, parameter) for name, parameter in named
                        if parameter.requires_grad and
                        bool(_HIDDEN_MATRIX.fullmatch(name)) == (kind == "muon")]
            if selected:
                groups.append(dict(params=[p for _, p in selected],
                                   param_names=[name for name, _ in selected], kind=kind))
        return HybridMuon(groups, lr=config.learning_rate,
                          betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
                          momentum=config.muon_momentum, ns_steps=config.muon_ns_steps,
                          ns_dtype=config.muon_ns_dtype, rms_scale=config.muon_rms_scale)
    if config.optimizer == "adam":
        if config.weight_decay != 0:
            raise ValueError("Nonzero weight_decay requires optimizer=adamw")
        return torch.optim.Adam([dict(params=[p for _, p in named],
                                      param_names=[n for n, _ in named])],
                                weight_decay=0.0, **common)
    if config.optimizer != "adamw" or config.weight_decay_policy != "matrix_only":
        raise ValueError("Expected adamw with matrix_only weight decay")
    groups = []
    for decay in (True, False):
        selected = [(n, p) for n, p in named if p.requires_grad and (p.ndim >= 2) == decay]
        if selected:
            groups.append(dict(params=[p for _, p in selected],
                               param_names=[n for n, _ in selected],
                               weight_decay=config.weight_decay if decay else 0.0))
    return torch.optim.AdamW(groups, **common)


def optimizer_metadata(optimizer, model):
    """Parameter names are stored independently of opaque optimizer state IDs."""
    by_id = {id(p): name for name, p in model.named_parameters()}
    metadata = {"version": 1,
                "param_groups": [[by_id[id(p)] for p in group["params"]]
                                 for group in optimizer.param_groups]}
    base = unwrap_optimizer(optimizer)
    if isinstance(base, HybridMuon):
        metadata.update(version=2, kind="hybrid_muon",
                        group_kinds=[group["kind"] for group in optimizer.param_groups],
                        muon_recipe=base.recipe())
    return metadata


def _source_names(saved, model, metadata):
    groups = saved["param_groups"]
    if metadata is not None:
        if metadata.get("version") not in (1, 2):
            raise ValueError("Unsupported optimizer metadata version")
        names = metadata["param_groups"]
    elif all("param_names" in group for group in groups):
        names = [group["param_names"] for group in groups]
    else:
        # Historical Adam used model.parameters(), including frozen pos_embed.
        names = [[name for name, _ in model.named_parameters()]]
        if len(groups) != 1 or len(groups[0]["params"]) != len(names[0]):
            raise ValueError("Cannot safely map unnamed legacy optimizer parameter groups")
    if len(groups) != len(names):
        raise ValueError("Optimizer metadata group count mismatch")
    result = {}
    seen = set()
    model_names = dict(model.named_parameters())
    for group, group_names in zip(groups, names):
        if len(group["params"]) != len(group_names):
            raise ValueError("Optimizer metadata parameter count mismatch")
        if "param_names" in group and group["param_names"] != group_names:
            raise ValueError("Optimizer names disagree with saved metadata")
        for index, name in zip(group["params"], group_names):
            if name not in model_names or name in seen or index in result:
                raise ValueError(f"Invalid or duplicate optimizer parameter: {name}")
            seen.add(name)
            result[index] = name
    if set(saved["state"]) - set(result):
        raise ValueError("Optimizer state contains unknown parameter IDs")
    return result


def _validate_moments(name, parameter, state):
    if not state:
        return
    if not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
        raise ValueError(f"Incomplete Adam moments for {name}")
    step = state["step"]
    if isinstance(step, torch.Tensor):
        if step.numel() != 1 or not step.is_floating_point():
            raise ValueError(f"Invalid optimizer step for {name}")
        step = step.item()
    if not math.isfinite(step) or step < 0 or int(step) != step:
        raise ValueError(f"Invalid optimizer step for {name}")
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key not in state:
            continue
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"Optimizer {key} shape mismatch for {name}")
        if value.dtype != parameter.dtype:
            raise ValueError(f"Optimizer {key} dtype mismatch for {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite optimizer {key} for {name}")


def _validate_muon_state(name, parameter, state):
    if not state:
        return
    if set(state) != {"step", "momentum_buffer"}:
        raise ValueError(f"Incomplete Muon momentum for {name}")
    # Reuse Adam's step and tensor checks without accepting Adam state migration.
    _validate_moments(name, parameter, dict(step=state["step"],
                                           exp_avg=state["momentum_buffer"],
                                           exp_avg_sq=state["momentum_buffer"]))


def _hybrid_source_kinds(saved, metadata):
    kinds = [group.get("kind") for group in saved["param_groups"]]
    hybrid = "muon" in kinds or (metadata or {}).get("kind") == "hybrid_muon"
    if hybrid:
        if "muon" not in kinds or any(kind not in ("muon", "adam") for kind in kinds):
            raise ValueError("Invalid hybrid optimizer group kinds")
        if metadata is not None and (metadata.get("version") != 2
                or metadata.get("kind") != "hybrid_muon" or metadata.get("group_kinds") != kinds):
            raise ValueError("Hybrid optimizer metadata disagrees with saved groups")
        for group in saved["param_groups"]:
            if group.get("weight_decay") != 0:
                raise ValueError("Hybrid optimizer checkpoint has nonzero weight decay")
            if (not math.isfinite(group.get("lr", float("nan"))) or group["lr"] < 0
                    or len(group.get("betas", ())) != 2
                    or any(not 0 <= beta < 1 for beta in group["betas"])
                    or not math.isfinite(group.get("eps", float("nan"))) or group["eps"] <= 0):
                raise ValueError("Invalid hybrid auxiliary Adam settings")
            if group["kind"] == "muon":
                required = {"momentum", "ns_steps", "ns_dtype", "rms_scale", "nesterov"}
                if (not required <= group.keys() or group["nesterov"] is not True
                        or not 0 <= group["momentum"] < 1
                        or not isinstance(group["ns_steps"], int) or isinstance(group["ns_steps"], bool)
                        or group["ns_steps"] < 1 or group["ns_dtype"] not in ("fp16", "fp32")
                        or not math.isfinite(group["rms_scale"]) or group["rms_scale"] <= 0):
                    raise ValueError("Incomplete hybrid Muon checkpoint recipe")
                recipe = dict(version=1, scaling="rms", coefficients=list(NS_COEFFICIENTS),
                              **{key: group[key] for key in required})
                if metadata is not None and metadata.get("muon_recipe") != recipe:
                    raise ValueError("Muon recipe disagrees with checkpoint metadata")
    elif metadata is not None and metadata.get("version") != 1:
        raise ValueError("Unexpected metadata for an Adam optimizer")
    return hybrid, kinds


def load_optimizer_state(optimizer, saved, model, config, metadata=None, *, warm_start=False):
    """Validate CPU moments, map by name, then restore to current parameter groups.

    Source dictionaries are never mutated. The checkpoint loader keeps these
    tensors on CPU so migration does not allocate a second model's state on GPU.
    """
    target_hybrid = isinstance(unwrap_optimizer(optimizer), HybridMuon)
    source_hybrid, source_kinds = _hybrid_source_kinds(saved, metadata)
    if source_hybrid != target_hybrid:
        raise ValueError("Adam and hybrid Muon optimizer states cannot be migrated")
    source_names = _source_names(saved, model, metadata)
    named = dict(model.named_parameters())
    by_name = {source_names[index]: state for index, state in saved["state"].items()}
    kinds_by_name = {source_names[index]: kind
                     for group, kind in zip(saved["param_groups"], source_kinds)
                     for index in group["params"]}
    for name, state in by_name.items():
        if kinds_by_name[name] == "muon":
            _validate_muon_state(name, named[name], state)
        else:
            _validate_moments(name, named[name], state)
            if target_hybrid and state and (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                                           or (state["exp_avg_sq"] < 0).any()):
                raise ValueError(f"Invalid hybrid auxiliary Adam state for {name}")
    destination = optimizer.state_dict()
    target_names = optimizer_metadata(optimizer, model)["param_groups"]
    all_source_names = set(source_names.values())
    if any(name not in all_source_names for group in target_names for name in group):
        raise ValueError("Checkpoint optimizer is missing destination parameters")
    if target_hybrid:
        for group, names in zip(optimizer.param_groups, target_names):
            if any(kinds_by_name[name] != group["kind"] for name in names):
                raise ValueError("Hybrid checkpoint changed parameter optimizer roles")
        if not warm_start:
            for source, target in zip(saved["param_groups"], optimizer.param_groups):
                keys = ("betas", "eps", "momentum", "ns_steps", "ns_dtype", "rms_scale", "nesterov")
                if any(source.get(key) != target.get(key) for key in keys):
                    raise ValueError("Same-run hybrid optimizer recipe changed")
    if not warm_start:
        original_names = [[source_names[index] for index in group["params"]]
                          for group in saved["param_groups"]]
        if original_names != target_names:
            raise ValueError("Same-run optimizer parameter groups changed")
    remapped = {"state": {}, "param_groups": []}
    for group_index, (group, names) in enumerate(zip(destination["param_groups"], target_names)):
        values = dict(group if warm_start else saved["param_groups"][group_index])
        values.update(params=group["params"], param_names=names)
        remapped["param_groups"].append(values)
        for index, name in zip(group["params"], names):
            if name in by_name:
                remapped["state"][index] = deepcopy(by_name[name]) if target_hybrid else by_name[name]
    optimizer.load_state_dict(remapped)
    if warm_start:
        for group in optimizer.param_groups:
            decay = (config.weight_decay if config.optimizer == "adamw"
                     and group["params"][0].ndim >= 2 else 0.0)
            group.update(lr=config.learning_rate, initial_lr=config.learning_rate,
                         betas=(config.adam_beta1, config.adam_beta2),
                         eps=config.adam_epsilon, weight_decay=decay)
            if target_hybrid:
                group.update(momentum=config.muon_momentum, ns_steps=config.muon_ns_steps,
                             ns_dtype=config.muon_ns_dtype, rms_scale=config.muon_rms_scale,
                             nesterov=True)
