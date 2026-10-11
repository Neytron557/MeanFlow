"""Conservative recursive audit of checkpoints claiming an unseen 5k partition."""

import hashlib
import json
from pathlib import Path

import torch

from .fid_cache import sha256_file


def parse_ancestor_path_maps(values):
    mappings = []
    for value in values or []:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            raise ValueError("Ancestor path maps must be OLD_PREFIX=NEW_PREFIX")
        mappings.append((Path(old).expanduser(), Path(new).expanduser()))
    return mappings


def audit_unseen_lineage(config, checkpoint, split_sha256, *, project_root, path_maps=(), checkpoint_step=None):
    """Require every declared warm-start/fork ancestor to use the same 45k split.

    Paths from another server are remapped only through explicit prefix mappings.
    A missing ancestor is not interpreted as a fresh run. The audit records each
    ancestor's file and configuration digest, rather than silently trusting an
    arbitrary nonempty ``init_from`` value.
    """
    project_root = Path(project_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    visited, active, records = set(), set(), []

    def resolve_source(value):
        path = Path(value).expanduser()
        for old, new in sorted(path_maps, key=lambda item: len(item[0].parts), reverse=True):
            try:
                relative = path.relative_to(old)
            except ValueError:
                continue
            path = new / relative
            break
        if not path.is_absolute():
            path = project_root / path
        if path.is_dir():
            path = path / "weights.pt"
        if not path.is_file():
            raise ValueError(f"Cannot certify warm start or fork: missing ancestor checkpoint {path}")
        return path.resolve()

    def visit(saved_config, path, *, child_step=None, step=None, edge=None):
        if path in active:
            raise ValueError("Cyclic checkpoint warm start/fork lineage cannot certify unseen rows")
        if path in visited:
            return
        if saved_config.get("data_split") != "train":
            raise ValueError("Unseen holdout diagnostics retire for any full-data warm start or fork ancestor")
        if saved_config.get("data_split_sha256") != split_sha256:
            raise ValueError("Warm start/fork ancestor split SHA256 differs from the diagnostic partition")
        if child_step is not None and (type(step) is not int or step < 0 or step > child_step):
            raise ValueError("Warm start/fork ancestor update count is incompatible with its child")
        active.add(path)
        configuration_hash = hashlib.sha256(json.dumps(saved_config, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        records.append({"checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
                        "config_sha256": configuration_hash, "step": step,
                        "data_split": "train", "data_split_sha256": split_sha256,
                        "ancestor_edge": edge})
        for source_key in ("init_from", "fork_from"):
            source = saved_config.get(source_key)
            if not source:
                continue
            parent = resolve_source(source)
            parent_weights = torch.load(parent, map_location="cpu", weights_only=True, mmap=True)
            parent_config = parent_weights.get("config") if isinstance(parent_weights, dict) else None
            if not isinstance(parent_config, dict):
                raise ValueError("Warm start/fork ancestor has no auditable checkpoint configuration")
            visit(parent_config, parent, child_step=step, step=parent_weights.get("step"), edge=source_key)
        active.remove(path)
        visited.add(path)

    # The caller already loaded and verified this checkpoint's model metadata.
    visit(config, checkpoint, step=checkpoint_step)
    return {"status": "all_declared_ancestors_same_45k_partition", "split_sha256": split_sha256,
            "checkpoints": records,
            "path_maps": [{"old_prefix": str(old), "new_prefix": str(new)} for old, new in path_maps]}
