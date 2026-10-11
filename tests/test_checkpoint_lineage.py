"""A migrated 45k warm start is unseen only when every ancestor is audited."""

import pytest
import torch

from cifar10.lineage import audit_unseen_lineage, parse_ancestor_path_maps


SPLIT = "a" * 64


def config(**changes):
    return {"data_split": "train", "data_split_sha256": SPLIT, "init_from": "", "fork_from": "", **changes}


def checkpoint(tmp_path, name, cfg, step):
    directory = tmp_path / name
    directory.mkdir()
    torch.save({"config": cfg, "step": step}, directory / "weights.pt")
    return directory


def audit(path, cfg, **kwargs):
    return audit_unseen_lineage(cfg, path / "weights.pt", SPLIT, project_root=path.parent, **kwargs)


def test_migrated_45k_warm_start_recursively_audits_matching_ancestors(tmp_path):
    source = checkpoint(tmp_path, "fresh", config(), 120000)
    warm_cfg = config(init_from=str(source))
    warm = checkpoint(tmp_path, "warm", warm_cfg, 140000)
    fork_cfg = config(fork_from=str(warm))
    fork = checkpoint(tmp_path, "fork", fork_cfg, 160000)
    receipt = audit(fork, fork_cfg, checkpoint_step=160000)
    assert len(receipt["checkpoints"]) == 3
    assert [row["step"] for row in receipt["checkpoints"]] == [160000, 140000, 120000]
    assert all(row["data_split_sha256"] == SPLIT and len(row["checkpoint_sha256"]) == 64
               and len(row["config_sha256"]) == 64 for row in receipt["checkpoints"])


@pytest.mark.parametrize("ancestor_changes,message", [
    ({"data_split": "full"}, "retire"),
    ({"data_split_sha256": "b" * 64}, "SHA256"),
    ({"init_from": "missing"}, "missing ancestor"),
])
def test_unsafe_ancestor_is_not_allowed(ancestor_changes, message, tmp_path):
    source = checkpoint(tmp_path, "source", config(**ancestor_changes), 120000)
    cfg = config(init_from=str(source))
    warm = checkpoint(tmp_path, "warm", cfg, 140000)
    with pytest.raises(ValueError, match=message):
        audit(warm, cfg, checkpoint_step=140000)


def test_cyclic_lineage_and_future_ancestor_are_rejected(tmp_path):
    cyclic_cfg = config(init_from=str(tmp_path / "cycle"))
    cycle = checkpoint(tmp_path, "cycle", cyclic_cfg, 10)
    with pytest.raises(ValueError, match="Cyclic"):
        audit(cycle, cyclic_cfg, checkpoint_step=10)
    source = checkpoint(tmp_path, "future", config(), 20)
    cfg = config(init_from=str(source))
    early = checkpoint(tmp_path, "early", cfg, 10)
    with pytest.raises(ValueError, match="update count"):
        audit(early, cfg, checkpoint_step=10)


def test_archived_paths_require_explicit_prefix_mapping(tmp_path):
    checkpoint(tmp_path, "source", config(), 10)
    cfg = config(init_from="/archived/project/source")
    warm = checkpoint(tmp_path, "warm", cfg, 20)
    with pytest.raises(ValueError, match="missing ancestor"):
        audit(warm, cfg, checkpoint_step=20)
    maps = parse_ancestor_path_maps([f"/archived/project={tmp_path}"])
    result = audit(warm, cfg, checkpoint_step=20, path_maps=maps)
    assert len(result["checkpoints"]) == 2
    assert result["path_maps"][0]["old_prefix"] == "/archived/project"
    with pytest.raises(ValueError, match="OLD_PREFIX"):
        parse_ancestor_path_maps(["not a mapping"])
