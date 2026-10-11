"""Loss 1 / loss 2 / total-loss plots written from the training log."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from cifar10.plots import CURVES, LossHistory, main, rolling_mean, save_loss_plots, smoothing_window

pytest.importorskip("matplotlib")


def test_rolling_mean_is_trailing_and_ignores_missing_values():
    values = [1.0, math.nan, 3.0, 5.0]
    np.testing.assert_allclose(rolling_mean(values, 2), [1.0, 1.0, 3.0, 4.0])
    assert smoothing_window(10) == 100 and smoothing_window(10**7) == 1000


def test_history_reads_old_and_new_rows_and_stops_at_checkpoint(tmp_path):
    log = tmp_path / "train.jsonl"
    rows = [{"step": 1, "loss": 7.9},  # A pre-v2 row has only the total loss.
            {"step": 2, "loss": 7.0, "loss_fm": 6.0, "loss_mf": 7.3},
            {"step": 3, "loss": 6.9, "loss_fm": None, "loss_mf": 6.9}]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"step":')
    history = LossHistory.from_jsonl(log, max_step=2)
    assert list(history.steps) == [1, 2]
    assert math.isnan(history.values["loss_fm"][0]) and history.values["loss_mf"][1] == 7.3


def test_three_pngs_are_written_and_empty_curves_skipped(tmp_path):
    history = LossHistory()
    generator = np.random.default_rng(0)
    for step in range(1, 301):
        base = 5.5 + 2.4 * math.exp(-step / 40)
        history.append({"step": step, "loss": base + generator.normal(0, 0.05),
                        "loss_fm": base - 0.6 + generator.normal(0, 0.1),
                        "loss_mf": base + 0.2 + generator.normal(0, 0.1)})
    written = save_loss_plots(tmp_path, history, divergence_loss=7.0)
    assert [path.name for path in written] == [name for _, name, _, _ in CURVES]
    for path in written:
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size[0] > 800
    assert not list((tmp_path / "plots").glob(".*.tmp"))
    old = LossHistory()
    old.append({"step": 1, "loss": 7.9})
    assert [path.name for path in save_loss_plots(tmp_path / "old", old)] == ["total_loss.png"]


def test_cli_regenerates_plots_for_a_run(tmp_path, capsys):
    (tmp_path / "train.jsonl").write_text(json.dumps({"step": 1, "loss": 7.9}) + "\n")
    (tmp_path / "config.json").write_text(json.dumps({"divergence_loss": 0.0}))
    main(["--run-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "total_loss.png" in output and "Skipped" in output
