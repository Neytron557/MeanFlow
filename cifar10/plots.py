"""The three training-loss plots: loss 1, loss 2, and the total loss.

MeanFlow trains on two kinds of samples in every batch, and each image shows one of the
three curves below.

* **Loss 1 - flow matching** (`loss_fm`): samples with ``r = t`` (``1 - ratio``, about
  25%). The JVP correction vanishes, so the target is the instantaneous velocity
  ``v = eps - x``: ordinary flow matching.
* **Loss 2 - MeanFlow** (`loss_mf`): samples with ``r < t`` (about 75%). The target
  is the average velocity from the MeanFlow identity, ``v - (t - r) du/dt``.
* **Total loss** (`loss`): the optimized objective, the mean over all samples,
  ``total = f * loss1 + (1 - f) * loss2`` with ``f`` the logged ``fm_fraction``.

Every value is the adaptive-weighted per-image loss ``E / (sg(E) + eps)^p`` that
training minimizes.

Regenerate the images of any run with ``python -m cifar10.plots --run-dir <output_dir>``.
"""

import argparse
from array import array
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = "plots"
# (JSONL key, file name, title, subtitle)
CURVES = (
    ("loss_fm", "loss1_flow_matching.png", "Loss 1: flow matching (r = t)",
     "Target: instantaneous velocity v = ε − x"),
    ("loss_mf", "loss2_meanflow.png", "Loss 2: MeanFlow (r < t)",
     "Target: average velocity v − (t − r)·du/dt"),
    ("loss", "total_loss.png", "Total loss: training objective",
     "Mean over all samples = f·loss 1 + (1 − f)·loss 2"),
)
# Reference-palette roles: series slot 1 and a light step of its ramp, plus ink.
SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, LINE, RAW = "#e1e0d9", "#c3c2b7", "#2a78d6", "#b7d3f6"


class LossHistory:
    """Compact per-update loss history; missing values (older logs) are NaN."""

    KEYS = tuple(curve[0] for curve in CURVES)

    def __init__(self):
        self.steps = array("q")
        self.values = {key: array("d") for key in self.KEYS}

    def __len__(self):
        return len(self.steps)

    def append(self, record):
        self.steps.append(int(record["step"]))
        for key in self.KEYS:
            value = record.get(key)
            self.values[key].append(math.nan if value is None else float(value))

    @classmethod
    def from_jsonl(cls, path, max_step=None):
        """Read ``train.jsonl``; tolerate a truncated trailing line."""
        history = cls()
        path = Path(path)
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if max_step is None or record["step"] <= max_step:
                        history.append(record)
        return history


def rolling_mean(values, window):
    """Trailing mean over the last ``window`` updates, ignoring NaN entries."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))])
    counts = np.concatenate([[0], np.cumsum(finite)])
    end = np.arange(1, len(x) + 1)
    start = np.maximum(end - window, 0)
    count = counts[end] - counts[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (sums[end] - sums[start]) / count
    mean[count == 0] = np.nan
    return mean


def smoothing_window(length):
    """100 updates, widening to at most 1,000 so long runs stay readable."""
    return min(1000, max(100, length // 500))


def save_loss_plots(output_dir, history, *, divergence_loss=0.0):
    """Write the three PNGs atomically to ``<output_dir>/plots``; return written paths.

    A curve without any finite value (a log from before these keys existed) is
    skipped. Raises ImportError when matplotlib is unavailable.
    """
    # The container's HOME is not writable; keep matplotlib's font cache in the repo.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    directory = Path(output_dir) / PLOT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(history.steps, dtype=float)
    window = smoothing_window(len(history))
    written = []
    for key, name, title, subtitle in CURVES:
        values = np.asarray(history.values[key], dtype=float)
        if not len(values) or not np.isfinite(values).any():
            continue
        smooth = rolling_mean(values, window)
        figure = Figure(figsize=(8, 4.5), dpi=150, facecolor=SURFACE, layout="constrained")
        axes = figure.subplots()
        axes.set_facecolor(SURFACE)
        axes.plot(steps, values, color=RAW, linewidth=0.8, label="Per update", zorder=1)
        axes.plot(steps, smooth, color=LINE, linewidth=2, solid_joinstyle="round",
                  solid_capstyle="round", label=f"Rolling mean over {window:,} updates", zorder=2)
        last = np.flatnonzero(np.isfinite(smooth))
        if len(last):  # Direct label on the line's end only.
            axes.annotate(f"{smooth[last[-1]]:.3f}", xy=(steps[last[-1]], smooth[last[-1]]),
                          xytext=(-2, 8), textcoords="offset points", ha="right", va="bottom",
                          fontsize=9, color=INK, annotation_clip=False,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        if key == "loss" and divergence_loss > 0:
            axes.axhline(divergence_loss, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
            # Right end, below the line: early curves cross the threshold on the left.
            axes.annotate(f"Divergence guard ({divergence_loss:g})", xy=(1, divergence_loss),
                          xycoords=("axes fraction", "data"), xytext=(-4, -4), textcoords="offset points",
                          ha="right", va="top", fontsize=8, color=INK_SECONDARY)
        axes.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=20)
        axes.annotate(f"{subtitle} · updates {int(steps[0]):,}–{int(steps[-1]):,}", xy=(0, 1),
                      xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                      fontsize=9, color=INK_SECONDARY, va="bottom")
        axes.set_xlabel("Optimizer update", color=INK_SECONDARY, fontsize=10)
        axes.set_ylabel("Adaptive-weighted loss", color=INK_SECONDARY, fontsize=10)
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axes.tick_params(colors=AXIS, labelcolor=MUTED, labelsize=9)
        axes.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(AXIS)
        legend = axes.legend(loc="upper right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
        path = directory / name
        staging = directory / f".{name}.tmp"
        figure.savefig(staging, format="png", facecolor=SURFACE)
        staging.replace(path)
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write loss 1, loss 2, and total-loss plots for a run.")
    parser.add_argument("--run-dir", required=True, help="Training output directory containing train.jsonl")
    parser.add_argument("--divergence-loss", type=float,
                        help="Guard threshold line on the total-loss plot (default: the run's config.json)")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    history = LossHistory.from_jsonl(run / "train.jsonl")
    if not len(history):
        parser.error(f"No training records in {run / 'train.jsonl'}")
    threshold = args.divergence_loss
    if threshold is None:
        config = run / "config.json"
        threshold = json.loads(config.read_text()).get("divergence_loss", 0.0) if config.is_file() else 0.0
    written = save_loss_plots(run, history, divergence_loss=threshold or 0.0)
    for path in written:
        print(path)
    missing = [name for key, name, _, _ in CURVES if run / PLOT_DIR / name not in written]
    if missing:
        print(f"Skipped (no values in this log): {', '.join(missing)}")


if __name__ == "__main__":
    main()
