"""Quick matplotlib figures for the toy problem, written as PNGs into distillation/toy/figures/.

Sample tensors may be actions [N, H, D] or real vectors [N, 4]; plots use step-1 coordinates. `plot_samples`
scatters student vs. teacher samples per task, `plot_dmd_arrows` draws -g (the direction a gradient step moves each
student sample, i.e. toward the teacher) on student samples, `plot_loss_curves` plots the "train" rows of one
training log and `plot_metric_curves` plots the "eval" rows of one or more logs with the reference levels dashed.
Every function returns the path of the PNG it wrote.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from distillation.toy.problem import REAL_D, to_real

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
TRAIN_KEYS = ("loss_dmd", "loss_fm", "loss_fake", "vdiff_norm", "grad_norm")
METRIC_KEYS = ("w2", "coverage", "success")
REFERENCE_STYLES = {"noise_floor": ":", "teacher_4step": "--"}


def step1_coords(a: Tensor) -> np.ndarray:
    """Step-1 coordinates [N, 2] as numpy from actions [N, H, D] or real vectors [N, 4]."""
    x = to_real(a) if a.dim() == 3 else a
    return x[:, :REAL_D].detach().double().cpu().numpy()


def _save(fig: Figure, name: str, out_dir: Path) -> Path:
    """Save `fig` as <out_dir>/<name>.png and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def _task_axes(num_tasks: int) -> tuple[Figure, np.ndarray]:
    """A new figure with a row of square axes, one per task."""
    fig = Figure(figsize=(4.5 * num_tasks, 4.5))
    return fig, fig.subplots(1, num_tasks, squeeze=False)[0]


def plot_samples(
    student: Mapping[int, Tensor], teacher: Mapping[int, Tensor], name: str, title: str = "", out_dir: Path = FIGURES_DIR
) -> Path:
    """Scatter student vs. teacher samples per task in step-1 coordinates."""
    fig, axes = _task_axes(len(teacher))
    for ax, task_id in zip(axes, sorted(teacher)):
        ax.scatter(*step1_coords(teacher[task_id]).T, s=3, alpha=0.35, label="teacher")
        ax.scatter(*step1_coords(student[task_id]).T, s=3, alpha=0.35, label="student")
        ax.set(title=f"task {task_id}", xlabel="step-1 x", ylabel="step-1 y", aspect="equal")
        ax.legend(markerscale=4, loc="upper right")
    fig.suptitle(title)
    return _save(fig, name, out_dir)


def plot_dmd_arrows(
    a_student: Mapping[int, Tensor],
    g: Mapping[int, Tensor],
    teacher: Mapping[int, Tensor],
    name: str,
    title: str = "",
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Arrows of -g (the DMD update direction) at student samples over teacher samples, per task (step 1)."""
    fig, axes = _task_axes(len(teacher))
    for ax, task_id in zip(axes, sorted(teacher)):
        ax.scatter(*step1_coords(teacher[task_id]).T, s=3, alpha=0.25, color="tab:blue", label="teacher")
        pos, step = step1_coords(a_student[task_id]), -step1_coords(g[task_id])
        ax.quiver(pos[:, 0], pos[:, 1], step[:, 0], step[:, 1], color="tab:red", angles="xy", width=0.003)
        ax.set(title=f"task {task_id}: -g on student samples", xlabel="step-1 x", ylabel="step-1 y", aspect="equal")
        ax.legend(markerscale=4, loc="upper right")
    fig.suptitle(title)
    return _save(fig, name, out_dir)


def plot_loss_curves(log: Mapping, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    """One panel per logged training quantity (losses, ||v_fake - v_teacher||, grad norm) of one training log."""
    keys = [key for key in TRAIN_KEYS if any(key in row for row in log["train"])]
    if not keys:
        raise ValueError(f"training log has none of the keys {TRAIN_KEYS}")
    fig = Figure(figsize=(3.6 * len(keys), 3.2), layout="tight")
    for ax, key in zip(fig.subplots(1, len(keys), squeeze=False)[0], keys):
        rows = [row for row in log["train"] if key in row]
        values = [row[key] for row in rows]
        ax.plot([row["step"] for row in rows], values)
        ax.set(title=key, xlabel="step", yscale="log" if min(values) > 0 else "linear")
    fig.suptitle(log.get("config", {}).get("run", ""))
    return _save(fig, name, out_dir)


def _plot_reference(ax: Axes, reference: Mapping, task: str, metric: str) -> None:
    """Horizontal lines at the noise-floor and 4-step-teacher levels of one metric."""
    for level, style in REFERENCE_STYLES.items():
        if level in reference:
            ax.axhline(reference[level][task][metric], color="gray", linestyle=style, label=level)


def plot_metric_curves(logs: Mapping[str, Mapping], name: str, out_dir: Path = FIGURES_DIR) -> Path:
    """W2, coverage and success per task over training for runs {label: log}, reference levels dashed."""
    first = next(iter(logs.values()))
    tasks = sorted(first["eval"][0]["metrics"])
    fig = Figure(figsize=(4.2 * len(METRIC_KEYS), 3.2 * len(tasks)), layout="tight")
    axes = fig.subplots(len(tasks), len(METRIC_KEYS), squeeze=False)
    for row, task in enumerate(tasks):
        for col, metric in enumerate(METRIC_KEYS):
            ax = axes[row, col]
            for label, log in logs.items():
                steps = [entry["step"] for entry in log["eval"]]
                ax.plot(steps, [entry["metrics"][task][metric] for entry in log["eval"]], marker=".", label=label)
            _plot_reference(ax, first.get("reference", {}), task, metric)
            ax.set(title=f"task {task}: {metric}", xlabel="step")
            if metric == "coverage":
                ax.set_ylim(-0.2, first["eval"][0]["metrics"][task]["num_modes"] + 0.2)
    axes[0, -1].legend(fontsize="small")
    return _save(fig, name, out_dir)
