"""Figures and tables of the Stage 3a toy runs from the logs: `python -m distillation.toy.report [--out-dir DIR]`.

Reads runs/<run>.json and .pt, writes the PNGs via plots.py (default figures/) and prints the RESULTS.md tables.
Pass (spec Section 6), per task: W2 reduction >= 80% from the initial student (read literally at n = 2,000, at
n = 2,000 floor-corrected as (W2_init - W2_final) / (W2_init - floor), and literally at n = 20,000), every mode
covered, success >= 4-step teacher success - 0.05. DMD arrows: mean g over DRAWS (tau, eps) draws, fake = student.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import dmd_direction_at_noisy
from distillation.toy.gmm_teacher import make_teachers
from distillation.toy.metrics import exact_sampler, student_sampler
from distillation.toy.plots import FIGURES_DIR, plot_dmd_arrows, plot_loss_curves, plot_metric_curves, plot_samples
from distillation.toy.problem import TASK_IDS, action_mask, make_cond
from distillation.toy.train_toy import EVAL_SEED, INIT_CHECKPOINT, RUNS_DIR, load_student

TASKS = tuple(str(task_id) for task_id in TASK_IDS)
VARIANTS = {"fm_only": "FM only", "dmd_only": "DMD only", "dmd_fm": "DMD + FM"}
SWEEP = ("dmd_fm_lam1_k1", "dmd_fm", "dmd_fm_lam0.9_k1", "dmd_fm_lam0_k1",
         "dmd_fm_lam1_k5", "dmd_fm_lam0.99_k5", "dmd_fm_lam0.9_k5", "dmd_fm_lam0_k5")
SAMPLES_PER_TASK, ARROWS_PER_TASK, DRAWS = 2000, 300, 32
PASS_REDUCTION, SUCCESS_MARGIN = 0.8, 0.05
LEVELS = ("noise floor", "4-step teacher", "initial student")
METRIC_HEADER = [f"{name} t{t}" for name in ("W2", "cov", "succ") for t in TASKS]


def load_log(run: str) -> dict[str, Any]:
    """The JSON training log of one run."""
    return json.loads((RUNS_DIR / f"{run}.json").read_text())


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def metric_cells(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """W2, coverage and success cells, each for every task."""
    return ([f"{metrics[t]['w2']:.3f}" for t in TASKS] + [f"{metrics[t]['coverage']}/{metrics[t]['num_modes']}"
            for t in TASKS] + [f"{metrics[t]['success']:.3f}" for t in TASKS])


def reference_table(log: Mapping[str, Any]) -> str:
    """Noise floor, 4-step teacher and initial student at n = 2,000 and n = 20,000."""
    by_n = [("2,000", log["reference"], log["eval"][0]["metrics"]),
            ("20,000", log["final_n20000"]["reference"], log["final_n20000"]["initial_student"])]
    rows = [[name, n, *metric_cells(metrics)] for n, levels, init in by_n
            for name, metrics in zip(LEVELS, (levels["noise_floor"], levels["teacher_4step"], init))]
    return md_table(["level", "n", *METRIC_HEADER], rows)


def run_table(runs: list[str]) -> str:
    """Final metrics (n = 2,000), final W2 at n = 20,000 and wall time of each run."""
    rows = []
    for run in runs:
        log = load_log(run)
        cfg, large = log["config"], log["final_n20000"]["student"]
        w2_large = [f"{large[t]['w2']:.3f}" if large else "-" for t in TASKS]
        non_finite = f"step {log['diverged']['step']}" if log["diverged"] else "no"
        rows.append([run, f"{cfg['alpha_dmd']:g} / {cfg['alpha_fm']:g}", f"{cfg['lambda_ida']:g}", str(cfg["k_fake"]),
                     *metric_cells(log["eval"][-1]["metrics"]), *w2_large, f"{log['wall_time_s']:.0f}", non_finite])
    return md_table(["run", "α DMD / FM", "λ_IDA", "K_ψ", *METRIC_HEADER, *[f"W2@20k t{t}" for t in TASKS], "wall s",
                     "non-finite"], rows)


def verdict_table(log: Mapping[str, Any]) -> str:
    """Pass criterion per (W2 reading, task): W2 reduction, coverage, success vs. 4-step teacher - 0.05."""
    init, final, large = log["eval"][0]["metrics"], log["eval"][-1]["metrics"], log["final_n20000"]
    readings = [("n=2,000 literal", init, final, None, log["reference"]),
                ("n=2,000 floor-corrected", init, final, log["reference"]["noise_floor"], log["reference"]),
                ("n=20,000 literal", large["initial_student"], large["student"], None, large["reference"])]
    rows = []
    for name, start, end, floor, levels in readings:
        for t in TASKS:
            floor_w2 = floor[t]["w2"] if floor else 0.0
            reduction = (start[t]["w2"] - end[t]["w2"]) / (start[t]["w2"] - floor_w2)
            bar = levels["teacher_4step"][t]["success"] - SUCCESS_MARGIN
            ok = reduction >= PASS_REDUCTION and end[t]["coverage"] == end[t]["num_modes"] and end[t]["success"] >= bar
            rows.append([name, t, f"{start[t]['w2']:.3f}", f"{end[t]['w2']:.3f}", f"{floor_w2:.3f}" if floor else "-",
                         f"{100 * reduction:.0f}%", f"{end[t]['coverage']}/{end[t]['num_modes']}",
                         f"{end[t]['success']:.3f}", f"{bar:.3f}", "pass" if ok else "FAIL"])
    return md_table(["W2 reading", "task", "W2 init", "W2 final", "floor", "reduction", "cov", "succ",
                     "teacher − 0.05", "verdict"], rows)


def task_cond(task_id: int, n: int) -> dict[str, torch.Tensor]:
    """Toy cond of n samples of one task at o = 0."""
    return make_cond(torch.zeros(n), torch.full((n,), task_id))


def sample_figures(runs: tuple[str, ...], out_dir: Path) -> None:
    """Initial and final students of `runs` vs. exact teacher samples, per task, at o = 0 (the same noise for all)."""
    exact, conds = exact_sampler(make_teachers()), {k: task_cond(k, SAMPLES_PER_TASK) for k in TASK_IDS}
    finals = {run: load_student(RUNS_DIR / f"{run}.pt", "student") for run in runs}
    for name, student in {"init": load_student(INIT_CHECKPOINT), **finals}.items():
        generator = torch.Generator().manual_seed(EVAL_SEED)
        a_student = {k: student_sampler(student)(conds[k], SAMPLES_PER_TASK, generator) for k in TASK_IDS}
        a_exact = {k: exact(conds[k], SAMPLES_PER_TASK, generator) for k in TASK_IDS}
        plot_samples(a_student, a_exact, f"samples_{name}", f"{name}: 4-step student vs. exact teacher, o = 0", out_dir)


def arrow_figure(out_dir: Path) -> None:
    """Mean DMD direction over DRAWS (tau, eps) draws on initial-student samples, fake = student."""
    teachers, student = make_teachers(), load_student(INIT_CHECKPOINT)
    router, generator = TeacherRouter(dict(teachers)), torch.Generator().manual_seed(EVAL_SEED)
    a_student, g, a_exact = {}, {}, {}
    for k in TASK_IDS:
        cond, mask = task_cond(k, ARROWS_PER_TASK), action_mask(ARROWS_PER_TASK)
        a_student[k] = student_sampler(student)(cond, ARROWS_PER_TASK, generator)
        draws = [dmd_direction_at_noisy(student, router, a_student[k], cond, cond["task"], mask, DistillConfig.norm_eps,
                                        generator)[0] for _ in range(DRAWS)]
        g[k] = torch.stack(draws).mean(dim=0)
        a_exact[k] = exact_sampler(teachers)(task_cond(k, SAMPLES_PER_TASK), SAMPLES_PER_TASK, generator)
    plot_dmd_arrows(a_student, g, a_exact, "dmd_arrows_init", f"initial student, mean of {DRAWS} (τ, ε) draws, o = 0",
                    out_dir)


def main() -> None:
    """Write every figure and print every table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR, help="directory for the PNGs (default: figures/)")
    out_dir = parser.parse_args().out_dir
    plot_metric_curves({label: load_log(run) for run, label in VARIANTS.items()}, "metrics_variants", out_dir)
    sweep = {f"λ={log['config']['lambda_ida']:g}, K={log['config']['k_fake']}": log for log in map(load_log, SWEEP)}
    plot_metric_curves(sweep, "metrics_ida_sweep", out_dir)
    stable = {label: log for label, log in sweep.items() if log["config"]["lambda_ida"] >= 0.99}
    plot_metric_curves(stable, "metrics_ida_stable", out_dir)
    plot_loss_curves(load_log("dmd_fm"), "losses_dmd_fm", out_dir)
    sample_figures((*VARIANTS, "dmd_fm_lam1_k5", "dmd_fm_lam0_k1"), out_dir)
    arrow_figure(out_dir)
    print("## Reference levels\n" + reference_table(load_log("dmd_fm")))
    print("\n## Variants\n" + run_table(list(VARIANTS)))
    print("\n## IDA sweep\n" + run_table(list(SWEEP)))
    for run in ("dmd_fm", "dmd_fm_lam1_k5", "dmd_fm_lam0.99_k5"):
        print(f"\n## Pass verdict ({run})\n" + verdict_table(load_log(run)))
    print(f"\nTotal training wall time, 10 runs: {sum(load_log(r)['wall_time_s'] for r in {*VARIANTS, *SWEEP}):.0f} s")


if __name__ == "__main__":
    main()
