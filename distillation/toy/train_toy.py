"""Distil the pre-trained toy student into the exact teachers with the shared MTD update (CPU only).

Run from the project root, e.g.
  python -m distillation.toy.train_toy --run dmd_fm --alpha-dmd 0.5 --alpha-fm 0.5 --lambda-ida 0.99 --k-fake 1
The student starts from runs/student_init.pt, the fake network is a copy of it, and both use Adam. Every update calls
the shared `distill_step` on a fresh batch (tasks 50/50, o ~ U[-1, 1]) drawn from the training generator (--seed).
A training row (the mean of each metric over the last LOG_EVERY updates) is logged every LOG_EVERY updates, and the
student is evaluated at step 0 and every --eval-every updates. Every evaluation, and the reference levels, use a fresh
generator seeded with EVAL_SEED: common random numbers, so every evaluation of every run sees the same noise z, exact
teacher samples and projections. At n = 2,000 the noise floor is close to the target W2, so the initial student
(before training), the final student and the reference levels are also evaluated with FINAL_N samples per (task, o).
A non-finite training or evaluation metric stops the run cleanly: the step is recorded under "diverged", the bad
row is not logged, and the final student is not re-evaluated. wall_time_s covers the updates and the periodic
evaluations. The JSON log is written to runs/<run>.json and the final student and fake state dicts to runs/<run>.pt.
"""

import argparse
import copy
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from distillation.toy.gmm_teacher import GMMTeacher, make_teachers
from distillation.toy.metrics import evaluate, reference_levels, student_sampler
from distillation.toy.mlp_student import MLPVelocity
from distillation.toy.problem import sample_batch

RUNS_DIR = Path(__file__).resolve().parent / "runs"
INIT_CHECKPOINT = RUNS_DIR / "student_init.pt"
EVAL_SEED = 2026
EVAL_N = 2000
FINAL_N = 20000
LOG_EVERY = 10


def parse_args() -> argparse.Namespace:
    """Command-line options of one distillation run (defaults: DMD + FM, lambda_IDA 0.99, K_psi 1)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name; writes runs/<run>.json and runs/<run>.pt")
    parser.add_argument("--alpha-dmd", type=float, default=0.5)
    parser.add_argument("--alpha-fm", type=float, default=0.5)
    parser.add_argument("--lambda-ida", type=float, default=0.99)
    parser.add_argument("--k-fake", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_student(path: Path, key: str = "state_dict") -> MLPVelocity:
    """An MLPVelocity with the state dict `key` of a checkpoint (default: the pre-trained student of pretrain.py)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m distillation.toy.pretrain` first")
    student = MLPVelocity()
    student.load_state_dict(torch.load(path, map_location="cpu")[key])
    return student


def eval_generator() -> torch.Generator:
    """A fresh generator seeded with EVAL_SEED, so every evaluation draws the same random numbers."""
    return torch.Generator().manual_seed(EVAL_SEED)


def evaluate_student(
    student: nn.Module, teachers: Mapping[int, GMMTeacher], num_steps: int, n: int = EVAL_N
) -> dict[str, dict[str, Any]]:
    """Metrics of the student's num_steps FlowSolve samples vs. the exact teachers (common random numbers)."""
    return evaluate(student_sampler(student, num_steps), teachers, eval_generator(), n=n)


def evaluate_references(teachers: Mapping[int, GMMTeacher], num_steps: int, n: int) -> dict[str, Any]:
    """Noise floor and num_steps teacher with n samples per (task, o), from the evaluation seed."""
    return reference_levels(teachers, eval_generator(), num_steps, n=n)


def non_finite_keys(values: Mapping[str, float]) -> list[str]:
    """Names of the non-finite values."""
    return sorted(key for key, value in values.items() if not math.isfinite(value))


def window_mean(rows: list[dict[str, float]]) -> dict[str, float]:
    """Per-key mean of a list of distill_step metric dicts (all with the same keys)."""
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def train(
    student: nn.Module, fake: nn.Module, teachers: Mapping[int, GMMTeacher], cfg: DistillConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run args.steps MTD updates; return the train rows, eval rows and divergence record (None if finite)."""
    router = TeacherRouter(dict(teachers))
    opt_student = torch.optim.Adam(student.parameters(), lr=args.lr)
    opt_fake = torch.optim.Adam(fake.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    log: dict[str, Any] = {"train": [], "eval": [], "diverged": None}
    log["eval"].append({"step": 0, "metrics": evaluate_student(student, teachers, cfg.num_flow_steps)})
    window: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        batch = sample_batch(args.batch_size, generator)
        metrics = distill_step(student, fake, router, batch, cfg, opt_student, opt_fake, generator)
        if non_finite_keys(metrics):
            log["diverged"] = {"step": step, "keys": non_finite_keys(metrics)}
            break
        window.append(metrics)
        if step % LOG_EVERY == 0:
            log["train"].append({"step": step, **window_mean(window)})
            window = []
        if step % args.eval_every == 0:
            evaluation = evaluate_student(student, teachers, cfg.num_flow_steps)
            if non_finite_keys({task: m["w2"] for task, m in evaluation.items()}):
                log["diverged"] = {"step": step, "keys": ["eval_w2"]}
                break
            log["eval"].append({"step": step, "metrics": evaluation})
            print(f"[{args.run}] step {step}: {json.dumps(evaluation)}", flush=True)
    return log


def config_record(args: argparse.Namespace, cfg: DistillConfig) -> dict[str, Any]:
    """The run configuration as stored in the log."""
    return {"run": args.run, "alpha_dmd": cfg.alpha_dmd, "alpha_fm": cfg.alpha_fm, "lambda_ida": cfg.lambda_ida,
            "k_fake": cfg.k_fake, "num_flow_steps": cfg.num_flow_steps, "steps": args.steps,
            "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed, "eval_every": args.eval_every,
            "eval_seed": EVAL_SEED, "log_every": LOG_EVERY}


def main() -> None:
    """Distil, evaluate and write runs/<run>.json and runs/<run>.pt."""
    args = parse_args()
    cfg = DistillConfig(alpha_dmd=args.alpha_dmd, alpha_fm=args.alpha_fm, k_fake=args.k_fake,
                        lambda_ida=args.lambda_ida)
    teachers = make_teachers()
    student = load_student(INIT_CHECKPOINT)
    fake = copy.deepcopy(student)
    initial_large = evaluate_student(student, teachers, cfg.num_flow_steps, FINAL_N)
    start = time.time()
    log = train(student, fake, teachers, cfg, args)
    wall_time_s = time.time() - start
    final_large = None if log["diverged"] else evaluate_student(student, teachers, cfg.num_flow_steps, FINAL_N)
    log = {"config": config_record(args, cfg), **log,
           "reference": evaluate_references(teachers, cfg.num_flow_steps, EVAL_N),
           "final_n20000": {"student": final_large, "initial_student": initial_large,
                            "reference": evaluate_references(teachers, cfg.num_flow_steps, FINAL_N)},
           "wall_time_s": wall_time_s}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{args.run}.json").write_text(json.dumps(log, indent=1))
    torch.save({"student": student.state_dict(), "fake": fake.state_dict(), "config": log["config"]},
               RUNS_DIR / f"{args.run}.pt")
    print(f"[{args.run}] done in {wall_time_s:.1f} s, diverged: {log['diverged']}", flush=True)


if __name__ == "__main__":
    main()
