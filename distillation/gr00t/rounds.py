"""Distillation rounds of one run, from the project root in a2u-policy, on the GPU:
  flock -w 7200 <gpu.lock> conda run --no-capture-output -n a2u-policy python -m distillation.gr00t.rounds \
      --config configs/distillation/<run>.yaml
The config's K teachers each own one role of its task set (`pairing`); everything below loops over those roles.
Selection of the initial student, then per round r: collect (round r−1's export on every role; at r = 0 the initial
student on every role and each teacher on its own role) → train steps_per_round updates on the whole buffer (models
stay in memory; state/round_<r>/ is saved) → export to checkpoints/students/<run_id>/round_<r>/ → selection
(run_baseline per role, ≤ 2 at a time by RAM) → log; stop when no role's SR beat its best (initial SR included) for
`patience` rounds in a row, or at max_rounds; the best mean-SR round is recorded. Re-runs skip finished stages.
Collections and selections of fixed checkpoints are shared between runs (distillation/runs/{collections,selection}/)."""

import argparse
import json
import os
import shutil
import signal
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from distillation.gr00t.buffer import ObservationBuffer
from distillation.gr00t.collect import DEFAULT_PORT, Job, collect, parallel_slots, run_parallel, sim_command
from distillation.gr00t.export import export_student
from distillation.gr00t.train import Distiller, load_config, project_path, teacher_roles, unset_keys
from evaluation.config import PROJECT_ROOT, load_pairing
from expert_training.wandb_log_eval import load_dotenv

RUNS_ROOT = PROJECT_ROOT / "distillation" / "runs"
SHARED_SELECTION = RUNS_ROOT / "selection"
STUDENTS_ROOT = PROJECT_ROOT / "checkpoints" / "students"
SELECTION_SEEDS = range(1000, 1020)


def check_config(cfg: dict[str, Any]) -> None:
    """Raise on unset placeholders, selection seeds outside 1000–1019, or teachers that are not one per distinct role
    of the task set with that role's task name."""
    sel, roles = cfg["selection"], teacher_roles(cfg)
    errors = [f"{key} is unset" for key in unset_keys(cfg)]
    if not set(range(sel["seed_start"], sel["seed_start"] + sel["num_episodes"])) <= set(SELECTION_SEEDS):
        errors.append(f"selection seeds must lie in {SELECTION_SEEDS}")
    if not roles or len(set(roles)) != len(roles):
        errors.append(f"teachers need one or more distinct roles, got {roles}")
    tasks = load_pairing(cfg["pairing"])
    errors += [f"teacher task {t['task']!r} is not role {t['role']!r} of {cfg['pairing']}" for t in cfg["teachers"]
               if t["role"] not in tasks or tasks[t["role"]].name != t["task"]]
    if errors:
        raise ValueError(f"config {cfg['run_id']}: " + "; ".join(errors))


def stalled(history: Sequence[Mapping[str, float]], patience: int) -> bool:
    """True when no entry among the last `patience` exceeds, on any key, the best of all earlier entries (higher is
    better); never with patience 0 or too short a history."""
    if patience < 1 or len(history) <= patience:
        return False
    earlier, recent = history[:-patience], history[-patience:]
    return not any(entry[key] > max(e[key] for e in earlier) for entry in recent for key in entry)


def round_jobs(cfg: dict[str, Any], r: int) -> list[Job]:
    """Round r's collections: the current student on every role; in round 0 also each teacher on its role."""
    spec, run_id, student = cfg["collection"], cfg["run_id"], cfg["student"]
    label, checkpoint = student["label"], student["checkpoint"]
    if r > 0:
        label, checkpoint = f"{run_id}_round_{r - 1}", str(STUDENTS_ROOT / run_id / f"round_{r - 1}")
    jobs = [Job(label, role, 0, spec["student_episodes"], checkpoint) for role in teacher_roles(cfg)]
    if r == 0 and spec["teacher_episodes"]:
        jobs += [Job(t["label"], t["role"], spec["student_episodes"], spec["teacher_episodes"], t["checkpoint"])
                 for t in cfg["teachers"]]
    return jobs


def selection(cfg: dict[str, Any], checkpoint: str, results_root: Path, roles: list[str]) -> dict[str, float]:
    """Selection SR per role of `checkpoint`: run_baseline for the roles without a summary, then read all summaries."""
    sel, pairing_dir = cfg["selection"], results_root / cfg["pairing"]
    pending = [role for role in roles if not (pairing_dir / role / "summary.json").exists()]
    args = ["--pairing", cfg["pairing"], "--checkpoint", checkpoint, "--seed-start", str(sel["seed_start"]),
            "--num-episodes", str(sel["num_episodes"]), "--results-root", str(results_root), "--no-video"]
    commands = [sim_command("baseline_eval.run_baseline", *args, "--roles", role, "--launch-server", "--port",
                            str(DEFAULT_PORT + k)) for k, role in enumerate(pending)]
    run_parallel(commands, [results_root / f"{role}.log" for role in pending], parallel_slots())
    summaries = {role: json.loads((pairing_dir / role / "summary.json").read_text()) for role in roles}
    if any((s["checkpoint"], s["seed_start"], s["num_episodes"]) != (checkpoint, sel["seed_start"],
                                                                     sel["num_episodes"]) for s in summaries.values()):
        raise ValueError(f"{pairing_dir} holds results of another checkpoint or seed range for {roles}; delete them")
    return {role: s["success_rate"] for role, s in summaries.items()}


class RunLogger:
    """Rows to <run dir>/train_log.jsonl, stdout and W&B (offline unless WANDB_API_KEY is in .env, never printed)."""

    def __init__(self, run_dir: Path, cfg: dict[str, Any]) -> None:
        """Open the W&B run: project atomic2universal, group distill/<task set>, named after the run."""
        self.path = run_dir / "train_log.jsonl"
        load_dotenv(PROJECT_ROOT / ".env")
        self.run = wandb.init(project="atomic2universal", group=f"distill/{cfg['pairing']}", name=cfg["run_id"],
                              config=cfg, dir=run_dir, mode="online" if os.environ.get("WANDB_API_KEY") else "offline")
        self.run.define_metric("*", step_metric="step")

    def log(self, row: dict[str, Any]) -> None:
        """Append one row (with the global `step`) to every destination."""
        with open(self.path, "a") as file:
            file.write(json.dumps(row) + "\n")
        self.run.log(row)
        print(json.dumps(row), flush=True)


class Run:
    """One run: its directories, progress record (rounds.json), buffer and the lazily built distiller."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Create the run directory; a re-run must use the same config."""
        self.cfg, self.dir = cfg, RUNS_ROOT / cfg["run_id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        config_path, progress_path = self.dir / "config.json", self.dir / "rounds.json"
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError(f"{config_path} differs from this config; use a new run_id or delete {self.dir}")
        config_path.write_text(json.dumps(cfg, indent=1))
        self.progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"rounds": {}}
        self.logger, self.buffer = RunLogger(self.dir, cfg), ObservationBuffer()
        self.distiller, self.distiller_round = None, None

    def save_progress(self) -> None:
        """Write rounds.json atomically."""
        (self.dir / "rounds.json.tmp").write_text(json.dumps(self.progress, indent=1))
        (self.dir / "rounds.json.tmp").replace(self.dir / "rounds.json")

    def distiller_at(self, r: int) -> Distiller:
        """The distiller as after round r's training (r = −1: freshly initialized), loading saved state if needed."""
        if self.distiller is None:
            self.distiller, self.distiller_round = Distiller(self.cfg), -1
        if self.distiller_round != r:
            self.distiller_round = self.distiller.load(self.dir / "state" / f"round_{r}")["round"]
        return self.distiller

    def export(self, r: int) -> str:
        """Export the student after round r unless it exists; return its path."""
        out = STUDENTS_ROOT / self.cfg["run_id"] / f"round_{r}"
        if not out.exists():
            distiller = self.distiller_at(r)
            info = {"run_id": self.cfg["run_id"], "round": r, "step": distiller.step, "student": self.cfg["student"]}
            export_student(distiller.student, out, info, project_path(self.cfg["student"]["checkpoint"]))
        return str(out)

    def train(self, r: int) -> dict[str, Any]:
        """Round r's training from the state after round r − 1; saves state/round_<r>/."""
        distiller = self.distiller_at(r - 1)
        result = distiller.train(self.buffer, self.cfg["train"]["steps_per_round"], self.logger.log, r)
        distiller.save(self.dir / "state" / f"round_{r}", {"round": r})
        self.distiller_round = r
        return {**result, "step": distiller.step}

    def run_round(self, r: int) -> dict[str, float]:
        """Collect, train, export and select for round r (skipping finished stages); return the selection SRs."""
        cfg, record = self.cfg, self.progress["rounds"].setdefault(str(r), {})
        for directory in collect(round_jobs(cfg, r), r, cfg["pairing"]):
            self.buffer.add(directory)
        if r == 0 and cfg["train"]["holdout"]:
            self.buffer.holdout(cfg["train"]["holdout"], cfg["train"]["holdout_seed"])
        if "train" not in record:
            record["train"] = self.train(r)
            self.save_progress()
            shutil.rmtree(self.dir / "state" / f"round_{r - 1}", ignore_errors=True)
        record["selection"] = selection(cfg, self.export(r), self.dir / "selection" / f"round_{r}",
                                        teacher_roles(cfg))
        self.save_progress()
        self.logger.log({"round": r, "step": record["train"]["step"],
                         **{f"selection/{role}": sr for role, sr in record["selection"].items()}})
        return record["selection"]

    def execute(self) -> None:
        """Selection of the initial student (shared results), then rounds until the early stop or max_rounds."""
        cfg, student, roles = self.cfg, self.cfg["student"], teacher_roles(self.cfg)
        self.progress["init"] = selection(cfg, student["checkpoint"], SHARED_SELECTION / student["label"], roles)
        self.save_progress()
        history = [self.progress["init"]]
        for r in range(cfg["max_rounds"]):
            history.append(self.run_round(r))
            if stalled(history, cfg["patience"]):
                break
        best = int(np.argmax([np.mean(list(sr.values())) for sr in history[1:]]))
        self.progress.update(best_round=best, rounds_run=len(history) - 1)
        self.save_progress()


def main() -> None:
    """Run (or resume) the configured run; SIGTERM/SIGHUP exit cleanly, so subprocesses and their servers stop."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    check_config(cfg := load_config(parser.parse_args().config))
    for signal_number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_number, lambda number, frame: sys.exit(128 + number))
    Run(cfg).execute()


if __name__ == "__main__":
    main()
