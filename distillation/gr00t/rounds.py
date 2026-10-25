"""Distillation rounds of one run (spec §5, Levels 1–3), from the project root in a2u-policy, on the GPU:
  flock -w 7200 <gpu.lock> conda run --no-capture-output -n a2u-policy python -m distillation.gr00t.rounds \
      --config configs/distillation/<level>.yaml [--variant <name>]
Selection of the initial student (exported if perturbed) and reference policies, then per round r: collect (round r−1's
export, or the data policy and each teacher at r = 0) → train steps_per_round updates on the whole buffer (models stay
in memory; state/round_<r>/ is saved) → export to checkpoints/students/<run_id>/round_<r>/ → selection (run_baseline per
role, ≤ 2 at a time by RAM) → log; stop when no role's SR beat its best (initial SR included) for 2 rounds in a row, or
at max_rounds; the best mean-SR round is evaluated on the extra roles. Re-runs skip finished stages. Collections and
selections of fixed checkpoints are shared between runs (distillation/runs/{collections,selection}/)."""

import argparse
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from distillation.gr00t.buffer import ObservationBuffer
from distillation.gr00t.collect import DEFAULT_PORT, Job, collect, parallel_slots, run_parallel, sim_command
from distillation.gr00t.export import export_student
from distillation.gr00t.train import Distiller, OfflineMetric, checkpoint_path, load_config, stalled, unset_keys
from evaluation.config import PROJECT_ROOT
from expert_training.wandb_log_eval import load_dotenv

RUNS_ROOT = PROJECT_ROOT / "distillation" / "runs"
SHARED_SELECTION = RUNS_ROOT / "selection"
STUDENTS_ROOT = PROJECT_ROOT / "checkpoints" / "students"
SELECTION_SEEDS = range(1000, 1020)


def check_config(cfg: dict[str, Any]) -> None:
    """Raise on unset placeholders, selection seeds outside 1000–1019, or early stopping without a holdout."""
    sel, train = cfg["selection"], cfg["train"]
    errors = [f"{key} is unset" for key in unset_keys(cfg)]
    if not set(range(sel["seed_start"], sel["seed_start"] + sel["num_episodes"])) <= set(SELECTION_SEEDS):
        errors.append(f"selection seeds must lie in {SELECTION_SEEDS}")
    if train["patience"] and not train["holdout"]:
        errors.append("train.patience > 0 needs train.holdout > 0")
    if errors:
        raise ValueError(f"config {cfg['run_id']}: " + "; ".join(errors))


def round_jobs(cfg: dict[str, Any], r: int) -> list[Job]:
    """Round r's collections: the current student on every role; in round 0 also each teacher on its role."""
    spec, run_id = cfg["collection"], cfg["run_id"]
    label, checkpoint = spec["policy"], cfg["checkpoints"][spec["policy"]]
    if r > 0:
        label, checkpoint = f"{run_id}_round_{r - 1}", str(STUDENTS_ROOT / run_id / f"round_{r - 1}")
    jobs = [Job(label, role, 0, spec["student_episodes"], checkpoint) for role in cfg["roles"]]
    if r == 0 and spec["teacher_episodes"]:
        jobs += [Job(teacher, role, spec["student_episodes"], spec["teacher_episodes"], cfg["checkpoints"][teacher])
                 for role, teacher in cfg["teachers"].items()]
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
        """Open the W&B run: project atomic2universal, group distill/<level>, named after the run."""
        self.path = run_dir / "train_log.jsonl"
        load_dotenv(PROJECT_ROOT / ".env")
        self.run = wandb.init(project="atomic2universal", group=f"distill/{cfg['level']}", name=cfg["run_id"],
                              config=cfg, dir=run_dir, mode="online" if os.environ.get("WANDB_API_KEY") else "offline")
        self.run.define_metric("*", step_metric="step")

    def log(self, row: dict[str, Any]) -> None:
        """Append one row (with the global `step`) to every destination."""
        with open(self.path, "a") as file:
            file.write(json.dumps(row) + "\n")
        self.run.log(row)
        print(json.dumps(row), flush=True)


class Run:
    """One run: its directories, progress record (rounds.json), buffer, holdout and the lazily built distiller."""

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
        self.held, self.metric, self.distiller, self.distiller_round = None, None, None, None

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

    def export(self, r: int | None) -> str:
        """Export the student after round r (None: the initial student) unless it exists; return its path."""
        out = STUDENTS_ROOT / self.cfg["run_id"] / ("init" if r is None else f"round_{r}")
        if not out.exists():
            distiller = self.distiller_at(-1 if r is None else r)
            info = {"run_id": self.cfg["run_id"], "round": r, "step": distiller.step, "student": self.cfg["student"]}
            export_student(distiller.student, out, info, checkpoint_path(self.cfg, self.cfg["student"]["init"]))
        return str(out)

    def initial_selection(self) -> dict[str, float]:
        """Selection SR of the initial student on the roles and extra roles; also of each reference policy."""
        cfg, init, perturbed = self.cfg, self.cfg["student"]["init"], self.cfg["student"]["perturb_scale"] > 0
        self.progress["reference"] = {label: selection(cfg, cfg["checkpoints"][label], SHARED_SELECTION / label,
                                      cfg["roles"]) for label in cfg["selection"]["reference_policies"]}
        return selection(cfg, self.export(None) if perturbed else cfg["checkpoints"][init],
                         self.dir / "selection" / "init" if perturbed else SHARED_SELECTION / init,
                         cfg["roles"] + cfg["selection"]["extra_roles"])

    def train(self, r: int) -> dict[str, Any]:
        """Round r's training from the state after round r − 1; saves state/round_<r>/."""
        train_cfg, distiller = self.cfg["train"], self.distiller_at(r - 1)
        if self.held is not None and self.metric is None:
            self.metric = OfflineMetric(distiller, *self.held, train_cfg["holdout_seed"], train_cfg["batch_size"])
        result = distiller.train(self.buffer, train_cfg["steps_per_round"], self.logger.log, self.metric, r)
        distiller.save(self.dir / "state" / f"round_{r}", {"round": r})
        self.distiller_round = r
        return {**result, "step": distiller.step}

    def run_round(self, r: int) -> dict[str, float]:
        """Collect, train, export and select for round r (skipping finished stages); return the selection SRs."""
        cfg, record = self.cfg, self.progress["rounds"].setdefault(str(r), {})
        for directory in collect(round_jobs(cfg, r), r, cfg["pairing"]):
            self.buffer.add(directory)
        if r == 0 and cfg["train"]["holdout"]:
            self.held = self.buffer.holdout(cfg["train"]["holdout"], cfg["train"]["holdout_seed"])
        if "train" not in record:
            record["train"] = self.train(r)
            self.save_progress()
            shutil.rmtree(self.dir / "state" / f"round_{r - 1}", ignore_errors=True)
        record["selection"] = selection(cfg, self.export(r), self.dir / "selection" / f"round_{r}", cfg["roles"])
        self.save_progress()
        self.logger.log({"round": r, "step": record["train"]["step"],
                         **{f"selection/{role}": sr for role, sr in record["selection"].items()}})
        return record["selection"]

    def execute(self) -> None:
        """Initial selection, rounds until the early stop or max_rounds, then the best round's extra roles."""
        cfg, roles = self.cfg, self.cfg["roles"]
        self.progress["init"] = self.initial_selection()
        self.save_progress()
        history = [{role: self.progress["init"][role] for role in roles}]
        for r in range(cfg["max_rounds"]):
            history.append(self.run_round(r))
            if stalled(history, patience=2):
                break
        best = int(np.argmax([np.mean([sr[role] for role in roles]) for sr in history[1:]]))
        self.progress.update(best_round=best, rounds_run=len(history) - 1, best_extra_roles=selection(
            cfg, self.export(best), self.dir / "selection" / f"round_{best}", cfg["selection"]["extra_roles"]))
        self.save_progress()


def main() -> None:
    """Run (or resume) the configured run; SIGTERM/SIGHUP exit cleanly, so subprocesses and their servers stop."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", help="variant of the config (level2_single: dmd_fm or fm_only)")
    args = parser.parse_args()
    check_config(cfg := load_config(args.config, args.variant))
    for signal_number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_number, lambda number, frame: sys.exit(128 + number))
    Run(cfg).execute()


if __name__ == "__main__":
    main()
