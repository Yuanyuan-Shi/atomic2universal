"""Run metadata and extra per-iteration metrics for RLinf's EmbodiedRunner (patched in by sitecustomize.py).

RLinf's own metrics keep their names (env/, rollout/, train/, time/). This adds, once per iteration and to
every enabled backend (tensorboard, W&B):
  train/rollout_success          = env/success_once: training rollouts, with RL exploration noise (not evaluation)
  time/save_checkpoint           seconds spent in _save_checkpoint this iteration (0 without a save)
  speed/episodes_per_second      env/num_trajectories / time/step
  resources/...                  peak GPU memory per GPU and peak host RAM of the job (a2u_resources.py)
  seeds/distinct, seeds/min, ... training scene seeds used this iteration (a2u_scene_seeds.py records)
  best/score, best/step          windowed training success of the saved best checkpoint (a2u_best_checkpoint.py)
W&B (when runner.logger.logger_backends has it; id, resume, group, job type and tags come from WANDB_* set by
train_expert.sh): run name = A2U_RUN_ID; run config gains an `a2u` block (git commits, GPU type and count,
torch/CUDA, seed start, hardware file); metrics of one iteration go out in one wandb.log against the axis
`iteration` (= global step after the iteration, the checkpoint numbering), so a resumed job that repeats
iterations appends points instead of being rejected for a non-increasing W&B step.
"""

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf

import a2u_best_checkpoint
import a2u_scene_seeds
from a2u_resources import ResourceSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IterationBatchedWandb:
    """Stands in for the `wandb` module in RLinf's logger: buffers one step's logs and sends them together."""

    def __init__(self, wandb: ModuleType):
        self.wandb, self.step, self.buffer = wandb, None, {}

    def __getattr__(self, name):  # Table, run, ... (RLinf's log_table)
        return getattr(self.wandb, name)

    def log(self, data: dict, step: int) -> None:
        if self.step is not None and step != self.step:
            self.flush()
        self.step = step
        self.buffer.update(data)

    def flush(self) -> None:
        if self.buffer:
            self.wandb.log({**self.buffer, "iteration": self.step + 1})
        self.buffer = {}

    def finish(self) -> None:
        self.flush()
        self.wandb.finish()


def git_commit(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty", "--abbrev=40"],
                         capture_output=True, text=True)
    return out.stdout.strip() or f"unknown ({out.stderr.strip()})"


def run_metadata(cfg) -> dict:
    import torch

    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.split("\n")
    gpus = [name.strip() for name in gpus if name.strip()]
    structrl = PROJECT_ROOT / "third_party" / "StructRL"
    return {"project_commit": git_commit(PROJECT_ROOT), "structrl_commit": git_commit(structrl),
            "rlinf_commit": git_commit(structrl / "RLinf"), "gpu_type": gpus[0] if gpus else "none",
            "gpu_count": len(gpus), "torch": torch.__version__, "cuda": torch.version.cuda,
            "seed_start": cfg.env.train.seed, "hw_file": f"configs/rl/hw_{cfg.a2u_expert.hw}.yaml",
            "run_id": os.environ.get("A2U_RUN_ID"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename}


def seed_metrics(run_dir: Path, iteration: int) -> dict:
    """Distinct and minimum scene seeds of the resets that set up this iteration's rollouts."""
    epochs = a2u_scene_seeds.seed_settings(run_dir)["epochs"]
    seeds = defaultdict(set)
    for record in (run_dir / "seeds").glob("*.tsv"):
        for line in record.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 3:  # skip a line still being written
                seeds[int(fields[0]) // epochs].add(int(fields[1]))
    used = seeds.get(iteration, set())
    return {"seeds/distinct": len(used), "seeds/min": min(used, default=-1), "seeds/max": max(used, default=-1)}


def patch(module: ModuleType) -> None:
    runner_class = module.EmbodiedRunner
    original_init, original_save = runner_class.__init__, runner_class._save_checkpoint
    original_print = runner_class.print_metrics_table_async

    def __init__(self, cfg, *args, **kwargs):
        original_init(self, cfg, *args, **kwargs)
        self._a2u_save_seconds, self._a2u_resources = 0.0, ResourceSampler()
        self._a2u_best_window = OmegaConf.select(cfg, "a2u_expert.save_best_window", default=None) or 0  # 0: off
        bundle = self.metric_logger.logger
        if "wandb" in bundle:
            wandb = bundle["wandb"]
            wandb.run.name = os.environ.get("A2U_RUN_ID", wandb.run.name)
            wandb.define_metric("iteration")
            wandb.define_metric("*", step_metric="iteration")
            wandb.run.config.update({"a2u": run_metadata(cfg)}, allow_val_change=True)
            bundle["wandb"] = IterationBatchedWandb(wandb)  # the same dict is in metric_logger._all_loggers
            print(f"[a2u metrics] W&B run {wandb.run.id} ({wandb.run.name}): {wandb.run.url}", flush=True)

    def _save_checkpoint(self):
        start = time.monotonic()
        original_save(self)
        self._a2u_save_seconds += time.monotonic() - start

    def print_metrics_table_async(self, step, total_steps, start_time, metrics, start_step=0):
        best = {}
        run_dir = Path(os.environ["A2U_RUN_DIR"]) if os.environ.get("A2U_RUN_DIR") else None
        if self._a2u_best_window and run_dir and "env/success_once" in metrics:
            start = time.monotonic()  # runs between iterations, with the actor idle, like _save_checkpoint
            best = a2u_best_checkpoint.update(self, run_dir, self.global_step, float(metrics["env/success_once"]),
                                              self._a2u_best_window)
            self._a2u_save_seconds += time.monotonic() - start
        extra = {"time/save_checkpoint": self._a2u_save_seconds, **self._a2u_resources.take_peaks(), **best}
        self._a2u_save_seconds = 0.0
        if "env/success_once" in metrics:
            extra["train/rollout_success"] = metrics["env/success_once"]
        if metrics.get("time/step") and "env/num_trajectories" in metrics:
            extra["speed/episodes_per_second"] = metrics["env/num_trajectories"] / metrics["time/step"]
        if run_dir:
            extra |= seed_metrics(run_dir, step)
        self.metric_logger.log(extra, step)
        wandb = self.metric_logger.logger.get("wandb")
        if isinstance(wandb, IterationBatchedWandb):
            wandb.flush()
        shown = ("train/rollout_success", "best/score", "seeds/distinct", "seeds/min", "time/save_checkpoint",
                 "resources/peak_host_ram_gib", "resources/peak_gpu_mem_gib/max")
        print(f"[a2u metrics] iteration {step + 1}: " + ", ".join(f"{key}={extra[key]:.4g}" for key in shown if key in extra),
              file=sys.stderr, flush=True)
        original_print(self, step, total_steps, start_time, {**metrics, **extra}, start_step)

    runner_class.__init__, runner_class._save_checkpoint = __init__, _save_checkpoint
    runner_class.print_metrics_table_async = print_metrics_table_async
