"""GR00T distillation training: the shared MTD update (`distillation.step.distill_step`) on task-balanced buffer
batches, with the frozen backbone run on the fly (buffer.py stores raw observations).

A run config lists K teachers, one per task: {task, role, label, checkpoint}; the router serves the samples of role
i<k> (task id k, `buffer.task_id`) with that role's teacher. `Distiller` holds what persists across rounds: the fp32
student, the fake network (its copy), the frozen bf16 teachers (one per checkpoint) behind a `TeacherRouter`, one
`BackboneFeatures`, AdamW for student and fake (weight decay 0: unused embodiment slices stay bitwise unchanged) and
the generators. The next batch is sampled and transformed on a CPU thread during the current update (`data_wait_s`).
Every LOG_EVERY updates a row of means (losses, vdiff_norm, grad_norm, data_wait_s, steps/s, peak GPU memory) goes to
the caller's `log` (rounds.py: train_log.jsonl and W&B).

Standalone run (prints the rows; --save keeps the final state; rounds.py drives real runs), a2u-policy, GPU:
  python -m distillation.gr00t.train --config configs/distillation/mtd_pnp_cabinet_drawer.yaml \
      --collections DIR [DIR ...] --steps 5 --batch-size 8 [--save DIR]
"""

import argparse
import copy
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from distillation.gr00t.buffer import ObservationBuffer, task_id
from distillation.gr00t.features import BackboneFeatures
from distillation.gr00t.wrapper import GR00TVelocity
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from evaluation.config import PROJECT_ROOT

DEVICE = torch.device("cuda")
LOG_EVERY = 50


def load_config(path: Path) -> dict[str, Any]:
    """Read a run config."""
    return yaml.safe_load(Path(path).read_text())


def unset_keys(cfg: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose value is null (placeholders), also inside lists (`teachers.0.checkpoint`)."""
    keys = []
    for key, value in cfg.items():
        nested = dict(enumerate(value)) if isinstance(value, list) else value
        keys += unset_keys(nested, f"{prefix}{key}.") if isinstance(nested, Mapping) else []
        keys += [f"{prefix}{key}"] if value is None else []
    return keys


def teacher_roles(cfg: Mapping[str, Any]) -> list[str]:
    """The roles of the config's teachers, in config order (one task each)."""
    return [teacher["role"] for teacher in cfg["teachers"]]


def project_path(path: str) -> Path:
    """A config path (relative to the project root) as an absolute path."""
    return PROJECT_ROOT / path


class Distiller:
    """Student, fake network, routed teachers, shared backbone, optimizers and generators of one run."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load every model once (teachers once per unique checkpoint) and build the optimizers."""
        train_cfg = cfg["train"]
        self.cfg, self.dcfg, self.step = cfg, DistillConfig(**cfg["distill"]), 0
        init = project_path(cfg["student"]["checkpoint"])
        self.student = GR00TVelocity.from_checkpoint(init, torch.float32, DEVICE, trainable=True)
        self.fake = copy.deepcopy(self.student)
        teachers = {teacher["checkpoint"]: GR00TVelocity.from_checkpoint(
            project_path(teacher["checkpoint"]), torch.bfloat16, DEVICE, False) for teacher in cfg["teachers"]}
        self.router = TeacherRouter({task_id(t["role"]): teachers[t["checkpoint"]] for t in cfg["teachers"]})
        if len(self.router.teachers) != len(cfg["teachers"]):
            raise ValueError(f"teachers need distinct roles, got {teacher_roles(cfg)}")
        self.features = BackboneFeatures(list(dict.fromkeys([init, *map(project_path, teachers)])), DEVICE)
        self.opt_student = torch.optim.AdamW(self.student.parameters(), lr=train_cfg["lr"], weight_decay=0.0)
        self.opt_fake = torch.optim.AdamW(self.fake.parameters(), lr=train_cfg["lr"], weight_decay=0.0)
        self.generator = torch.Generator(DEVICE).manual_seed(train_cfg["seed"])
        self.rng = np.random.default_rng(train_cfg["seed"])

    def _batches(self, buffer: ObservationBuffer, num_steps: int) -> Iterator[tuple[list, np.ndarray, float]]:
        """Yield num_steps transformed balanced batches and the seconds waited for each; the next one is sampled and
        transformed on a CPU thread while the caller runs the current update."""
        def prepare() -> tuple[list, np.ndarray]:
            observations, task_ids = buffer.sample(self.cfg["train"]["batch_size"], True, rng=self.rng)
            return self.features.transform(observations), task_ids
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare)
            for k in range(num_steps):
                start = time.perf_counter()
                samples, task_ids = future.result()
                waited = time.perf_counter() - start
                if k + 1 < num_steps:
                    future = pool.submit(prepare)
                yield samples, task_ids, waited

    def _update(self, samples: list, task_ids: np.ndarray) -> dict[str, float]:
        """Backbone on the batch, then one `distill_step`; raise on a non-finite metric."""
        cond = self.features.encode(samples)
        batch = {"cond": cond, "task_ids": torch.as_tensor(task_ids, device=DEVICE), "action_mask": cond["action_mask"]}
        metrics = distill_step(self.student, self.fake, self.router, batch, self.dcfg, self.opt_student,
                               self.opt_fake, self.generator)
        bad = sorted(key for key, value in metrics.items() if not math.isfinite(value))
        if bad:
            raise FloatingPointError(f"update {self.step + 1}: non-finite {bad}")
        return metrics

    def _row(self, window: list[dict[str, float]], round_index: int) -> dict[str, Any]:
        """Log row: per-key means over the window, update rate and peak GPU memory."""
        means = {key: float(np.mean([m[key] for m in window])) for key in window[0] if key != "iter_s"}
        return {"round": round_index, "step": self.step, **means,
                "steps_per_s": len(window) / sum(m["iter_s"] for m in window),
                "max_mem_gb": torch.cuda.max_memory_allocated(DEVICE) / 1e9}

    def train(self, buffer: ObservationBuffer, num_steps: int, log: Callable[[dict[str, Any]], None],
              round_index: int = 0) -> dict[str, Any]:
        """num_steps updates on task-balanced batches; raise unless the buffer holds exactly the teachers' tasks."""
        tasks, routed = sorted(buffer.task_counts()), sorted(self.router.teachers)
        if tasks != routed:
            raise ValueError(f"buffer tasks {tasks} differ from the teachers' tasks {routed}")
        first, window, start = self.step + 1, [], time.perf_counter()
        for k, (samples, task_ids, waited) in enumerate(self._batches(buffer, num_steps), start=1):
            metrics = self._update(samples, task_ids)
            self.step += 1
            window.append({**metrics, "data_wait_s": waited, "iter_s": time.perf_counter() - start})
            if k % LOG_EVERY == 0 or k == num_steps:
                log(self._row(window, round_index))
                window = []
            start = time.perf_counter()
        torch.cuda.empty_cache()  # return cached activation memory to the policy servers that run between trainings
        return {"first_step": first, "steps": num_steps}

    def _stateful(self) -> dict[str, Any]:
        """The objects whose state_dict a resume needs."""
        return {"student": self.student, "fake": self.fake, "opt_student": self.opt_student, "opt_fake": self.opt_fake}

    def save(self, directory: Path, meta: dict[str, Any]) -> None:
        """Write each state dict and the CUDA generator state to its own file, then state.json (last)."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").unlink(missing_ok=True)
        for name, obj in self._stateful().items():
            torch.save(obj.state_dict(), directory / f"{name}.pt")
        torch.save(self.generator.get_state(), directory / "cuda_generator.pt")
        state = {**meta, "step": self.step, "numpy_rng": self.rng.bit_generator.state}
        (directory / "state.json").write_text(json.dumps(state))

    def load(self, directory: Path) -> dict[str, Any]:
        """Restore what `save` wrote (memory-mapped reads, copied into the GPU tensors); return state.json."""
        state = json.loads((directory / "state.json").read_text())
        options = {"map_location": "cpu", "mmap": True, "weights_only": True}
        for name, obj in self._stateful().items():
            obj.load_state_dict(torch.load(directory / f"{name}.pt", **options))
        self.generator.set_state(torch.load(directory / "cuda_generator.pt", **options))
        self.rng.bit_generator.state, self.step = state["numpy_rng"], state["step"]
        return state


def parse_args() -> argparse.Namespace:
    """Options of a standalone run (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True, help="complete collection directories")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, help="overrides train.batch_size")
    parser.add_argument("--save", type=Path, help="write the final state here (Distiller.save)")
    return parser.parse_args()


def main() -> None:
    """Train --steps updates on the given collections (train.holdout reserved, as in rounds.py); print the result."""
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {"steps_per_round": args.steps, "batch_size": args.batch_size}
    cfg["train"].update({key: value for key, value in overrides.items() if value is not None})
    if missing := unset_keys(cfg):
        raise ValueError(f"{args.config}: set {missing} first")
    buffer, distiller, train = ObservationBuffer(args.collections), Distiller(cfg), cfg["train"]
    if train["holdout"]:
        buffer.holdout(train["holdout"], train["holdout_seed"])
    print(json.dumps(distiller.train(buffer, args.steps, lambda row: print(json.dumps(row), flush=True))))
    if args.save:
        distiller.save(args.save, {"round": 0})


if __name__ == "__main__":
    main()
