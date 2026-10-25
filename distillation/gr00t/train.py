"""GR00T distillation training: the shared MTD update (`distillation.step.distill_step`) on task-balanced buffer
batches, with the frozen backbone run on the fly (buffer.py stores raw observations).

`Distiller` holds what persists across rounds: the fp32 student (perturbed for Level 1), the fake network (its copy),
the frozen bf16 teachers (one per checkpoint) behind a `TeacherRouter`, one `BackboneFeatures`, AdamW for student and
fake (weight decay 0: unused embodiment slices stay bitwise unchanged) and the generators. The next batch is sampled
and transformed on a CPU thread during the current update (`data_wait_s`). Every LOG_EVERY updates a row of means
(losses, vdiff_norm, grad_norm, data_wait_s, steps/s, peak GPU memory) goes to the caller's `log` (rounds.py:
train_log.jsonl and W&B). `OfflineMetric`: mean masked L2 distance of student and teacher chunks from the same fixed z
on held-out observations, real dims only.

Standalone run (prints the rows; --save keeps the final state; rounds.py drives real runs), a2u-policy, GPU:
  python -m distillation.gr00t.train --config configs/distillation/level2_single.yaml --variant dmd_fm \
      --collections DIR [DIR ...] --steps 5 --batch-size 8 --holdout 0 [--save DIR]
"""

import argparse
import copy
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from distillation.flow_utils import flow_solve, masked_norm
from distillation.gr00t.buffer import ObservationBuffer, RawObs, task_id
from distillation.gr00t.features import BackboneFeatures
from distillation.gr00t.wrapper import GR00TVelocity, perturb_
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from evaluation.config import PROJECT_ROOT

DEVICE = torch.device("cuda")
LOG_EVERY = 50


def load_config(path: Path, variant: str | None) -> dict[str, Any]:
    """Read a run config; if it has variants, merge the chosen one (required) and append its name to run_id."""
    cfg = yaml.safe_load(Path(path).read_text())
    variants = cfg.pop("variants", {})
    if (variant is None and variants) or (variant is not None and variant not in variants):
        raise ValueError(f"{path}: --variant must be one of {sorted(variants)}, got {variant!r}")
    if variant is not None:
        cfg = _merge(cfg, variants[variant])
        cfg["run_id"], cfg["variant"] = f"{cfg['run_id']}_{variant}", variant
    return cfg


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """`base` with `override` merged in recursively."""
    merged = dict(base)
    for key, value in override.items():
        nested = isinstance(value, Mapping) and isinstance(base.get(key), Mapping)
        merged[key] = _merge(base[key], value) if nested else value
    return merged


def unset_keys(cfg: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose value is null: placeholders marked `# set after Phase C` / `# set by calibration`."""
    keys = []
    for key, value in cfg.items():
        keys += unset_keys(value, f"{prefix}{key}.") if isinstance(value, Mapping) else []
        keys += [prefix + key] if value is None else []
    return keys


def checkpoint_path(cfg: Mapping[str, Any], label: str) -> Path:
    """Checkpoint directory of a policy label (config paths are relative to the project root)."""
    return PROJECT_ROOT / cfg["checkpoints"][label]


def stalled(history: Sequence[Mapping[str, float]], patience: int) -> bool:
    """True when no entry among the last `patience` exceeds, on any key, the best of all earlier entries (higher is
    better); never with patience 0 or too short a history."""
    if patience < 1 or len(history) <= patience:
        return False
    earlier, recent = history[:-patience], history[-patience:]
    return not any(entry[key] > max(e[key] for e in earlier) for entry in recent for key in entry)


class Distiller:
    """Student, fake network, routed teachers, shared backbone, optimizers and generators of one run."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load every model once (teachers once per unique checkpoint) and build the optimizers."""
        student_cfg, train_cfg = cfg["student"], cfg["train"]
        self.cfg, self.dcfg, self.step = cfg, DistillConfig(**cfg["distill"]), 0
        init = checkpoint_path(cfg, student_cfg["init"])
        self.student = GR00TVelocity.from_checkpoint(init, torch.float32, DEVICE, trainable=True)
        if student_cfg["perturb_scale"] > 0:
            perturb_(self.student, student_cfg["perturb_scale"], student_cfg["perturb_seed"])
        self.fake = copy.deepcopy(self.student)
        teachers = {label: GR00TVelocity.from_checkpoint(checkpoint_path(cfg, label), torch.bfloat16, DEVICE, False)
                    for label in dict.fromkeys(cfg["teachers"].values())}
        self.router = TeacherRouter({task_id(role): teachers[label] for role, label in cfg["teachers"].items()})
        self.features = BackboneFeatures([init, *(checkpoint_path(cfg, label) for label in teachers)], DEVICE)
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
              metric: "OfflineMetric | None" = None, round_index: int = 0) -> dict[str, Any]:
        """Up to num_steps updates. With a metric: evaluate before and every eval_every updates, and stop after
        `patience` evaluations without a new best (patience 0: never). Returns steps and offline distances."""
        eval_every, patience = self.cfg["train"]["eval_every"], self.cfg["train"]["patience"]
        result: dict[str, Any] = {"first_step": self.step + 1, "steps": 0, "offline_distance": {},
                                  "stopped_early": False}

        def evaluate() -> None:
            distance = result["offline_distance"][self.step] = metric.distance(self.student)
            log({"round": round_index, "step": self.step, "offline_distance": distance})
        if metric is not None:
            evaluate()
        window, start = [], time.perf_counter()
        for k, (samples, task_ids, waited) in enumerate(self._batches(buffer, num_steps), start=1):
            metrics = self._update(samples, task_ids)
            self.step, result["steps"] = self.step + 1, k
            window.append({**metrics, "data_wait_s": waited, "iter_s": time.perf_counter() - start})
            if k % LOG_EVERY == 0 or k == num_steps:
                log(self._row(window, round_index))
                window = []
            if metric is not None and k % eval_every == 0:
                evaluate()
                if stalled([{"neg_distance": -d} for d in result["offline_distance"].values()], patience):
                    result["stopped_early"] = True
                    break
            start = time.perf_counter()
        torch.cuda.empty_cache()  # return cached activation memory to the policy servers that run between trainings
        return result

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


class OfflineMetric:
    """Mean masked L2 distance between the student's and the teachers' FlowSolve chunks from the same noise z on
    held-out observations; cond, z (from `seed`) and the teacher chunks are computed once, `batch_size` at a time."""

    def __init__(self, distiller: Distiller, observations: Sequence[RawObs], task_ids: np.ndarray, seed: int,
                 batch_size: int) -> None:
        """Encode the held-out observations and solve the teachers once."""
        generator, self.num_steps, self.chunks = torch.Generator().manual_seed(seed), distiller.dcfg.num_flow_steps, []
        for i in range(0, len(observations), batch_size):
            cond = distiller.features(observations[i:i + batch_size])
            z = torch.randn(cond["action_mask"].shape, generator=generator).to(DEVICE)
            ids = torch.as_tensor(task_ids[i:i + batch_size], device=DEVICE)
            self.chunks.append((cond, z, distiller.router.flow_solve(z, cond, ids, self.num_steps)))

    @torch.no_grad()
    def distance(self, student: GR00TVelocity) -> float:
        """Mean over held-out observations of ‖A^S(z) − A^T(z)‖ over the real action dims."""
        d = [masked_norm(flow_solve(student, z, cond, self.num_steps, requires_grad=False) - a_teacher,
                         cond["action_mask"]) for cond, z, a_teacher in self.chunks]
        return torch.cat(d).mean().item()


def parse_args() -> argparse.Namespace:
    """Options of a standalone run (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--collections", type=Path, nargs="+", required=True, help="complete collection directories")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, help="overrides train.batch_size")
    parser.add_argument("--holdout", type=int, help="overrides train.holdout (0: no offline metric)")
    parser.add_argument("--save", type=Path, help="write the final state here (Distiller.save)")
    return parser.parse_args()


def main() -> None:
    """Train --steps updates on the given collections and print the result."""
    args = parse_args()
    cfg = load_config(args.config, args.variant)
    overrides = {"steps_per_round": args.steps, "batch_size": args.batch_size, "holdout": args.holdout}
    cfg["train"].update({key: value for key, value in overrides.items() if value is not None})
    if missing := unset_keys(cfg):
        raise ValueError(f"{args.config}: set {missing} first")
    buffer, distiller, train = ObservationBuffer(args.collections), Distiller(cfg), cfg["train"]
    held = buffer.holdout(train["holdout"], train["holdout_seed"]) if train["holdout"] else None
    metric = OfflineMetric(distiller, *held, train["holdout_seed"], train["batch_size"]) if held else None
    print(json.dumps(distiller.train(buffer, args.steps, lambda row: print(json.dumps(row), flush=True), metric)))
    if args.save:
        distiller.save(args.save, {"round": 0})


if __name__ == "__main__":
    main()
