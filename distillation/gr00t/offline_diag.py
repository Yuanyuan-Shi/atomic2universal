"""Offline comparison of students with their teachers on held-out buffer observations: the spec's same-noise
distance plus distribution-level statistics, overall and per action component (evaluation/policy.py ACTION_LAYOUT).

Per held-out observation: K noise draws z_1..z_K shared by student and teacher, the 4-step chunks S(z_k) and T(z_k)
(teacher of the observation's task via `TeacherRouter`), real action dims only, each chunk flattened:
  same_z          mean_k ‖S(z_k) − T(z_k)‖              spec §7's offline metric, averaged over K draws
  student_spread  mean_{k≠l} ‖S(z_k) − S(z_l)‖          how far two student samples are apart (collapse → small)
  teacher_spread  mean_{k≠l} ‖T(z_k) − T(z_l)‖          the same-noise distance of two unrelated teacher samples
  mean_gap        ‖mean_k S(z_k) − mean_k T(z_k)‖
  energy          2·E‖S − T'‖ − E‖S − S'‖ − E‖T − T'‖  (k ≠ l pairs), 0 iff the two chunk distributions agree
All are means over observations; `energy_sem` is the standard error of the overall energy distance over observations
(optimistic: queries of one episode are correlated). With K draws the mean gap is biased upward (≈ spread/√K).

Usage (a2u-policy, GPU, from the project root):
  python -m distillation.gr00t.offline_diag --collections DIR [DIR ...] --teachers i1=CKPT [i2=CKPT] \
      --students NAME=SPEC [NAME=SPEC ...]      SPEC = checkpoint folder | student state .pt
Pass the collections in the run's order (rounds.py: student jobs, then teacher jobs) so `holdout` returns the run's set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from distillation.flow_utils import flow_solve
from distillation.gr00t.buffer import ObservationBuffer, task_id
from distillation.gr00t.export import CKPT60K
from distillation.gr00t.features import REAL_ACTION_DIM, BackboneFeatures
from distillation.gr00t.wrapper import GR00TVelocity
from distillation.router import TeacherRouter
from evaluation.policy import ACTION_LAYOUT

DEVICE = torch.device("cuda")
NUM_STEPS = 4


def action_groups() -> dict[str, slice]:
    """Real-dim slice of each action component, in the order of ACTION_LAYOUT."""
    groups, start = {}, 0
    for key, width in ACTION_LAYOUT:
        groups[key.removeprefix("action.")] = slice(start, start + width)
        start += width
    return groups


def chunk_stats(s: Tensor, t: Tensor) -> dict[str, Tensor]:
    """Per-observation statistics [B] of student chunks s and teacher chunks t, both [K, B, H, d]."""
    s, t = s.flatten(2).transpose(0, 1), t.flatten(2).transpose(0, 1)  # [B, K, H·d]
    st, ss, tt = torch.cdist(s, t), torch.cdist(s, s), torch.cdist(t, t)
    off = ~torch.eye(s.shape[1], dtype=torch.bool, device=s.device)
    return {"same_z": st.diagonal(dim1=1, dim2=2).mean(1), "student_spread": ss[:, off].mean(1),
            "teacher_spread": tt[:, off].mean(1), "mean_gap": (s.mean(1) - t.mean(1)).norm(dim=1),
            "energy": 2 * st[:, off].mean(1) - ss[:, off].mean(1) - tt[:, off].mean(1)}


def summarize(pairs: list[tuple[Tensor, Tensor]]) -> dict[str, dict[str, float]]:
    """Means over observations, overall (with the energy SEM) and per action component."""
    def means(sl: slice) -> dict[str, Tensor]:
        rows = [chunk_stats(s[..., sl], t[..., sl]) for s, t in pairs]
        return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    overall = means(slice(None))
    out = {"overall": {key: round(v.mean().item(), 4) for key, v in overall.items()}}
    out["overall"]["energy_sem"] = round(overall["energy"].std().item() / np.sqrt(len(overall["energy"])), 4)
    out.update({name: {key: round(v.mean().item(), 4) for key, v in means(sl).items()}
                for name, sl in action_groups().items()})
    return out


def load_student(spec: str) -> GR00TVelocity:
    """A frozen fp32 student from a checkpoint folder or a `Distiller.save` student state (of a ckpt-60000 student)."""
    if Path(spec).is_dir():
        student = GR00TVelocity.from_checkpoint(Path(spec), torch.float32, DEVICE, trainable=False)
    else:
        student = GR00TVelocity.from_checkpoint(CKPT60K, torch.float32, DEVICE, trainable=False)
        student.load_state_dict(torch.load(spec, map_location="cpu", mmap=True, weights_only=True))
    return student.requires_grad_(False)


def solve(model: GR00TVelocity | tuple[TeacherRouter, Tensor], z: Tensor, cond: dict[str, Tensor], real: int) -> Tensor:
    """Real-dim 4-step chunks [K, B, H, real] from K noise draws z [K, B, H, D]; `model` is a student or a
    `(router, task_ids)` pair."""
    with torch.no_grad():
        if isinstance(model, tuple):
            return torch.stack([model[0].flow_solve(zk, cond, model[1], NUM_STEPS)[..., :real].float() for zk in z])
        return torch.stack([flow_solve(model, zk, cond, NUM_STEPS, False)[..., :real].float() for zk in z])


def parse_args() -> argparse.Namespace:
    """Options (see the module docstring); RESULTS.md lists the exact commands behind its tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--teachers", nargs="+", required=True, help="ROLE=CHECKPOINT per task")
    parser.add_argument("--students", nargs="+", required=True, help="NAME=SPEC")
    parser.add_argument("--holdout", type=int, default=200, help="held-out observations (buffer.holdout)")
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--k", type=int, default=8, help="noise draws per observation")
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Print one JSON line of statistics per student."""
    args = parse_args()
    teachers = {task_id(role): Path(path) for role, path in (item.split("=", 1) for item in args.teachers)}
    models = {path: GR00TVelocity.from_checkpoint(path, torch.bfloat16, DEVICE, False)
              for path in set(teachers.values())}
    router = TeacherRouter({task: models[path] for task, path in teachers.items()})
    features = BackboneFeatures(sorted(set(teachers.values()) | {CKPT60K}), DEVICE)
    observations, task_ids = ObservationBuffer(args.collections).holdout(args.holdout, args.holdout_seed)
    generator, batches = torch.Generator().manual_seed(args.noise_seed), []
    for i in range(0, len(observations), args.batch_size):
        cond = features(observations[i:i + args.batch_size])
        z = torch.randn((args.k, *cond["action_mask"].shape), generator=generator).to(DEVICE)
        ids = torch.as_tensor(task_ids[i:i + args.batch_size], device=DEVICE)
        batches.append((cond, z, solve((router, ids), z, cond, REAL_ACTION_DIM)))
    for name, spec in (item.split("=", 1) for item in args.students):
        student = load_student(spec)
        pairs = [(solve(student, z, cond, REAL_ACTION_DIM), t) for cond, z, t in batches]
        print(json.dumps({"student": name, "spec": spec, **summarize(pairs)}), flush=True)
        del student
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
