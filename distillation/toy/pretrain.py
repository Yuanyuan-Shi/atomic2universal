"""Pre-train the toy student on a perturbed base distribution: `python -m distillation.toy.pretrain`.

The base distribution is a deliberately wrong copy of the teachers (a GMMTeacher per task): every mode mean is
shifted by SHIFT along the step-1 unit vector SHIFT_DIRECTION (the same shift on both steps, so chunks stay
coherent), sigma is multiplied by SIGMA_SCALE, and task-2 mode HALF_WEIGHT_MODE gets half weight. A fresh MLPVelocity
is trained on it by native flow matching through the shared flow_matching_loss (uniform tau; the adapter converts
conventions), with Adam and a cosine learning-rate decay. The saved student is evaluated with 4-step FlowSolve
against the true teachers. Tuning for toy success ~50-60% on both tasks: the spec's start (shift 0.35, sigma x 2.5)
gives ~29% even with the exact base velocity; shift 0.25 with sigma x 1.4 gives ~57% / 55%, a vertical shift keeps the
two tasks within ~3 points (a diagonal one opens an ~8-point gap), and 8,000 steps are used because 4,000 under-fit
the 3-mode task by 3-5 points. Along the success contour the initial sliced W2 is set mostly by the shift (0.16-0.24); the larger
shift was chosen to keep the initial W2 well above the noise floor.
The checkpoint runs/student_init.pt holds {"state_dict", "base", "metrics", "steps", "seed"}.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from distillation.dmd_loss import flow_matching_loss
from distillation.toy.gmm_teacher import GMMTeacher, make_teachers
from distillation.toy.metrics import evaluate, student_sampler
from distillation.toy.mlp_student import MLPVelocity
from distillation.toy.problem import H, OBS_SHIFT, SIGMA, TASK_IDS, make_cond, mode_means, sample_batch

SHIFT = 0.25
SHIFT_DIRECTION = (0.0, 1.0)
SIGMA_SCALE = 1.4
HALF_WEIGHT_MODE = 0
STEPS = 8000
BATCH_SIZE = 512
LR = 1e-3
SEED = 0
CHECKPOINT = Path(__file__).resolve().parent / "runs" / "student_init.pt"


def base_teachers(shift: float = SHIFT, sigma_scale: float = SIGMA_SCALE) -> dict[int, GMMTeacher]:
    """Base distribution per task: shifted means, sigma * sigma_scale, half weight on task-2 mode HALF_WEIGHT_MODE."""
    offset = shift * torch.tensor(SHIFT_DIRECTION, dtype=torch.float64).repeat(H)
    base = {}
    for task_id in TASK_IDS:
        means = mode_means(task_id)
        weights = torch.ones(means.shape[0], dtype=torch.float64)
        if task_id == 2:
            weights[HALF_WEIGHT_MODE] = 0.5
        base[task_id] = GMMTeacher(means + offset, weights, SIGMA * sigma_scale, torch.tensor(OBS_SHIFT))
    return base


def sample_base(base: dict[int, GMMTeacher], batch_size: int, generator: torch.Generator) -> tuple[Tensor, dict]:
    """A training batch and its clean actions [B, H, D], each drawn from its task's base distribution."""
    batch = sample_batch(batch_size, generator)
    obs, task = batch["cond"]["obs"], batch["task_ids"]
    a_clean = torch.zeros(batch["action_mask"].shape, device=obs.device)
    for task_id, teacher in base.items():
        sel = task == task_id
        a_clean[sel] = teacher.sample_exact(make_cond(obs[sel], task[sel]), int(sel.sum()), generator)
    return a_clean, batch


def pretrain(base: dict[int, GMMTeacher], steps: int, generator: torch.Generator) -> MLPVelocity:
    """Train a fresh MLPVelocity by flow matching on the base distribution."""
    model = MLPVelocity(generator=generator)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(1, steps + 1):
        a_clean, batch = sample_base(base, BATCH_SIZE, generator)
        loss = flow_matching_loss(model, a_clean, batch["cond"], batch["action_mask"], generator)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        if step % 500 == 0:
            print(f"step {step}: flow-matching loss {loss.item():.4f}", flush=True)
    return model


def main() -> None:
    """Pre-train, evaluate against the true teachers and save the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shift", type=float, default=SHIFT)
    parser.add_argument("--sigma-scale", type=float, default=SIGMA_SCALE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(SEED)
    start = time.time()
    model = pretrain(base_teachers(args.shift, args.sigma_scale), args.steps, generator)
    metrics = evaluate(student_sampler(model), make_teachers(), generator)
    base = {"shift": args.shift, "shift_direction": list(SHIFT_DIRECTION), "sigma_scale": args.sigma_scale,
            "half_weight_mode": HALF_WEIGHT_MODE}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base": base, "metrics": metrics, "steps": args.steps,
                "seed": SEED}, args.out)
    print(json.dumps({"base": base, "metrics": metrics, "wall_time_s": round(time.time() - start, 1)}, indent=1))


if __name__ == "__main__":
    main()
