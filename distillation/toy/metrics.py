"""Toy metrics against the exact teachers, on the real 4-D action vector.

Per task and observation o in {-0.8, 0, 0.8}, n = 2,000 samples are compared with exact teacher samples:
  sliced W2 - Wasserstein-2 averaged over 128 random 1-D projections (sqrt of the mean squared quantile gap);
  coverage  - modes holding >= 1/(2M) of the samples, each sample assigned to its nearest step-1 centre if within
              3 sigma of it (98.9% of an exact mode's mass);
  success   - fraction of samples whose step-1 point is within r = sigma * sqrt(-2 ln 0.1) of a step-1 centre
              (90% of an exact mode's mass).
Both use step-1 coordinates: step 2 repeats the mode identity, and a 4-D radius would count a shift twice.
`evaluate` averages W2 and success over the observations and takes the minimum coverage. Reference levels: the
noise floor (exact vs. exact, independent draws) and the 4-step teacher (flow_solve of the teacher vs. exact).
All metric math runs in float64.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from distillation.flow_utils import flow_solve
from distillation.interfaces import VelocityModel
from distillation.router import TeacherRouter
from distillation.toy.gmm_teacher import GMMTeacher
from distillation.toy.problem import D, H, REAL_D, make_cond, to_real

SampleFn = Callable[[Mapping[str, Tensor], int, torch.Generator], Tensor]
SUCCESS_RADIUS = math.sqrt(-2.0 * math.log(0.1))
COVERAGE_RADIUS = 3.0


def sliced_w2(x: Tensor, y: Tensor, generator: torch.Generator, n_proj: int = 128) -> float:
    """Sliced Wasserstein-2 distance between equal-size samples x, y [N, d] over n_proj random directions."""
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(f"x and y must be [N, d] of equal shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    dirs = torch.randn(x.shape[1], n_proj, generator=generator, device=x.device, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    x_proj = torch.sort(x.to(torch.float64) @ dirs, dim=0).values
    y_proj = torch.sort(y.to(torch.float64) @ dirs, dim=0).values
    return float((x_proj - y_proj).pow(2).mean().sqrt())


def nearest_step1(x: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    """Distance [N] and index [N] of the nearest step-1 centre for samples x [N, 4] and mode means centers [M, 4]."""
    step1, centers1 = x[:, :REAL_D].to(torch.float64), centers[:, :REAL_D].to(torch.float64)
    return torch.cdist(step1, centers1).min(dim=1)


def mode_coverage(x: Tensor, centers: Tensor, sigma: float) -> int:
    """Number of modes holding >= 1/(2M) of the samples x [N, 4] (nearest step-1 centre, within 3 sigma)."""
    dist, nearest = nearest_step1(x, centers)
    counts = torch.bincount(nearest[dist <= COVERAGE_RADIUS * sigma], minlength=centers.shape[0])
    return int((counts * 2 * centers.shape[0] >= x.shape[0]).sum())


def toy_success(x: Tensor, centers: Tensor, sigma: float) -> float:
    """Fraction of samples x [N, 4] whose step-1 point lies within sigma * sqrt(-2 ln 0.1) of a step-1 centre."""
    dist, _ = nearest_step1(x, centers)
    return float((dist <= SUCCESS_RADIUS * sigma).to(torch.float64).mean())


def evaluate(
    sample_fn: SampleFn,
    teachers: Mapping[int, GMMTeacher],
    generator: torch.Generator,
    n: int = 2000,
    obs_values: Sequence[float] = (-0.8, 0.0, 0.8),
) -> dict[str, dict[str, Any]]:
    """Per-task metrics {"<task>": {"w2", "coverage", "num_modes", "success"}} of sample_fn vs. the exact teachers."""
    return {
        str(task_id): _evaluate_task(sample_fn, task_id, teacher, generator, n, obs_values)
        for task_id, teacher in teachers.items()
    }


def _evaluate_task(
    sample_fn: SampleFn,
    task_id: int,
    teacher: GMMTeacher,
    generator: torch.Generator,
    n: int,
    obs_values: Sequence[float],
) -> dict[str, Any]:
    """Metrics of one task: W2 and success averaged over obs_values, coverage minimised over them."""
    w2, coverage, success = [], [], []
    for o in obs_values:
        obs = torch.full((n,), o, device=generator.device)
        cond = make_cond(obs, torch.full((n,), task_id, device=generator.device))
        x = to_real(sample_fn(cond, n, generator))
        y = to_real(teacher.sample_exact(cond, n, generator))
        centers = teacher.centers(obs[:1])[0]
        w2.append(sliced_w2(x, y, generator))
        coverage.append(mode_coverage(x, centers, teacher.sigma))
        success.append(toy_success(x, centers, teacher.sigma))
    return {
        "w2": sum(w2) / len(w2),
        "coverage": min(coverage),
        "num_modes": teacher.num_modes,
        "success": sum(success) / len(success),
    }


def _noise(n: int, generator: torch.Generator) -> Tensor:
    """Initial noise z ~ N(0, I) of shape [n, H, D], float32, on the generator's device."""
    return torch.randn((n, H, D), generator=generator, device=generator.device)


def student_sampler(model: VelocityModel, num_steps: int = 4) -> SampleFn:
    """sample_fn drawing FlowSolve(model; z, cond, num_steps) samples without gradients."""
    return lambda cond, n, generator: flow_solve(model, _noise(n, generator), cond, num_steps, requires_grad=False)


def teacher_sampler(teachers: Mapping[int, VelocityModel], num_steps: int = 4) -> SampleFn:
    """sample_fn drawing routed teacher FlowSolve samples (each sample's own task teacher)."""
    router = TeacherRouter(dict(teachers))
    return lambda cond, n, generator: router.flow_solve(_noise(n, generator), cond, cond["task"], num_steps)


def exact_sampler(teachers: Mapping[int, GMMTeacher]) -> SampleFn:
    """sample_fn drawing exact samples from the teacher of the single task present in cond."""

    def sample(cond: Mapping[str, Tensor], n: int, generator: torch.Generator) -> Tensor:
        """Exact samples [n, H, D] from the one teacher named by cond["task"]."""
        tasks = cond["task"].unique().tolist()
        if len(tasks) != 1:
            raise ValueError(f"exact_sampler needs a single-task cond, got task ids {tasks}")
        return teachers[tasks[0]].sample_exact(cond, n, generator)

    return sample


def reference_levels(
    teachers: Mapping[int, GMMTeacher], generator: torch.Generator, num_steps: int = 4, n: int = 2000
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reference metrics with n samples: noise floor (exact vs. exact) and the num_steps FlowSolve teacher vs. exact."""
    return {
        "noise_floor": evaluate(exact_sampler(teachers), teachers, generator, n=n),
        "teacher_4step": evaluate(teacher_sampler(teachers, num_steps), teachers, generator, n=n),
    }
