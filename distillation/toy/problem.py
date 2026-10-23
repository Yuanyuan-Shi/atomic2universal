"""Toy problem constants, action layout and batch helpers.

Actions use GR00T's layout [B, H, D] with H = 2 steps and D = 3 dims: dims 0-1 are real, dim 2 is padding
(mask false). The real part is the 4-D vector [step1_x, step1_y, step2_x, step2_y]. Each task is a Gaussian
mixture over that vector with equal weights and isotropic std SIGMA; mode i has mean (m_i, m_i + DELTA) + o * OBS_SHIFT,
where m_i is a step-1 centre from TASK_CENTERS and o in [-1, 1] is the observation. Nearest step-1 centres are
1.0 apart (task 1) and >= 1.10 apart (task 2), i.e. >= 6.7 SIGMA; the observation shift moves all modes of a task
together, so this separation holds for every o. The success radius r = SIGMA * sqrt(-2 ln 0.1) = 0.32 is far below
half the separation, so the exact teacher's toy success is ~90% and every mode is covered.
"""

from typing import Any

import torch
from torch import Tensor

H = 2
D = 3
REAL_D = 2
TASK_IDS = (1, 2)
SIGMA = 0.15
DELTA = (0.3, 0.15)
OBS_SHIFT = (0.4, 0.2, 0.4, 0.2)
TASK_CENTERS = {
    1: ((-0.5, 0.0), (0.5, 0.0)),
    2: ((0.0, 0.6), (-0.55, -0.35), (0.55, -0.35)),
}


def action_mask(batch_size: int, device: torch.device | str | None = None) -> Tensor:
    """Boolean mask [B, H, D]: true on the real dims, false on the padding dim."""
    mask = torch.zeros(batch_size, H, D, dtype=torch.bool, device=device)
    mask[..., :REAL_D] = True
    return mask


def to_real(actions: Tensor) -> Tensor:
    """Real 4-D vectors [B, 4] from actions [B, H, D] (drops the padding dim)."""
    if actions.dim() != 3 or actions.shape[1:] != (H, D):
        raise ValueError(f"expected actions of shape [B, {H}, {D}], got {tuple(actions.shape)}")
    return actions[..., :REAL_D].reshape(actions.shape[0], H * REAL_D)


def from_real(x: Tensor) -> Tensor:
    """Actions [B, H, D] from real 4-D vectors [B, 4], with zeros in the padding dim."""
    if x.dim() != 2 or x.shape[1] != H * REAL_D:
        raise ValueError(f"expected real vectors of shape [B, {H * REAL_D}], got {tuple(x.shape)}")
    steps = x.reshape(x.shape[0], H, REAL_D)
    return torch.cat([steps, steps.new_zeros(x.shape[0], H, D - REAL_D)], dim=-1)


def mode_means(task_id: int) -> Tensor:
    """Mode means [M, 4] of a task at o = 0, in float64: (m, m + DELTA) for each step-1 centre m."""
    if task_id not in TASK_CENTERS:
        raise ValueError(f"unknown task id {task_id}; known: {TASK_IDS}")
    centres = torch.tensor(TASK_CENTERS[task_id], dtype=torch.float64)
    return torch.cat([centres, centres + torch.tensor(DELTA, dtype=torch.float64)], dim=1)


def make_cond(obs: Tensor, task: Tensor) -> dict[str, Tensor]:
    """Toy conditioning {"obs": float [B], "task": long [B]}."""
    if obs.dim() != 1 or task.shape != obs.shape:
        raise ValueError(f"obs and task must both be [B], got {tuple(obs.shape)} and {tuple(task.shape)}")
    return {"obs": obs, "task": task.long()}


def sample_batch(batch_size: int, generator: torch.Generator) -> dict[str, Any]:
    """Batch dict (cond, task_ids, action_mask): tasks drawn 50/50 from TASK_IDS, o ~ U[-1, 1]."""
    device = generator.device
    task_ids = torch.tensor(TASK_IDS, device=device)
    task = task_ids[torch.randint(len(TASK_IDS), (batch_size,), generator=generator, device=device)]
    obs = 2.0 * torch.rand(batch_size, generator=generator, device=device) - 1.0
    return {"cond": make_cond(obs, task), "task_ids": task, "action_mask": action_mask(batch_size, device)}
