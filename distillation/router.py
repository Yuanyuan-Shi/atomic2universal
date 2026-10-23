"""Teacher routing for mixed-task batches. Each sample of task k is served by its own teacher T_k: the
router groups the batch by task id, calls each teacher once on its subset (through the shared
`flow_utils` adapter, so teachers stay in their native convention) and reassembles the outputs in the
original batch order, in the dtype of the input actions. `cond` is opaque except for subsetting: a
Tensor indexed on dim 0, or a Mapping of such values (recursively)."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from distillation.flow_utils import flow_solve, velocity_paper
from distillation.interfaces import VelocityModel

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def subset_cond(cond: Any, sel: Tensor, batch_size: int) -> Any:
    """Select the samples `sel` (boolean [B]) from a Tensor cond or a (nested) Mapping of Tensors."""
    if isinstance(cond, Tensor):
        if cond.dim() == 0 or cond.shape[0] != batch_size:
            raise ValueError(f"cond tensor of shape {tuple(cond.shape)} has no batch dim of size {batch_size}")
        return cond[sel]
    if isinstance(cond, Mapping):
        return {key: subset_cond(value, sel, batch_size) for key, value in cond.items()}
    raise TypeError(f"cond must be a Tensor or a Mapping of Tensors, got {type(cond).__name__}")


class TeacherRouter:
    """Routes every sample to the teacher of its task and returns outputs in the original batch order."""

    def __init__(self, teachers: dict[int, VelocityModel]) -> None:
        """Store the task id → teacher map; raise if it is empty."""
        if not teachers:
            raise ValueError("TeacherRouter needs at least one teacher")
        self.teachers = dict(teachers)

    def velocity_paper(self, a_tau: Tensor, tau: Tensor, cond: Any, task_ids: Tensor) -> Tensor:
        """Paper-convention velocity v_{T_k}(A^tau, tau) of each sample's teacher, [B, H, D]."""
        return self._route(
            a_tau, cond, task_ids, lambda teacher, sel, c: velocity_paper(teacher, a_tau[sel], tau[sel], c)
        )

    def flow_solve(self, z: Tensor, cond: Any, task_ids: Tensor, num_steps: int) -> Tensor:
        """Teacher samples A^T = FlowSolve(v_{T_k}; z, cond, N) of each sample's teacher, without gradients."""
        return self._route(
            z, cond, task_ids, lambda teacher, sel, c: flow_solve(teacher, z[sel], c, num_steps, requires_grad=False)
        )

    def _route(
        self, like: Tensor, cond: Any, task_ids: Tensor, run: Callable[[VelocityModel, Tensor, Any], Tensor]
    ) -> Tensor:
        """Call `run(teacher, sel, cond[sel])` once per task present and scatter the results into batch order."""
        self._check_task_ids(task_ids, like.shape[0])
        out = torch.empty_like(like)
        for task_id in task_ids.unique().tolist():
            sel = task_ids == task_id
            out[sel] = run(self.teachers[task_id], sel, subset_cond(cond, sel, like.shape[0])).to(out.dtype)
        return out

    def _check_task_ids(self, task_ids: Tensor, batch_size: int) -> None:
        """Raise unless task_ids is an integer [B] tensor whose ids all have a teacher."""
        if task_ids.shape != (batch_size,) or task_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"task_ids must be an integer tensor of shape [{batch_size}], got "
                             f"{task_ids.dtype} {tuple(task_ids.shape)}")
        unknown = sorted(set(task_ids.unique().tolist()) - self.teachers.keys())
        if unknown:
            raise ValueError(f"no teacher for task ids {unknown}; known ids {sorted(self.teachers)}")
