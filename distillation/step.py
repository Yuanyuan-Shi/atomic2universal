"""One multi-teacher distillation update, in the exact order shared by the toy and the GR00T runs:
(1) student sample A^S = FlowSolve(v_theta; z) with gradients; 
(2) K_psi fake-network updates by flow matching on the detached A^S; 
(3) fresh tau and eps, A^tau from the detached A^S, teacher and fake velocities as constants, the DMD direction g and the surrogate loss on the graph-carrying A^S;
(4) teacher samples A^T = FlowSolve(v_{T_k}; z') without gradients and the student's flow-matching
loss on them; 
(5) one student step on alpha_DMD·L_DMD + alpha_FM·L_FM, where a branch whose alpha is 0 is skipped entirely; 
(6) IDA of the fake network toward the student. All noise is float32 with the
shape and device of the action mask and is drawn from one explicit torch.Generator."""

from typing import Any

import torch
from torch import Tensor, nn

from distillation.dmd_loss import dmd_direction, dmd_surrogate_loss, flow_matching_loss, ida_update
from distillation.flow_utils import at_least_fp32, flow_solve, masked_norm, noisy_action, velocity_paper
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter


def _unpack_batch(batch: dict[str, Any]) -> tuple[Any, Tensor, Tensor]:
    """Return (cond, task_ids, action_mask) after checking the keys and the mask's dtype and rank."""
    missing = sorted({"cond", "task_ids", "action_mask"} - batch.keys())
    if missing:
        raise KeyError(f"batch is missing keys {missing}")
    mask = batch["action_mask"]
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.dim() != 3:
        raise ValueError("batch['action_mask'] must be a torch.bool tensor of shape [B, H, D]")
    return batch["cond"], batch["task_ids"], mask


def _gaussian_noise(mask: Tensor, generator: torch.Generator) -> Tensor:
    """Standard normal float32 noise with the shape and device of the action mask."""
    return torch.randn(mask.shape, generator=generator, dtype=torch.float32, device=mask.device)


def _uniform_tau(mask: Tensor, generator: torch.Generator) -> Tensor:
    """Per-sample paper time tau ~ U[0, 1), float32 [B] on the device of the action mask."""
    return torch.rand(mask.shape[:1], generator=generator, dtype=torch.float32, device=mask.device)


def _update_fake(
    fake: nn.Module, a_sample: Tensor, cond: Any, mask: Tensor, k_fake: int,
    opt_fake: torch.optim.Optimizer, generator: torch.Generator,
) -> float:
    """Step (2): k_fake optimizer steps of flow matching of the fake network on detached student samples."""
    losses = []
    for _ in range(k_fake):
        opt_fake.zero_grad(set_to_none=True)
        loss_fake = flow_matching_loss(fake, a_sample, cond, mask, generator)
        loss_fake.backward()
        opt_fake.step()
        losses.append(loss_fake.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def dmd_direction_at_noisy(
    fake: nn.Module, router: TeacherRouter, a_sample: Tensor, cond: Any, task_ids: Tensor, mask: Tensor,
    norm_eps: float, generator: torch.Generator,
) -> tuple[Tensor, float]:
    """Step (3a): g at (A^tau, tau) from fresh tau, eps, with both velocities as constants; also mean ‖v_psi − v_T‖."""
    tau, eps = _uniform_tau(mask, generator), _gaussian_noise(mask, generator)
    a_tau = noisy_action(a_sample, eps, tau)
    v_teacher = router.velocity_paper(a_tau, tau, cond, task_ids)
    v_fake = velocity_paper(fake, a_tau, tau, cond)
    g = dmd_direction(v_fake, v_teacher, tau, mask, norm_eps)
    vdiff_norm = masked_norm(at_least_fp32(v_fake) - at_least_fp32(v_teacher), mask).mean()
    return g, vdiff_norm.item()


def _dmd_branch(
    student: nn.Module, fake: nn.Module, router: TeacherRouter, cond: Any, task_ids: Tensor, mask: Tensor,
    cfg: DistillConfig, opt_fake: torch.optim.Optimizer, generator: torch.Generator,
) -> tuple[Tensor, dict[str, float]]:
    """Steps (1)–(3): student sample, fake updates, and the DMD surrogate loss with its metrics."""
    a_student = flow_solve(student, _gaussian_noise(mask, generator), cond, cfg.num_flow_steps, requires_grad=True)
    a_sample = a_student.detach()
    loss_fake = _update_fake(fake, a_sample, cond, mask, cfg.k_fake, opt_fake, generator)
    g, vdiff_norm = dmd_direction_at_noisy(fake, router, a_sample, cond, task_ids, mask, cfg.norm_eps, generator)
    loss_dmd = dmd_surrogate_loss(a_student, g, mask)
    return loss_dmd, {"loss_dmd": loss_dmd.item(), "loss_fake": loss_fake, "vdiff_norm": vdiff_norm}


def _fm_branch(
    student: nn.Module, router: TeacherRouter, cond: Any, task_ids: Tensor, mask: Tensor,
    cfg: DistillConfig, generator: torch.Generator,
) -> Tensor:
    """Step (4): teacher samples A^T without gradients, then the student's flow-matching loss on them."""
    a_teacher = router.flow_solve(_gaussian_noise(mask, generator), cond, task_ids, cfg.num_flow_steps)
    return flow_matching_loss(student, a_teacher, cond, mask, generator)


def _grad_norm(module: nn.Module) -> float:
    """Total L2 norm of the module's parameter gradients, in at least fp32; raises if there are none."""
    norms = [torch.linalg.vector_norm(at_least_fp32(p.grad)) for p in module.parameters() if p.grad is not None]
    if not norms:
        raise RuntimeError("student has no parameter gradients after backward; is it trainable?")
    return torch.linalg.vector_norm(torch.stack(norms)).item()


def _student_step(student: nn.Module, opt_student: torch.optim.Optimizer, loss: Tensor) -> float:
    """Step (5): backward on the weighted student loss, record the gradient norm, then step."""
    opt_student.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = _grad_norm(student)
    opt_student.step()
    return grad_norm


def distill_step(
    student: nn.Module, fake: nn.Module, router: TeacherRouter, batch: dict[str, Any], cfg: DistillConfig,
    opt_student: torch.optim.Optimizer, opt_fake: torch.optim.Optimizer, generator: torch.Generator,
) -> dict[str, float]:
    """Run one MTD update (steps 1–6) and return its float metrics (keys depend on which branches ran)."""
    cond, task_ids, mask = _unpack_batch(batch)
    metrics: dict[str, float] = {}
    terms: list[Tensor] = []
    if cfg.alpha_dmd > 0:
        loss_dmd, dmd_metrics = _dmd_branch(student, fake, router, cond, task_ids, mask, cfg, opt_fake, generator)
        terms.append(cfg.alpha_dmd * loss_dmd)
        metrics.update(dmd_metrics)
    if cfg.alpha_fm > 0:
        loss_fm = _fm_branch(student, router, cond, task_ids, mask, cfg, generator)
        terms.append(cfg.alpha_fm * loss_fm)
        metrics["loss_fm"] = loss_fm.item()
    metrics["grad_norm"] = _student_step(student, opt_student, torch.stack(terms).sum())
    ida_update(fake, student, cfg.lambda_ida)
    return metrics
