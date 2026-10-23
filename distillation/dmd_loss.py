"""Losses and parameter updates of multi-teacher distillation, all in the paper convention (tau = 1 is
noise). The DMD direction g is built from the normalized difference of the fake and teacher velocities
(no division by tau) and reaches the student through a stop-gradient surrogate whose gradient with
respect to A^S is exactly g / B. The flow-matching loss serves both the student (on teacher samples)
and the fake network (on detached student samples). IDA pulls the fake network toward the student.
Every norm and loss uses only masked-in entries and is computed in at least float32."""

from typing import Any

import torch
from torch import Tensor, nn

from distillation.flow_utils import (
    at_least_fp32,
    masked_mean_sq,
    masked_norm,
    masked_sum_sq,
    noisy_action,
    per_sample,
    velocity_paper,
)
from distillation.interfaces import VelocityModel


def dmd_direction(v_fake: Tensor, v_teacher: Tensor, tau: Tensor, mask: Tensor, eps: float) -> Tensor:
    """g = −(1 − tau)·(v_fake − v_teacher) / (‖v_fake − v_teacher‖₂ + eps), masked, zero in padding."""
    v_diff = at_least_fp32(v_fake) - at_least_fp32(v_teacher)
    scale = -(1 - at_least_fp32(tau)) / (masked_norm(v_diff, mask) + eps)
    return torch.where(mask, per_sample(scale, v_diff) * v_diff, 0.0)


def dmd_surrogate_loss(a_student: Tensor, g: Tensor, mask: Tensor) -> Tensor:
    """L_DMD = (1/2B)·Σ_b ‖A^S_b − sg(A^S_b − g_b)‖² over masked-in entries, so ∂L_DMD/∂A^S = g / B."""
    a = at_least_fp32(a_student)
    if g.shape != a.shape:
        raise ValueError(f"g shape {tuple(g.shape)} must equal student action shape {tuple(a.shape)}")
    target = (a - g).detach()
    return 0.5 * masked_sum_sq(a - target, mask).mean()


def flow_matching_loss(
    model: VelocityModel, a_clean: Tensor, cond: Any, mask: Tensor, generator: torch.Generator
) -> Tensor:
    """L_FM = mean over masked-in entries of (v(A^tau, tau) − (eps − A))², tau ~ U[0, 1), eps ~ N(0, I)."""
    tau = torch.rand(a_clean.shape[:1], generator=generator, dtype=a_clean.dtype, device=a_clean.device)
    eps = torch.randn(a_clean.shape, generator=generator, dtype=a_clean.dtype, device=a_clean.device)
    v = velocity_paper(model, noisy_action(a_clean, eps, tau), tau, cond)
    v_target = at_least_fp32(eps) - at_least_fp32(a_clean)
    return masked_mean_sq(at_least_fp32(v) - v_target, mask)


def _trainable_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """Name → parameter for every parameter of `module` with requires_grad=True."""
    return {name: p for name, p in module.named_parameters() if p.requires_grad}


def _check_matching(psi: dict[str, nn.Parameter], theta: dict[str, nn.Parameter]) -> None:
    """Raise ValueError unless the fake and student trainable parameters match in names and shapes."""
    if psi.keys() != theta.keys():
        only_fake = sorted(psi.keys() - theta.keys())
        only_student = sorted(theta.keys() - psi.keys())
        raise ValueError(f"IDA parameter names differ: only in fake {only_fake}, only in student {only_student}")
    mismatched = [f"{n}: {tuple(p.shape)} vs {tuple(theta[n].shape)}" for n, p in psi.items()
                  if p.shape != theta[n].shape]
    if mismatched:
        raise ValueError(f"IDA parameter shapes differ (fake vs student): {mismatched}")


@torch.no_grad()
def ida_update(fake: nn.Module, student: nn.Module, lam: float) -> None:
    """IDA in place: psi ← lam·psi + (1 − lam)·theta over matching trainable parameters (exact at lam = 0, 1)."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    psi, theta = _trainable_parameters(fake), _trainable_parameters(student)
    _check_matching(psi, theta)
    for name, p in psi.items():
        p.mul_(lam).add_(theta[name].to(p), alpha=1.0 - lam)
