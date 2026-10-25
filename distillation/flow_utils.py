"""Convention adapter and flow utilities shared by the GR00T distillation runs and their tests. All method
math uses the paper convention (tau = 1 is noise, A^tau = (1 - tau)·A + tau·eps, velocity target
eps − A); every model exposes GR00T's native convention, verified in the Isaac-GR00T fork at commit
9d7d7a9, file third_party/Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py: training
uses `noisy_trajectory = (1 - t) * noise + t * actions` and `velocity = actions - noise` (lines
314–315), so tau = 1 − t and v_paper = −v_native. The four conversion functions below are the only
place where that time/sign conversion happens. `flow_solve` reproduces the sampler `get_action`
(lines 353–408): start from noise at t = 0, `dt = 1.0 / num_steps` (373–374), loop
`for t in range(num_steps)` (377) with `t_cont = t / float(num_steps)` (378), i.e. t_k = k/N, and the
Euler update `actions = actions + dt * pred_velocity` (407); GR00T's discretization of t_cont into
buckets (379) belongs inside the model. The masked reductions exclude padded entries with
`torch.where` (never by multiplying) and compute in at least float32."""

from typing import Any

import torch
from torch import Tensor

from distillation.interfaces import VelocityModel


def paper_to_native_time(tau: Tensor) -> Tensor:
    """Map paper time tau (1 = noise) to native time t = 1 − tau (0 = noise)."""
    return 1 - tau


def native_to_paper_time(t: Tensor) -> Tensor:
    """Map native time t (0 = noise) to paper time tau = 1 − t (1 = noise)."""
    return 1 - t


def native_to_paper_velocity(v: Tensor) -> Tensor:
    """Map a native velocity (action − noise) to the paper velocity (noise − action)."""
    return -v


def paper_to_native_velocity(v: Tensor) -> Tensor:
    """Map a paper velocity (noise − action) to the native velocity (action − noise)."""
    return -v


def at_least_fp32(x: Tensor) -> Tensor:
    """Promote half/bfloat16 to float32; float32 and float64 are returned unchanged."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def per_sample(tau: Tensor, like: Tensor) -> Tensor:
    """Reshape a per-sample tensor [B] to [B, 1, 1] so it broadcasts over the H and D axes of `like`."""
    if like.dim() != 3:
        raise ValueError(f"expected actions of shape [B, H, D], got {tuple(like.shape)}")
    if tau.shape != like.shape[:1]:
        raise ValueError(f"expected a per-sample tensor of shape [{like.shape[0]}], got {tuple(tau.shape)}")
    return tau[:, None, None]


def velocity_paper(model: VelocityModel, a_tau: Tensor, tau: Tensor, cond: Any) -> Tensor:
    """Paper-convention velocity v(A^tau, tau) = −model.velocity(A^tau, 1 − tau, cond)."""
    return native_to_paper_velocity(model.velocity(a_tau, paper_to_native_time(tau), cond))


def noisy_action(a: Tensor, eps: Tensor, tau: Tensor) -> Tensor:
    """Paper-convention noisy action A^tau = (1 − tau)·A + tau·eps, with tau [B] broadcast over H and D."""
    if eps.shape != a.shape:
        raise ValueError(f"eps shape {tuple(eps.shape)} must equal action shape {tuple(a.shape)}")
    tau_b = per_sample(tau, a)
    return (1 - tau_b) * a + tau_b * eps


def flow_solve(model: VelocityModel, z: Tensor, cond: Any, num_steps: int, requires_grad: bool) -> Tensor:
    """GR00T's sampler: from z at t = 0, N Euler steps a ← a + (1/N)·velocity(a, k/N), k = 0..N−1."""
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError(f"num_steps must be an int >= 1, got {num_steps}")
    dt = 1.0 / num_steps
    a = z
    with torch.set_grad_enabled(requires_grad):
        for k in range(num_steps):
            t_k = torch.full(z.shape[:1], k / num_steps, dtype=z.dtype, device=z.device)
            a = a + dt * model.velocity(a, t_k, cond).to(a.dtype)
    return a


def _check_mask(x: Tensor, mask: Tensor) -> None:
    """Raise unless `mask` is a boolean tensor with exactly the shape of `x`."""
    if mask.dtype != torch.bool:
        raise ValueError(f"action_mask must be torch.bool, got {mask.dtype}")
    if mask.shape != x.shape:
        raise ValueError(f"action_mask shape {tuple(mask.shape)} must equal tensor shape {tuple(x.shape)}")


def masked_sum_sq(x: Tensor, mask: Tensor) -> Tensor:
    """Per-sample sum of squares over masked-in entries, Σ_{h,d} mask·x², shape [B], in at least fp32."""
    _check_mask(x, mask)
    x_in = torch.where(mask, at_least_fp32(x), 0.0)
    return x_in.square().flatten(1).sum(dim=1)


def masked_norm(x: Tensor, mask: Tensor) -> Tensor:
    """Per-sample L2 norm over masked-in entries, shape [B], in at least fp32."""
    return masked_sum_sq(x, mask).sqrt()


def masked_mean_sq(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of x² over all masked-in entries of the batch (GR00T's action-loss reduction), in at least fp32."""
    sum_sq = masked_sum_sq(x, mask).sum()
    num_in = mask.sum()
    if num_in.item() == 0:
        raise ValueError("action_mask has no masked-in entries; the mean is undefined")
    return sum_sq / num_in
