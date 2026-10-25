"""Test doubles and small helpers shared by the distillation tests. `TinyVelocity` is a small native-convention
MLP whose parameters come from an explicit generator, so gradient tests run on a real network in float64 without the
global RNG; `SpyVelocity` wraps any VelocityModel and records every call with its output (or fails the test when it is
called), which the routing, FlowSolve and `distill_step` tests use to prove what ran and on which inputs;
`ConstantVelocity` returns a fixed velocity; `ShiftVelocity` is a student whose FlowSolve samples are N(mean, I) with a
trainable mean, and `GaussianFake` is the exact fake network of that distribution. The remaining functions build seeded
inputs of the Gaussian-mixture problem (mixture.py) in the GR00T layout [B, H, D] and the test-side oracles (Eq. (9),
masking by selection, errors, order-agnostic row comparison)."""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from distillation.flow_utils import noisy_action
from distillation.tests.mixture import D, H, GMMTeacher, make_cond, to_real

F64 = torch.float64


def make_generator(seed: int) -> torch.Generator:
    """A CPU torch.Generator seeded with `seed`."""
    return torch.Generator().manual_seed(seed)


def masked(x: Tensor, mask: Tensor) -> Tensor:
    """x with padded entries replaced by zero (selection, never multiplication)."""
    return torch.where(mask, x, torch.zeros_like(x))


def rel_err(got: Tensor, want: Tensor) -> float:
    """Relative error ‖got − want‖ / ‖want‖ over all entries."""
    return ((got - want).norm() / want.norm()).item()


def score_from_velocity(v_paper: Tensor, x: Tensor, tau: Tensor) -> Tensor:
    """Eq. (9): s(x, tau) = −(x + (1 − tau)·v_paper(x, tau)) / tau, with tau [B] broadcast over H, D."""
    tau_b = tau[:, None, None]
    return -(x + (1 - tau_b) * v_paper) / tau_b


def random_obs(n: int, generator: torch.Generator, dtype: torch.dtype = F64) -> Tensor:
    """Observations o ~ U[−1, 1], shape [n]."""
    return 2 * torch.rand(n, generator=generator, dtype=dtype) - 1


def task_cond(obs: Tensor, task_id: int) -> dict[str, Tensor]:
    """Cond of a batch that is entirely task `task_id`."""
    return make_cond(obs, torch.full(obs.shape, task_id, dtype=torch.long))


def gaussian_actions(n: int, generator: torch.Generator, dtype: torch.dtype = F64) -> Tensor:
    """Standard normal actions (or noise) of shape [n, H, D]."""
    return torch.randn(n, H, D, generator=generator, dtype=dtype)


def noisy_points(source: GMMTeacher, cond: Mapping[str, Tensor], tau: Tensor, generator: torch.Generator) -> Tensor:
    """A^tau = (1 − tau)·A + tau·eps in float64, A exact samples of `source`, eps ~ N(0, I)."""
    a = source.sample_exact(cond, tau.shape[0], generator).to(F64)
    return noisy_action(a, gaussian_actions(tau.shape[0], generator), tau)


def single_gaussian(mean: Tensor, sigma: float) -> GMMTeacher:
    """One-mode GMMTeacher with mean [4] and no observation shift."""
    return GMMTeacher(mean.to(F64)[None], torch.ones(1, dtype=F64), sigma, torch.zeros(4, dtype=F64))


def sorted_by_first_entry(actions: Tensor, *aligned: Tensor) -> list[Tensor]:
    """actions and its row-aligned tensors, all reordered by actions[:, 0, 0] (order-agnostic comparison)."""
    order = actions[:, 0, 0].argsort()
    return [actions[order], *(x[order] for x in aligned)]


@torch.no_grad()
def reinit_(module: nn.Module, seed: int) -> nn.Module:
    """Redraw every parameter from N(0, 1/fan_in) (vectors: N(0, 0.01)) with a seeded generator."""
    generator = make_generator(seed)
    for p in module.parameters():
        std = 1 / math.sqrt(p.shape[-1]) if p.dim() > 1 else 0.1
        p.copy_(std * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return module


class TinyVelocity(nn.Module):
    """Two-layer tanh MLP on [flattened actions, t, obs] → native velocity [B, H, D] (padding not zeroed)."""

    def __init__(self, seed: int, hidden: int = 8, dtype: torch.dtype = F64) -> None:
        """Draw all weights from N(0, 1/fan_in) and biases from N(0, 0.01) with a generator seeded by `seed`."""
        super().__init__()
        generator = make_generator(seed)
        n_in, n_out = H * D + 2, H * D

        def draw(*shape: int, std: float) -> nn.Parameter:
            """A new parameter of `shape` drawn from N(0, std²) with the seeded generator."""
            return nn.Parameter(std * torch.randn(*shape, generator=generator, dtype=dtype))

        self.w_in = draw(hidden, n_in, std=1 / math.sqrt(n_in))
        self.b_in = draw(hidden, std=0.1)
        self.w_out = draw(n_out, hidden, std=1 / math.sqrt(hidden))
        self.b_out = draw(n_out, std=0.1)

    def velocity(self, actions: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Native velocity in the parameter dtype."""
        dtype = self.w_in.dtype
        x = torch.cat([actions.flatten(1).to(dtype), t[:, None].to(dtype), cond["obs"][:, None].to(dtype)], 1)
        return (torch.tanh(x @ self.w_in.T + self.b_in) @ self.w_out.T + self.b_out).view(actions.shape)


class ConstantVelocity:
    """Native velocity equal to the fixed tensor c [B, H, D], whatever the inputs."""

    def __init__(self, c: Tensor) -> None:
        """Store the constant velocity c."""
        self.c = c

    def velocity(self, actions: Tensor, t: Tensor, cond: Any) -> Tensor:
        """Return a copy of c."""
        return self.c.clone()


class SpyVelocity(nn.Module):
    """Wraps a VelocityModel and records each call; with `forbidden` set, any call fails the test."""

    def __init__(self, inner: Any, name: str = "model") -> None:
        """Wrap `inner` (registered as a submodule if it is one); `name` labels failure messages."""
        super().__init__()
        self.inner = inner
        self.name = name
        self.forbidden = False
        self.calls: list[dict[str, Any]] = []

    def velocity(self, actions: Tensor, t: Tensor, cond: Any) -> Tensor:
        """Delegate to the wrapped model and record (actions, t, cond, grad mode, output)."""
        if self.forbidden:
            raise AssertionError(f"{self.name}.velocity must not be called in this configuration")
        out = self.inner.velocity(actions, t, cond)
        record = {"actions": actions.detach().clone(), "t": t.detach().clone(), "cond": cond}
        self.calls.append({**record, "grad": torch.is_grad_enabled(), "output": out.detach().clone()})
        return out


class ShiftVelocity(nn.Module):
    """Native velocity equal to a trainable constant `mean` [H, D], so FlowSolve(z) = z + mean ~ N(mean, I)."""

    def __init__(self, mean: Tensor) -> None:
        """Start the trainable mean at `mean` [H, D]."""
        super().__init__()
        self.mean = nn.Parameter(mean.clone())

    def velocity(self, actions: Tensor, t: Tensor, cond: Any) -> Tensor:
        """The constant velocity broadcast to the batch, in the actions' dtype."""
        return self.mean.expand(actions.shape).to(actions.dtype)


class GaussianFake(nn.Module):
    """Exact fake network of ShiftVelocity samples: the native velocity of N(mean, I) on the real dims."""

    def __init__(self, mean: Tensor) -> None:
        """Start the trainable mean at `mean` [H, D] (same parameter name and shape as ShiftVelocity)."""
        super().__init__()
        self.mean = nn.Parameter(mean.clone())

    def velocity(self, actions: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Exact velocity of the one-mode unit-variance GMM at the current mean (differentiable in the mean)."""
        return single_gaussian(to_real(self.mean[None])[0], 1.0).velocity(actions, t, cond)
