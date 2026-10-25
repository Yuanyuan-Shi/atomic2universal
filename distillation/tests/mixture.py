"""Gaussian-mixture test problem with known answers for the CPU tests of the shared distillation modules.

Actions use GR00T's layout [B, H, D] with H = 2 steps and D = 3 dims: dims 0-1 are real, dim 2 is padding (mask
false). The real part is the 4-D vector [step1_x, step1_y, step2_x, step2_y]. Each task is a Gaussian mixture over
that vector with equal weights and isotropic std SIGMA; mode i has mean (m_i, m_i + DELTA) + o * OBS_SHIFT, where m_i
is a step-1 centre from TASK_CENTERS and o in [-1, 1] is the observation.

`GMMTeacher` is a VelocityModel whose velocity is the exact flow-matching velocity of such a mixture. For component i
with weight pi_i and mean mu_i(o) = mu_i + o * s, the noisy action x = (1 - tau) A + tau eps at paper time tau is
N((1 - tau) mu_i, Var I) with Var = (1 - tau)^2 sigma^2 + tau^2 (shared by all components, so it cancels in the
responsibilities):
  r_i ∝ pi_i N(x; (1 - tau) mu_i, Var I),
  E[A | x, i] = mu_i + (1 - tau) sigma^2 / Var (x - (1 - tau) mu_i),   E[eps | x, i] = tau / Var (x - (1 - tau) mu_i),
  v_paper = sum_i r_i (E[eps | x, i] - E[A | x, i]),                     score = -sum_i r_i (x - (1 - tau) mu_i) / Var.
The native velocity is -v_paper at tau = 1 - t; both conversions go through flow_utils. All math runs in float64;
outputs take the input dtype and are zero in the padding dim.

`MLPVelocity` is a trainable native-convention velocity network for this problem: [real actions, sin/cos(t), o, task
one-hot] → real velocity, in the parameter dtype (bf16, float32 or float64), zeros in the padding dim.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from distillation.flow_utils import at_least_fp32, native_to_paper_time, paper_to_native_velocity

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
F64 = torch.float64


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
    centres = torch.tensor(TASK_CENTERS[task_id], dtype=F64)
    return torch.cat([centres, centres + torch.tensor(DELTA, dtype=F64)], dim=1)


def make_cond(obs: Tensor, task: Tensor) -> dict[str, Tensor]:
    """Conditioning {"obs": float [B], "task": long [B]}."""
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


def sliced_w2(x: Tensor, y: Tensor, generator: torch.Generator, n_proj: int = 128) -> float:
    """Sliced Wasserstein-2 distance between equal-size samples x, y [N, d] over n_proj random directions."""
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(f"x and y must be [N, d] of equal shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    dirs = torch.randn(x.shape[1], n_proj, generator=generator, device=x.device, dtype=F64)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    x_proj = torch.sort(x.to(F64) @ dirs, dim=0).values
    y_proj = torch.sort(y.to(F64) @ dirs, dim=0).values
    return float((x_proj - y_proj).pow(2).mean().sqrt())


class GMMTeacher:
    """Gaussian mixture over the real 4-D action vector with exact native velocity, score and sampling."""

    def __init__(self, means: Tensor, weights: Tensor, sigma: float, obs_shift: Tensor) -> None:
        """Store means [M, 4], normalised weights [M], sigma and obs_shift [4] in float64; raise on bad shapes."""
        if means.dim() != 2 or weights.shape != means.shape[:1] or obs_shift.shape != means.shape[1:]:
            raise ValueError(
                f"need means [M, d], weights [M], obs_shift [d]; got {tuple(means.shape)}, "
                f"{tuple(weights.shape)}, {tuple(obs_shift.shape)}"
            )
        if sigma <= 0 or bool((weights <= 0).any()):
            raise ValueError(f"sigma and weights must be positive, got sigma={sigma}, weights={weights.tolist()}")
        self.means = means.to(F64)
        self.weights = weights.to(F64) / weights.to(F64).sum()
        self.sigma = float(sigma)
        self.obs_shift = obs_shift.to(F64)

    @property
    def num_modes(self) -> int:
        """Number of mixture components M."""
        return self.means.shape[0]

    def centers(self, obs: Tensor) -> Tensor:
        """Shifted mode means mu_i(o) [B, M, 4] in float64 for observations obs [B]."""
        means, shift = self.means.to(obs.device), self.obs_shift.to(obs.device)
        return means[None] + obs.to(F64)[:, None, None] * shift[None, None]

    def velocity(self, actions: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Exact native velocity [B, H, D] at native time t (input dtype, zeros in padding)."""
        tau = native_to_paper_time(t.to(F64))
        v_paper = self._velocity_paper_real(to_real(actions).to(F64), tau, self._obs(cond, actions.shape[0]))
        return from_real(paper_to_native_velocity(v_paper)).to(actions.dtype)

    def exact_score(self, x: Tensor, tau: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Exact score grad log p_tau(x) [B, H, D] at paper time tau (input dtype, zeros in padding)."""
        _, resid, var, r = self._posterior(to_real(x).to(F64), tau.to(F64), self._obs(cond, x.shape[0]))
        score = -(r[..., None] * resid).sum(dim=1) / var[:, None]
        return from_real(score).to(x.dtype)

    def sample_exact(self, cond: Mapping[str, Tensor], n: int, generator: torch.Generator) -> Tensor:
        """n exact samples [n, H, D] at observations cond["obs"] [n] (dtype of obs, zeros in padding)."""
        obs = self._obs(cond, n)
        idx = torch.multinomial(self.weights.to(obs.device), n, replacement=True, generator=generator)
        mu = self.centers(obs)[torch.arange(n, device=obs.device), idx]
        noise = torch.randn(mu.shape, generator=generator, device=obs.device, dtype=F64)
        return from_real(mu + self.sigma * noise).to(obs.dtype)

    def _posterior(self, x: Tensor, tau: Tensor, obs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Means mu [B, M, 4], residuals x - (1 - tau) mu [B, M, 4], Var [B] and responsibilities r [B, M]."""
        mu = self.centers(obs)
        resid = x[:, None, :] - (1.0 - tau)[:, None, None] * mu
        var = (1.0 - tau) ** 2 * self.sigma**2 + tau**2
        logits = self.weights.to(x.device).log() - 0.5 * resid.pow(2).sum(dim=-1) / var[:, None]
        return mu, resid, var, torch.softmax(logits, dim=1)

    def _velocity_paper_real(self, x: Tensor, tau: Tensor, obs: Tensor) -> Tensor:
        """Exact paper-convention velocity [B, 4] on real vectors: sum_i r_i (E[eps | x, i] - E[A | x, i])."""
        mu, resid, var, r = self._posterior(x, tau, obs)
        mean_a = mu + ((1.0 - tau) * self.sigma**2 / var)[:, None, None] * resid
        mean_eps = (tau / var)[:, None, None] * resid
        return (r[..., None] * (mean_eps - mean_a)).sum(dim=1)

    @staticmethod
    def _obs(cond: Mapping[str, Tensor], batch_size: int) -> Tensor:
        """The observations cond["obs"], checked to be [batch_size]."""
        obs = cond["obs"]
        if obs.shape != (batch_size,):
            raise ValueError(f"cond['obs'] must have shape [{batch_size}], got {tuple(obs.shape)}")
        return obs


def make_teachers() -> dict[int, GMMTeacher]:
    """The exact teachers keyed by task id: equal mode weights, std SIGMA, observation shift OBS_SHIFT."""
    return {
        k: GMMTeacher(mode_means(k), torch.ones(mode_means(k).shape[0], dtype=F64), SIGMA, torch.tensor(OBS_SHIFT))
        for k in TASK_IDS
    }


class MLPVelocity(nn.Module):
    """Native-convention velocity MLP: [real actions, sin/cos(t), o, task one-hot] -> real velocity. Weights use
    PyTorch's default Linear init, drawn from `generator` when one is given."""

    def __init__(
        self,
        hidden: int = 256,
        num_layers: int = 3,
        time_dim: int = 64,
        task_ids: Sequence[int] = TASK_IDS,
        generator: torch.Generator | None = None,
    ) -> None:
        """Build the MLP; with a generator, its initial weights are drawn from it."""
        super().__init__()
        if time_dim % 2 or num_layers < 1:
            raise ValueError(f"time_dim must be even and num_layers >= 1, got {time_dim}, {num_layers}")
        self.time_dim = time_dim
        self.register_buffer("task_ids", torch.tensor(list(task_ids)), persistent=False)
        widths = [H * REAL_D + time_dim + 1 + len(task_ids)] + [hidden] * num_layers
        layers: list[nn.Module] = []
        for width_in, width_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(width_in, width_out), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, H * REAL_D))
        if generator is not None:
            self._init_weights(generator)

    def velocity(self, actions: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Native velocity [B, H, D] in the parameter dtype, zeros in the padding dim."""
        dtype = self.net[0].weight.dtype
        features = torch.cat(
            [
                to_real(actions).to(dtype),
                self._time_embedding(t).to(dtype),
                cond["obs"].to(dtype)[:, None],
                self._task_one_hot(cond["task"]).to(dtype),
            ],
            dim=1,
        )
        return from_real(self.net(features))

    forward = velocity

    def _time_embedding(self, t: Tensor) -> Tensor:
        """Sinusoidal embedding [B, time_dim] of t in [0, 1] (scaled by 1000, max period 10^4), >= float32."""
        t = at_least_fp32(t)
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=t.dtype, device=t.device) / half)
        args = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def _task_one_hot(self, task: Tensor) -> Tensor:
        """One-hot [B, num_tasks] of task ids; raises on ids the network was not built for."""
        one_hot = task[:, None] == self.task_ids[None, :]
        if not bool(one_hot.any(dim=1).all()):
            raise ValueError(f"unknown task ids {sorted(set(task.tolist()) - set(self.task_ids.tolist()))}")
        return one_hot

    @torch.no_grad()
    def _init_weights(self, generator: torch.Generator) -> None:
        """Re-draw every Linear layer with PyTorch's default init, using `generator`."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                bound = 1.0 / math.sqrt(layer.in_features)
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
