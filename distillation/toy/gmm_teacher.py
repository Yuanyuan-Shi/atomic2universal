"""Exact Gaussian-mixture teacher for the toy problem.

GMMTeacher is a VelocityModel whose velocity is the exact flow-matching velocity of a Gaussian mixture over the
real 4-D action vector (see problem.py), so the toy has known answers. For component i with weight pi_i and mean
mu_i(o) = mu_i + o * s, the noisy action x = (1 - tau) A + tau eps at paper time tau is N((1 - tau) mu_i, Var I)
with Var = (1 - tau)^2 sigma^2 + tau^2 (shared by all components, so it cancels in the responsibilities):
  r_i ∝ pi_i N(x; (1 - tau) mu_i, Var I),
  E[A | x, i] = mu_i + (1 - tau) sigma^2 / Var (x - (1 - tau) mu_i),   E[eps | x, i] = tau / Var (x - (1 - tau) mu_i),
  v_paper = sum_i r_i (E[eps | x, i] - E[A | x, i]),                     score = -sum_i r_i (x - (1 - tau) mu_i) / Var.
The native velocity is -v_paper at tau = 1 - t; both conversions go through flow_utils. All math runs in float64;
outputs take the input dtype and are zero in the padding dim. The class holds no trainable parameters.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from distillation.flow_utils import native_to_paper_time, paper_to_native_velocity
from distillation.toy.problem import OBS_SHIFT, SIGMA, TASK_IDS, from_real, mode_means, to_real

F64 = torch.float64


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
    """The exact toy teachers keyed by task id: equal mode weights, std SIGMA, observation shift OBS_SHIFT."""
    return {
        k: GMMTeacher(mode_means(k), torch.ones(mode_means(k).shape[0], dtype=F64), SIGMA, torch.tensor(OBS_SHIFT))
        for k in TASK_IDS
    }
