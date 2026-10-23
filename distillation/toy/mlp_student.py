"""MLP velocity network for the toy student (and the fake network, which is a copy of it).

MLPVelocity is a VelocityModel in the native convention (t = 0 noise, t = 1 action). Its input is the flattened
real action vector [B, 4], a sinusoidal embedding of t, the observation o and a one-hot of the task id; three hidden
layers of 256 units with SiLU map it to the real velocity [B, 4], which is reshaped to [B, H, D] with zeros in the
padding dim. Inputs are cast to the parameter dtype and the output is in the parameter dtype, so the same network
runs in bf16, float32 or float64. Weights are initialised with PyTorch's default Linear scheme, drawn from an
explicit torch.Generator when one is given.
"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from distillation.flow_utils import at_least_fp32
from distillation.toy.problem import H, REAL_D, TASK_IDS, from_real, to_real


class MLPVelocity(nn.Module):
    """Native-convention velocity MLP: [real actions, sin/cos(t), o, task one-hot] -> real velocity."""

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
