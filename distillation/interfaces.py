"""Shared interfaces of the distillation code. `VelocityModel` is the one protocol every teacher,
student and fake network implements, always in its native (GR00T) convention; the shared modules
convert to the paper convention only through `flow_utils`. `DistillConfig` holds the hyper-parameters
of one distillation update and validates them on construction, so a bad run fails before any compute."""

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


class VelocityModel(Protocol):
    """Velocity network in its native convention (GR00T: t=0 noise, t=1 action, v = action − noise)."""

    def velocity(self, actions: Tensor, t: Tensor, cond: Any) -> Tensor:
        """Return the native velocity [B, H, D] at actions [B, H, D] and continuous time t [B] in [0, 1]."""


@dataclass(frozen=True)
class DistillConfig:
    """Hyper-parameters of one MTD update: loss weights, fake updates per step, IDA rate, solver steps."""

    alpha_dmd: float
    alpha_fm: float
    k_fake: int
    lambda_ida: float
    num_flow_steps: int = 4
    norm_eps: float = 1e-8

    def __post_init__(self) -> None:
        """Raise ValueError naming every field that is out of range."""
        checks = [
            (0.0 <= self.alpha_dmd <= 1.0, f"alpha_dmd must be in [0, 1], got {self.alpha_dmd}"),
            (0.0 <= self.alpha_fm <= 1.0, f"alpha_fm must be in [0, 1], got {self.alpha_fm}"),
            (
                abs(self.alpha_dmd + self.alpha_fm - 1.0) <= 1e-9,
                f"alpha_dmd + alpha_fm must equal 1, got {self.alpha_dmd + self.alpha_fm}",
            ),
            (isinstance(self.k_fake, int) and self.k_fake >= 1, f"k_fake must be an int >= 1, got {self.k_fake}"),
            (0.0 <= self.lambda_ida <= 1.0, f"lambda_ida must be in [0, 1], got {self.lambda_ida}"),
            (
                isinstance(self.num_flow_steps, int) and self.num_flow_steps >= 1,
                f"num_flow_steps must be an int >= 1, got {self.num_flow_steps}",
            ),
            (self.norm_eps >= 0.0, f"norm_eps must be >= 0, got {self.norm_eps}"),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("Invalid DistillConfig: " + "; ".join(errors))
