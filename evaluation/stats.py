"""Binomial confidence intervals for small-sample success rates.

The Wilson score interval is used because it stays well behaved at 0/n and n/n,
where the normal approximation collapses to a zero-width interval.
"""

import math


def wilson_interval(num_success: int, num_trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return the (low, high) Wilson score interval for num_success out of num_trials."""
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if not 0 <= num_success <= num_trials:
        raise ValueError(f"num_success must be in [0, {num_trials}], got {num_success}")
    proportion = num_success / num_trials
    z_squared = z * z
    denominator = 1 + z_squared / num_trials
    center = (proportion + z_squared / (2 * num_trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / num_trials + z_squared / (4 * num_trials**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
