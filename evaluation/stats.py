"""Binomial confidence intervals for small-sample success rates and their differences.

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


def newcombe_interval(success_a: int, trials_a: int, success_b: int, trials_b: int,
                      z: float = 1.96) -> tuple[float, float]:
    """Return the (low, high) 95% interval for p_a - p_b from two independent samples.

    Newcombe's hybrid score method (1998, method 10): combines the two Wilson intervals.
    """
    p_a, p_b = success_a / trials_a, success_b / trials_b
    low_a, high_a = wilson_interval(success_a, trials_a, z)
    low_b, high_b = wilson_interval(success_b, trials_b, z)
    difference = p_a - p_b
    low = difference - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    high = difference + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return low, high
