"""Spec test 1 — the convention adapter. The teacher's native velocity, converted by
`flow_utils.velocity_paper` and turned into a score with Eq. (9), s = −(x + (1 − τ)·v_paper)/τ, must equal the
teacher's exact score (float64, τ ∈ [0.05, 0.95]). Negative controls remove the sign flip (or the time flip)
inside the adapter and show the same check then fails by a wide margin. Small companion tests pin the four
conversion functions, `noisy_action` (τ = 1 is noise), the `at_least_fp32` promotion rule and the convention of
the shared flow-matching loss (its value for the exact teacher against an independent Monte Carlo estimate)."""

import pytest
import torch

from distillation import flow_utils
from distillation.dmd_loss import flow_matching_loss
from distillation.tests.helpers import (
    F64,
    gaussian_actions,
    make_generator,
    masked,
    noisy_points,
    random_obs,
    score_from_velocity,
    toy_cond,
)
from distillation.toy.gmm_teacher import GMMTeacher
from distillation.toy.problem import TASK_IDS, action_mask

TAU_LOW, TAU_HIGH = 0.05, 0.95
NUM_POINTS = 256


def adapter_score_error(teacher: GMMTeacher, task_id: int, seed: int) -> float:
    """Max abs error over masked-in entries of Eq. (9) on the adapter velocity vs. the exact score."""
    generator = make_generator(seed)
    cond = toy_cond(random_obs(NUM_POINTS, generator), task_id)
    tau = TAU_LOW + (TAU_HIGH - TAU_LOW) * torch.rand(NUM_POINTS, generator=generator, dtype=F64)
    x = noisy_points(teacher, cond, tau, generator)
    s_adapter = score_from_velocity(flow_utils.velocity_paper(teacher, x, tau, cond), x, tau)
    s_exact = teacher.exact_score(x, tau, cond)
    assert s_adapter.dtype == F64 and s_exact.dtype == F64
    return masked(s_adapter - s_exact, action_mask(NUM_POINTS)).abs().max().item()


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_01_adapter_eq9_score_matches_exact_score(teachers: dict[int, GMMTeacher], task_id: int) -> None:
    """Test 1: Eq. (9) applied to velocity_paper(teacher) equals exact_score to < 1e-5 (float64)."""
    assert adapter_score_error(teachers[task_id], task_id, seed=task_id) < 1e-5


@pytest.mark.parametrize("conversion", ["native_to_paper_velocity", "paper_to_native_time"])
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_01_negative_control_without_conversion_error_is_large(
    teachers: dict[int, GMMTeacher], monkeypatch: pytest.MonkeyPatch, conversion: str, task_id: int
) -> None:
    """Test 1 control: with the adapter's sign (or time) flip replaced by the identity, the error is large."""
    monkeypatch.setattr(flow_utils, conversion, lambda value: value)
    assert adapter_score_error(teachers[task_id], task_id, seed=task_id) > 0.1


def test_01_conversions_are_the_documented_maps() -> None:
    """Test 1 support: tau = 1 − t, v_paper = −v_native, and each pair of maps are mutual inverses."""
    x = torch.linspace(-1.5, 1.5, 7, dtype=F64)
    assert torch.equal(flow_utils.paper_to_native_time(x), 1 - x)
    assert torch.equal(flow_utils.native_to_paper_time(x), 1 - x)
    assert torch.equal(flow_utils.native_to_paper_velocity(x), -x)
    assert torch.equal(flow_utils.paper_to_native_velocity(x), -x)
    assert torch.equal(flow_utils.native_to_paper_time(flow_utils.paper_to_native_time(x)), x)
    assert torch.equal(flow_utils.paper_to_native_velocity(flow_utils.native_to_paper_velocity(x)), x)


def test_01_velocity_paper_negates_native_velocity_at_flipped_time() -> None:
    """Test 1 support: velocity_paper(model, a, tau) = −model.velocity(a, 1 − tau) for a recording model."""
    generator = make_generator(1)
    a_tau, v_native = gaussian_actions(4, generator), gaussian_actions(4, generator)
    tau = torch.rand(4, generator=generator, dtype=F64)
    seen = {}

    class Recorder:
        """VelocityModel that records its inputs and returns the fixed native velocity v_native."""

        def velocity(self, actions: torch.Tensor, t: torch.Tensor, cond: object) -> torch.Tensor:
            """Record (actions, t, cond) and return v_native."""
            seen.update(actions=actions, t=t, cond=cond)
            return v_native

    cond = {"obs": torch.zeros(4, dtype=F64)}
    v_paper = flow_utils.velocity_paper(Recorder(), a_tau, tau, cond)
    assert torch.equal(v_paper, -v_native)
    assert torch.equal(seen["t"], 1 - tau) and torch.equal(seen["actions"], a_tau) and seen["cond"] is cond


def test_01_noisy_action_is_paper_convention() -> None:
    """Test 1 support: A^tau = (1 − tau)·A + tau·eps per sample, so tau = 0 is the action and tau = 1 is noise."""
    generator = make_generator(2)
    a, eps = gaussian_actions(3, generator), gaussian_actions(3, generator)
    tau = torch.tensor([0.0, 0.25, 1.0], dtype=F64)
    a_tau = flow_utils.noisy_action(a, eps, tau)
    assert torch.equal(a_tau[0], a[0]) and torch.equal(a_tau[2], eps[2])
    torch.testing.assert_close(a_tau[1], 0.75 * a[1] + 0.25 * eps[1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_01_flow_matching_loss_uses_paper_convention(teachers: dict[int, GMMTeacher], task_id: int) -> None:
    """Test 1 support: L_FM of the exact teacher matches a Monte Carlo E‖v_paper − (ε − A)‖² to 5%."""
    teacher, n, generator = teachers[task_id], 20000, make_generator(30 + task_id)
    cond, mask = toy_cond(random_obs(n, generator), task_id), action_mask(n)
    a = teacher.sample_exact(cond, n, generator).to(F64)
    tau, eps = torch.rand(n, generator=generator, dtype=F64), gaussian_actions(n, generator)
    v = flow_utils.velocity_paper(teacher, flow_utils.noisy_action(a, eps, tau), tau, cond)
    oracle = (masked(v - (eps - a), mask).square().sum() / mask.sum()).item()
    zero_velocity = (masked(eps - a, mask).square().sum() / mask.sum()).item()
    loss = flow_matching_loss(teacher, a, cond, mask, make_generator(40 + task_id))
    assert loss.dtype == F64
    assert loss.item() == pytest.approx(oracle, rel=0.05)
    assert loss.item() < 0.5 * zero_velocity


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.bfloat16, torch.float32), (torch.float16, torch.float32), (torch.float32, torch.float32), (F64, F64)],
)
def test_01_at_least_fp32_promotion_rule(dtype: torch.dtype, expected: torch.dtype) -> None:
    """Test 1/7 support: half and bfloat16 are promoted to float32; float32 and float64 are kept."""
    assert flow_utils.at_least_fp32(torch.ones(2, dtype=dtype)).dtype == expected
