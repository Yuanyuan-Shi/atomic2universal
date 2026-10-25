"""Spec tests 6 and 7 — masks and precision. Test 6 replaces every padded entry by a large value, +inf and nan
and checks that no masked reduction, DMD direction, surrogate loss (or its gradient) or flow-matching loss
changes, and that the test models ignore padded inputs; its negative control shows that a norm which multiplies
by the mask instead of selecting turns the same +inf into nan. Test 7 feeds bf16 velocities and networks and
checks that g and every loss are float32 and agree with the float32 computation to bf16 tolerance; its control
shows that doing the DMD arithmetic in bf16 would be caught."""

import copy

import pytest
import torch
from torch import Tensor

from distillation.dmd_loss import dmd_direction, dmd_surrogate_loss, flow_matching_loss
from distillation.flow_utils import masked_mean_sq, masked_norm, masked_sum_sq
from distillation.tests.helpers import F64, gaussian_actions, make_generator, masked, reinit_, task_cond
from distillation.tests.mixture import D, H, GMMTeacher, MLPVelocity, action_mask

NORM_EPS = 1e-8
PADDING_FILLS = (1e6, float("inf"), float("nan"))


def with_padding(x: Tensor, mask: Tensor, fill: float) -> Tensor:
    """x with every padded entry replaced by `fill` (masked-in entries unchanged)."""
    return torch.where(mask, x, torch.full_like(x, fill))


def random_mask(n: int, seed: int) -> Tensor:
    """Per-entry random boolean mask [n, H, D] (as with mixed embodiments), at least one entry in per sample."""
    mask = torch.rand(n, H, D, generator=make_generator(seed)) < 0.6
    mask[:, 0, 0] = True
    return mask


@pytest.mark.parametrize("mask_kind", ["fixed", "random"])
def test_06_masked_reductions_ignore_padding(mask_kind: str) -> None:
    """Test 6: masked_sum_sq / masked_norm / masked_mean_sq match the oracle and ignore padded values."""
    x = gaussian_actions(5, make_generator(60))
    mask = action_mask(5) if mask_kind == "fixed" else random_mask(5, seed=61)
    sum_sq = masked(x, mask).square().flatten(1).sum(dim=1)
    torch.testing.assert_close(masked_sum_sq(x, mask), sum_sq, rtol=1e-14, atol=0)
    torch.testing.assert_close(masked_norm(x, mask), sum_sq.sqrt(), rtol=1e-14, atol=0)
    torch.testing.assert_close(masked_mean_sq(x, mask), sum_sq.sum() / mask.sum(), rtol=1e-14, atol=0)
    for fill in PADDING_FILLS:
        x_bad = with_padding(x, mask, fill)
        assert torch.equal(masked_sum_sq(x_bad, mask), masked_sum_sq(x, mask))
        assert torch.equal(masked_norm(x_bad, mask), masked_norm(x, mask))
        assert torch.equal(masked_mean_sq(x_bad, mask), masked_mean_sq(x, mask))


def test_06_masked_mean_sq_rejects_empty_mask() -> None:
    """Test 6 support: a mean over zero masked-in entries is undefined and raises."""
    x = gaussian_actions(2, make_generator(62))
    with pytest.raises(ValueError):
        masked_mean_sq(x, torch.zeros_like(x, dtype=torch.bool))


def test_06_negative_control_multiplied_mask_turns_inf_into_nan() -> None:
    """Test 6 control: (x·mask) with +inf in padding gives a nan norm; masked_norm stays finite and exact."""
    x, mask = gaussian_actions(4, make_generator(63)), action_mask(4)
    x_bad = with_padding(x, mask, float("inf"))
    multiplied_norm = (x_bad * mask).square().flatten(1).sum(dim=1).sqrt()
    assert torch.isnan(multiplied_norm).all()
    assert torch.equal(masked_norm(x_bad, mask), masked_norm(x, mask))


@pytest.mark.parametrize("mask_kind", ["fixed", "random"])
def test_06_dmd_direction_ignores_padding(mask_kind: str) -> None:
    """Test 6: padded values of v_fake and v_teacher never change g, and g is zero in padding."""
    generator, n = make_generator(64), 6
    v_fake, v_teacher = gaussian_actions(n, generator), gaussian_actions(n, generator)
    tau = torch.rand(n, generator=generator, dtype=F64)
    mask = action_mask(n) if mask_kind == "fixed" else random_mask(n, seed=65)
    g = dmd_direction(v_fake, v_teacher, tau, mask, NORM_EPS)
    assert torch.count_nonzero(g[~mask]) == 0
    for fill in PADDING_FILLS:
        g_bad = dmd_direction(with_padding(v_fake, mask, fill), with_padding(v_teacher, mask, -fill), tau, mask,
                              NORM_EPS)
        assert torch.equal(g_bad, g)


def surrogate_loss_and_grad(a_student: Tensor, g: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """(L_DMD, ∂L_DMD/∂A^S) for a leaf copy of a_student."""
    a_leaf = a_student.clone().requires_grad_(True)
    loss = dmd_surrogate_loss(a_leaf, g, mask)
    (grad,) = torch.autograd.grad(loss, a_leaf)
    return loss.detach(), grad


def test_06_surrogate_loss_and_gradient_ignore_padding() -> None:
    """Test 6: padded values of A^S and g change neither L_DMD nor its gradient; the padded gradient is 0."""
    generator, n = make_generator(66), 5
    a_student, g, mask = gaussian_actions(n, generator), gaussian_actions(n, generator), action_mask(n)
    loss, grad = surrogate_loss_and_grad(a_student, g, mask)
    assert torch.count_nonzero(grad[~mask]) == 0
    for fill in PADDING_FILLS:
        loss_bad, grad_bad = surrogate_loss_and_grad(with_padding(a_student, mask, fill),
                                                     with_padding(g, mask, fill), mask)
        assert torch.equal(loss_bad, loss)
        assert torch.equal(grad_bad, grad)


def small_model(kind: str, teacher: GMMTeacher) -> object:
    """The exact teacher, or a seeded float64 MLPVelocity student."""
    return teacher if kind == "teacher" else reinit_(MLPVelocity(), seed=67).to(F64)


@pytest.mark.parametrize("kind", ["teacher", "student"])
def test_06_flow_matching_loss_ignores_padding(teachers: dict[int, GMMTeacher], kind: str) -> None:
    """Test 6: with the same τ, ε draws, padded values of the clean action never change L_FM."""
    n, model = 64, small_model(kind, teachers[1])
    cond, mask = task_cond(2 * torch.rand(n, generator=make_generator(68), dtype=F64) - 1, 1), action_mask(n)
    a_clean = teachers[1].sample_exact(cond, n, make_generator(69)).to(F64)
    loss = flow_matching_loss(model, a_clean, cond, mask, make_generator(70))
    assert torch.isfinite(loss)
    for fill in PADDING_FILLS:
        loss_bad = flow_matching_loss(model, with_padding(a_clean, mask, fill), cond, mask, make_generator(70))
        assert torch.equal(loss_bad, loss)


@pytest.mark.parametrize("kind", ["teacher", "student"])
def test_06_models_ignore_padded_inputs(teachers: dict[int, GMMTeacher], kind: str) -> None:
    """Test 6 support: mixture teacher and MLP student outputs are zero in padding and blind to padded inputs."""
    generator, n, model = make_generator(71), 8, small_model(kind, teachers[2])
    a, t, mask = gaussian_actions(n, generator), torch.rand(n, generator=generator, dtype=F64), action_mask(n)
    cond = task_cond(2 * torch.rand(n, generator=generator, dtype=F64) - 1, 2)
    v = model.velocity(a, t, cond)
    assert v.dtype == F64 and torch.count_nonzero(v[~mask]) == 0
    for fill in PADDING_FILLS:
        assert torch.equal(model.velocity(with_padding(a, mask, fill), t, cond), v)


def bf16_velocity_pair(n: int = 64) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(v_fake, v_teacher, tau, mask): independent float32 velocities, float32 tau, the problem's mask."""
    generator = make_generator(72)
    v_fake = gaussian_actions(n, generator, dtype=torch.float32)
    v_teacher = gaussian_actions(n, generator, dtype=torch.float32)
    return v_fake, v_teacher, torch.rand(n, generator=generator), action_mask(n)


def test_07_bf16_velocities_give_float32_direction_within_bf16_tolerance() -> None:
    """Test 7: bf16 inputs give a float32 g, exact up to fp32 rounding of the bf16 values, and within 1e-2 of fp32."""
    v_fake, v_teacher, tau, mask = bf16_velocity_pair()
    g_fp32 = dmd_direction(v_fake, v_teacher, tau, mask, NORM_EPS)
    g_bf16 = dmd_direction(v_fake.bfloat16(), v_teacher.bfloat16(), tau, mask, NORM_EPS)
    g_rounded_inputs = dmd_direction(v_fake.bfloat16().float(), v_teacher.bfloat16().float(), tau, mask, NORM_EPS)
    assert g_fp32.dtype == torch.float32 and g_bf16.dtype == torch.float32
    torch.testing.assert_close(g_bf16, g_rounded_inputs, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(g_bf16, g_fp32, rtol=0, atol=1e-2)


def test_07_negative_control_bf16_arithmetic_is_detected() -> None:
    """Test 7 control: the same g computed with bf16 arithmetic misses the fp32-arithmetic check by > 1e-4."""
    v_fake, v_teacher, tau, mask = bf16_velocity_pair()
    v_diff = v_fake.bfloat16() - v_teacher.bfloat16()
    norm = masked(v_diff, mask).square().flatten(1).sum(dim=1).sqrt()
    g_bf16_math = masked(-(1 - tau.bfloat16())[:, None, None] * v_diff / (norm[:, None, None] + NORM_EPS), mask)
    g_rounded_inputs = dmd_direction(v_fake.bfloat16().float(), v_teacher.bfloat16().float(), tau, mask, NORM_EPS)
    assert (g_bf16_math.float() - g_rounded_inputs).abs().max().item() > 1e-4


def test_07_reductions_and_surrogate_loss_are_float32_for_bf16_inputs() -> None:
    """Test 7: masked reductions and L_DMD (with its float32 value) are float32 when A^S is bf16."""
    generator, n = make_generator(73), 16
    mask = action_mask(n)
    a_bf16 = gaussian_actions(n, generator, torch.float32).bfloat16()
    g = gaussian_actions(n, generator, torch.float32)
    for reduce in (masked_sum_sq, masked_norm, masked_mean_sq):
        assert reduce(a_bf16, mask).dtype == torch.float32
    loss_bf16 = dmd_surrogate_loss(a_bf16.clone().requires_grad_(True), g, mask)
    loss_rounded = dmd_surrogate_loss(a_bf16.float(), g, mask)
    assert loss_bf16.dtype == torch.float32
    torch.testing.assert_close(loss_bf16.detach(), loss_rounded, rtol=1e-6, atol=0)


def test_07_flow_matching_loss_is_float32_for_bf16_network(teachers: dict[int, GMMTeacher]) -> None:
    """Test 7: L_FM of a bf16 copy of the student is float32 and within bf16 tolerance of the fp32 loss."""
    n, model_fp32 = 128, reinit_(MLPVelocity(), seed=74)
    model_bf16 = copy.deepcopy(model_fp32).to(torch.bfloat16)
    cond, mask = task_cond(2 * torch.rand(n, generator=make_generator(75)) - 1, 1), action_mask(n)
    a_clean = teachers[1].sample_exact(cond, n, make_generator(76)).float()
    loss_fp32 = flow_matching_loss(model_fp32, a_clean, cond, mask, make_generator(77))
    loss_bf16 = flow_matching_loss(model_bf16, a_clean, cond, mask, make_generator(77))
    assert loss_fp32.dtype == torch.float32 and loss_bf16.dtype == torch.float32
    torch.testing.assert_close(loss_bf16, loss_fp32, rtol=2e-2, atol=0)
