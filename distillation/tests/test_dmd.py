"""Spec tests 2–5 — the DMD direction and its surrogate loss. Test 2 checks the velocity form of g against the
score form w·(1 − τ)(s_ψ − s_T) built from two exact GMMs (float64, 1e-10). Test 3 checks that the surrogate's
gradient is g/B with respect to A^S and the manual Σ gᵀ∂A/∂θ / B with respect to the parameters of a tiny
float64 network (relative error < 1e-6). Test 4 checks that one small DMD step on samples of a shifted Gaussian
moves them toward an exact teacher (negative control: swapping fake and teacher moves them away). Test 5 checks
that identical teacher, student and fake give g = 0 and a zero gradient, directly and through `distill_step`."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from distillation.dmd_loss import dmd_direction, dmd_surrogate_loss
from distillation.flow_utils import flow_solve, noisy_action, velocity_paper
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from distillation.tests.helpers import (
    F64,
    TinyVelocity,
    gaussian_actions,
    make_generator,
    masked,
    noisy_points,
    random_obs,
    rel_err,
    single_gaussian,
    task_cond,
)
from distillation.tests.mixture import OBS_SHIFT, SIGMA, GMMTeacher, action_mask, make_teachers, mode_means, to_real

NORM_EPS = 1e-8


def fake_and_teacher_gmms() -> tuple[GMMTeacher, GMMTeacher]:
    """A 'fake' GMM (two shifted, wider, unequal modes) and the exact task-2 teacher (three modes)."""
    fake = GMMTeacher(mode_means(2)[:2] + 0.2, torch.tensor([0.3, 0.7], dtype=F64), 2 * SIGMA,
                      torch.tensor(OBS_SHIFT, dtype=F64))
    return fake, make_teachers()[2]


@pytest.mark.parametrize("norm_eps", [0.0, NORM_EPS, 1e-2])
@pytest.mark.parametrize("tau_values", [(1e-4,), (0.3,), (0.9,), (1e-4, 0.3, 0.9)])
def test_02_velocity_form_equals_score_form(tau_values: tuple[float, ...], norm_eps: float) -> None:
    """Test 2: g = w·(1 − τ)(s_ψ − s_T), w = 1/(‖s_ψ − s_T‖ + eps·(1 − τ)/τ), to 1e-10 in float64."""
    fake, teacher = fake_and_teacher_gmms()
    generator, n = make_generator(2), 30
    tau = torch.tensor(tau_values, dtype=F64).repeat(n // len(tau_values))
    cond, mask = task_cond(random_obs(n, generator), 2), action_mask(n)
    x = noisy_points(fake, cond, tau, generator)
    g = dmd_direction(velocity_paper(fake, x, tau, cond), velocity_paper(teacher, x, tau, cond), tau, mask, norm_eps)
    s_diff = masked(fake.exact_score(x, tau, cond) - teacher.exact_score(x, tau, cond), mask)
    w = 1 / (s_diff.flatten(1).norm(dim=1) + norm_eps * (1 - tau) / tau)
    expected = (w * (1 - tau))[:, None, None] * s_diff
    assert g.dtype == F64
    assert masked(g - expected, mask).abs().max().item() < 1e-10
    assert torch.count_nonzero(g[~mask]) == 0


def student_sample(model: nn.Module, n: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """(A^S = FlowSolve(model) with gradients, an arbitrary g with garbage in padding, the problem's mask)."""
    generator = make_generator(seed)
    cond = task_cond(random_obs(n, generator), 1)
    a_student = flow_solve(model, gaussian_actions(n, generator), cond, 4, requires_grad=True)
    return a_student, gaussian_actions(n, generator), action_mask(n)


def explicit_jacobian_vjp(a: Tensor, weights: Tensor, mask: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    """Σ over masked-in (b, h, d) of weights[b, h, d]·∂a[b, h, d]/∂θ, one Jacobian row per autograd call."""
    total = [torch.zeros_like(p) for p in params]
    for index in map(tuple, mask.nonzero().tolist()):
        rows = torch.autograd.grad(a[index], params, retain_graph=True)
        total = [acc + weights[index] * row for acc, row in zip(total, rows)]
    return total


def test_03_surrogate_gradient_wrt_student_actions_is_g_over_b() -> None:
    """Test 3: ∂L_DMD/∂A^S = g/B on masked-in entries and 0 in padding (relative error < 1e-6)."""
    a_student, g, mask = student_sample(TinyVelocity(seed=3), n=5, seed=3)
    loss = dmd_surrogate_loss(a_student, g, mask)
    (grad_a,) = torch.autograd.grad(loss, a_student)
    assert loss.dtype == F64
    assert rel_err(grad_a, masked(g, mask) / a_student.shape[0]) < 1e-6


def test_03_surrogate_parameter_gradient_equals_manual_sum() -> None:
    """Test 3: ∇_θ L_DMD = Σ gᵀ∂A^S/∂θ / B (explicit Jacobian rows and autograd VJP), rel. error < 1e-6."""
    model = TinyVelocity(seed=4)
    a_student, g, mask = student_sample(model, n=4, seed=4)
    params = list(model.parameters())
    weights = masked(g, mask) / a_student.shape[0]
    got = torch.autograd.grad(dmd_surrogate_loss(a_student, g, mask), params, retain_graph=True)
    manual = explicit_jacobian_vjp(a_student, weights, mask, params)
    vjp = torch.autograd.grad(a_student, params, grad_outputs=weights)
    flat = [torch.cat([t.flatten() for t in ts]) for ts in (got, manual, vjp)]
    assert flat[0].norm() > 0
    assert rel_err(flat[0], flat[1]) < 1e-6
    assert rel_err(flat[0], flat[2]) < 1e-6


MU_TEACHER = torch.tensor([0.0, 0.0, 0.3, 0.15], dtype=F64)
STUDENT_SHIFT = torch.tensor([0.6, -0.4, 0.6, -0.4], dtype=F64)


def dmd_step_on_shifted_gaussian(swap: bool, lr: float = 0.05, n: int = 2048) -> tuple[Tensor, Tensor, Tensor]:
    """(A, A after one SGD step on L_DMD, mask) for student samples of N(μ_T + shift, σ²) and an exact fake."""
    teacher, fake = single_gaussian(MU_TEACHER, SIGMA), single_gaussian(MU_TEACHER + STUDENT_SHIFT, SIGMA)
    generator = make_generator(4)
    cond, mask = task_cond(torch.zeros(n, dtype=F64), 1), action_mask(n)
    a = fake.sample_exact(cond, n, generator).to(F64)
    tau = torch.rand(n, generator=generator, dtype=F64)
    a_tau = noisy_action(a, gaussian_actions(n, generator), tau)
    v_fake, v_teacher = velocity_paper(fake, a_tau, tau, cond), velocity_paper(teacher, a_tau, tau, cond)
    if swap:
        v_fake, v_teacher = v_teacher, v_fake
    g = dmd_direction(v_fake, v_teacher, tau, mask, NORM_EPS)
    a_leaf = a.clone().requires_grad_(True)
    dmd_surrogate_loss(a_leaf, g, mask).backward()
    return a, (a_leaf - lr * n * a_leaf.grad).detach(), mask


def test_04_dmd_step_moves_samples_toward_teacher_mean() -> None:
    """Test 4: the sample mean gets closer to μ_T and every sample moves along μ_T − μ_S; padding is untouched."""
    a, a_new, mask = dmd_step_on_shifted_gaussian(swap=False)
    mean_before, mean_after = to_real(a).mean(dim=0), to_real(a_new).mean(dim=0)
    assert (mean_after - MU_TEACHER).norm() < (mean_before - MU_TEACHER).norm() - 1e-3
    moves = to_real(a_new - a)
    cosine = F.cosine_similarity(moves, -STUDENT_SHIFT.expand_as(moves), dim=1)
    assert cosine.min().item() > 0.999
    assert torch.equal(a_new[~mask], a[~mask])


def test_04_negative_control_swapped_velocities_move_away() -> None:
    """Test 4 control: with fake and teacher swapped (sign error in g) the mean moves away from μ_T."""
    a, a_new, _ = dmd_step_on_shifted_gaussian(swap=True)
    mean_before, mean_after = to_real(a).mean(dim=0), to_real(a_new).mean(dim=0)
    assert (mean_after - MU_TEACHER).norm() > (mean_before - MU_TEACHER).norm() + 1e-3


def test_05_identical_networks_give_zero_direction_and_gradient() -> None:
    """Test 5: teacher = student = fake gives g = 0 exactly and an exactly zero DMD parameter gradient."""
    student = TinyVelocity(seed=5)
    fake, teacher = copy.deepcopy(student), copy.deepcopy(student)
    generator, n = make_generator(5), 6
    cond, mask = task_cond(random_obs(n, generator), 1), action_mask(n)
    a_student = flow_solve(student, gaussian_actions(n, generator), cond, 4, requires_grad=True)
    tau = torch.rand(n, generator=generator, dtype=F64)
    a_tau = noisy_action(a_student.detach(), gaussian_actions(n, generator), tau)
    with torch.no_grad():
        v_teacher = TeacherRouter({1: teacher}).velocity_paper(a_tau, tau, cond, torch.ones(n, dtype=torch.long))
        v_fake = velocity_paper(fake, a_tau, tau, cond)
    g = dmd_direction(v_fake, v_teacher, tau, mask, NORM_EPS)
    assert torch.count_nonzero(g) == 0
    loss = dmd_surrogate_loss(a_student, g, mask)
    grads = torch.autograd.grad(loss, list(student.parameters()))
    assert loss.item() == 0.0
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)


def test_05_identity_through_distill_step_gives_zero_loss_and_gradient() -> None:
    """Test 5 via distill_step (α_DMD = 1, frozen fake optimizer): loss_dmd, vdiff_norm and grad_norm are 0."""
    student = TinyVelocity(seed=6, dtype=torch.float32)
    fake, teacher = copy.deepcopy(student), copy.deepcopy(student)
    n = 8
    obs = random_obs(n, make_generator(6), dtype=torch.float32)
    batch = {"cond": task_cond(obs, 1), "task_ids": torch.ones(n, dtype=torch.long), "action_mask": action_mask(n)}
    cfg = DistillConfig(alpha_dmd=1.0, alpha_fm=0.0, k_fake=1, lambda_ida=0.5)
    opt_student = torch.optim.SGD(student.parameters(), lr=0.1)
    opt_fake = torch.optim.SGD(fake.parameters(), lr=0.0)
    out = distill_step(student, fake, TeacherRouter({1: teacher}), batch, cfg, opt_student, opt_fake,
                       make_generator(7))
    assert out["loss_dmd"] == 0.0 and out["vdiff_norm"] == 0.0 and out["grad_norm"] == 0.0
