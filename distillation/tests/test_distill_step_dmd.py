"""Spec test 11, continued — what the DMD branch of `distill_step` computes and where. The direction test runs one
step with α_DMD = 1, α_FM = 0 and plain SGD on a ShiftVelocity student (FlowSolve samples N(μ_S, I)), an exact
fake (GaussianFake) and a unit-variance Gaussian teacher, and checks that the student's mean moves along μ_T − μ_S
(swapping v_fake and v_teacher would move it away). The spy tests run a DMD-only step with a ShiftVelocity student
whose samples sit far from the noise (mean 3) and check that every input of the fake network during its K_ψ
updates, and the point where the fake and teacher velocities are evaluated, is a noisy version
(1 − τ)·A^S + τ·ε of that step's own student sample, with per-sample τ ∈ [0, 1) and a standard-normal implied ε
(so not A^S itself, not pure noise or another sample, not a fixed τ), and that teacher and fake share A^τ and τ."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor

from distillation.flow_utils import native_to_paper_time
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from distillation.tests.helpers import (
    GaussianFake,
    ShiftVelocity,
    SpyVelocity,
    make_generator,
    single_gaussian,
    sorted_by_first_entry,
    toy_cond,
)
from distillation.toy.gmm_teacher import make_teachers
from distillation.toy.problem import D, H, action_mask, from_real, sample_batch, to_real

MU_TEACHER = torch.tensor([0.0, 0.0, 0.3, 0.15])
STUDENT_SHIFT = torch.tensor([0.6, -0.4, 0.6, -0.4])
STUDENT_OFFSET = 3.0
BATCH_SIZE = 256


def test_11_dmd_step_moves_student_mean_toward_teacher() -> None:
    """Test 11 (direction): α_DMD = 1, exact fake, plain SGD — the student's mean moves along μ_T − μ_S."""
    mu_student = from_real((MU_TEACHER + STUDENT_SHIFT)[None])[0]
    student, fake = ShiftVelocity(mu_student), GaussianFake(mu_student)
    router = TeacherRouter({1: single_gaussian(MU_TEACHER, 1.0)})
    batch = {"cond": toy_cond(torch.zeros(BATCH_SIZE), 1), "task_ids": torch.ones(BATCH_SIZE, dtype=torch.long),
             "action_mask": action_mask(BATCH_SIZE)}
    cfg = DistillConfig(alpha_dmd=1.0, alpha_fm=0.0, k_fake=1, lambda_ida=1.0)
    opt_student = torch.optim.SGD(student.parameters(), lr=0.5)
    opt_fake = torch.optim.SGD(fake.parameters(), lr=0.0)
    distill_step(student, fake, router, batch, cfg, opt_student, opt_fake, make_generator(122))
    new_mean = student.mean.detach()
    move = to_real((new_mean - mu_student)[None])[0]
    assert move.norm().item() > 0.05
    assert F.cosine_similarity(move, -STUDENT_SHIFT, dim=0).item() > 0.999
    assert torch.equal(new_mean[..., 2], mu_student[..., 2])


def run_spied_dmd_step(k_fake: int = 2) -> SimpleNamespace:
    """One DMD-only distill_step with spies on a ShiftVelocity student (mean 3), its fake and the exact teachers."""
    batch = sample_batch(BATCH_SIZE, make_generator(120))
    student = SpyVelocity(ShiftVelocity(torch.full((H, D), STUDENT_OFFSET)), "student")
    fake = SpyVelocity(ShiftVelocity(torch.zeros(H, D)), "fake")
    teachers = {k: SpyVelocity(teacher, f"teacher{k}") for k, teacher in make_teachers().items()}
    cfg = DistillConfig(alpha_dmd=1.0, alpha_fm=0.0, k_fake=k_fake, lambda_ida=0.5)
    opt_student = torch.optim.SGD(student.parameters(), lr=1e-2)
    opt_fake = torch.optim.SGD(fake.parameters(), lr=1e-2)
    distill_step(student, fake, TeacherRouter(teachers), batch, cfg, opt_student, opt_fake, make_generator(121))
    assert len(student.calls) == cfg.num_flow_steps
    last = student.calls[-1]
    a_student = last["actions"] + (1.0 / cfg.num_flow_steps) * last["output"]
    return SimpleNamespace(fake=fake, teachers=teachers, batch=batch, cfg=cfg, a_student=a_student)


def implied_noise(x: Tensor, tau: Tensor, a_student: Tensor, mask: Tensor, min_tau: float = 1e-2) -> Tensor:
    """ε solving x = (1 − τ)·A^S + τ·ε, on the masked-in entries of samples with τ > min_tau."""
    keep = tau > min_tau
    eps = (x - (1 - tau)[:, None, None] * a_student) / tau[:, None, None]
    return eps[keep][mask[keep]]


def assert_noisy_version_of_student(x: Tensor, t: Tensor, a_student: Tensor, mask: Tensor) -> None:
    """x = (1 − τ)·A^S + τ·ε with τ = 1 − t drawn per sample in [0, 1) and an implied ε that is N(0, 1)-like."""
    tau = native_to_paper_time(t)
    assert tau.shape == a_student.shape[:1]
    assert bool(((tau >= 0) & (tau < 1)).all())
    assert tau.unique().numel() > tau.numel() // 2, "tau must be drawn per sample"
    assert tau.min().item() < 0.1 and tau.max().item() > 0.9, "tau must cover [0, 1)"
    eps = implied_noise(x, tau, a_student, mask)
    assert abs(eps.mean().item()) < 0.15, f"implied eps mean {eps.mean().item():.3f}: x is not built from A^S"
    assert 0.85 < eps.std().item() < 1.15, f"implied eps std {eps.std().item():.3f}: x is not built from A^S"


def test_11_fake_is_trained_on_noisy_versions_of_this_steps_student_sample() -> None:
    """Test 11: each K_ψ fake update sees (1 − τ)·A^S + τ·ε of this step's own A^S, with fresh per-sample τ, ε."""
    rig = run_spied_dmd_step(k_fake=2)
    training_calls = [call for call in rig.fake.calls if call["grad"]]
    assert len(training_calls) == rig.cfg.k_fake
    for call in training_calls:
        assert_noisy_version_of_student(call["actions"], call["t"], rig.a_student, rig.batch["action_mask"])
        assert torch.equal(call["cond"]["obs"], rig.batch["cond"]["obs"])
    assert not torch.equal(training_calls[0]["t"], training_calls[1]["t"])
    assert not torch.equal(training_calls[0]["actions"], training_calls[1]["actions"])


def test_11_dmd_velocities_are_evaluated_at_a_noisy_student_sample() -> None:
    """Test 11: v_ψ is evaluated once, without gradients, at A^τ = (1 − τ)·A^S + τ·ε with per-sample τ (not A^S)."""
    rig = run_spied_dmd_step()
    dmd_calls = [call for call in rig.fake.calls if not call["grad"]]
    assert len(dmd_calls) == 1
    assert_noisy_version_of_student(dmd_calls[0]["actions"], dmd_calls[0]["t"], rig.a_student,
                                    rig.batch["action_mask"])


def test_11_teacher_and_fake_velocities_share_the_same_noisy_point() -> None:
    """Test 11: each teacher is called once, without gradients, on exactly the fake's A^τ and τ rows of its task."""
    rig = run_spied_dmd_step()
    (fake_call,) = [call for call in rig.fake.calls if not call["grad"]]
    for k, spy in rig.teachers.items():
        assert len(spy.calls) == 1 and not spy.calls[0]["grad"]
        sel = rig.batch["task_ids"] == k
        got = sorted_by_first_entry(spy.calls[0]["actions"], spy.calls[0]["t"])
        want = sorted_by_first_entry(fake_call["actions"][sel], fake_call["t"][sel])
        assert all(torch.equal(got_x, want_x) for got_x, want_x in zip(got, want))
