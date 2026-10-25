"""Spec test 11 — `distill_step`. Student, fake network and teachers are wrapped in recording spies, and the
optimizers count their steps, so the tests can prove which branches ran: with α_DMD = 0 there is no student
FlowSolve sample, no fake velocity and no fake update (the fake and the router's velocity are made to fail if
touched) while IDA still runs; with α_FM = 0 there is no teacher sampling (router.flow_solve fails if touched).
Every configuration must return exactly the contract's keys as finite Python floats, the reported gradient norm
must equal the student's gradient norm before the step, the run must be reproducible from its generator, and
DistillConfig must reject invalid settings."""

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from distillation.tests.helpers import SpyVelocity, make_generator, reinit_
from distillation.tests.mixture import MLPVelocity, make_teachers, sample_batch

DMD_KEYS = {"loss_dmd", "loss_fake", "vdiff_norm"}
BATCH_SIZE = 16


def make_rig(alpha_dmd: float, k_fake: int = 1, lambda_ida: float = 0.5) -> SimpleNamespace:
    """Spied student / fake MLPs (different seeds), spied exact teachers, a mixed-task batch, config and optimizers."""
    student = SpyVelocity(reinit_(MLPVelocity(), seed=110), "student")
    fake = SpyVelocity(reinit_(MLPVelocity(), seed=111), "fake")
    teachers = {k: SpyVelocity(teacher, f"teacher{k}") for k, teacher in make_teachers().items()}
    cfg = DistillConfig(alpha_dmd=alpha_dmd, alpha_fm=1.0 - alpha_dmd, k_fake=k_fake, lambda_ida=lambda_ida)
    rig = SimpleNamespace(student=student, fake=fake, teachers=teachers, router=TeacherRouter(teachers), cfg=cfg,
                          batch=sample_batch(BATCH_SIZE, make_generator(112)), fake_steps=0, student_grad_norms=[])
    rig.opt_student = torch.optim.SGD(student.parameters(), lr=1e-2)
    rig.opt_fake = torch.optim.SGD(fake.parameters(), lr=1e-2)
    rig.opt_fake.register_step_pre_hook(lambda *_: setattr(rig, "fake_steps", rig.fake_steps + 1))
    rig.opt_student.register_step_pre_hook(lambda *_: rig.student_grad_norms.append(total_grad_norm(student)))
    return rig


def total_grad_norm(module: nn.Module) -> float:
    """L2 norm of all parameter gradients of `module`, in float64."""
    return math.sqrt(sum(p.grad.double().square().sum().item() for p in module.parameters() if p.grad is not None))


def run_step(rig: SimpleNamespace, seed: int = 113) -> dict[str, Any]:
    """One distill_step on the rig."""
    return distill_step(rig.student, rig.fake, rig.router, rig.batch, rig.cfg, rig.opt_student, rig.opt_fake,
                        make_generator(seed))


def forbidden(name: str) -> Any:
    """A stand-in method that fails the test when called."""
    def fail(*args: Any, **kwargs: Any) -> None:
        """Raise AssertionError naming the forbidden call."""
        raise AssertionError(f"{name} must not be called in this configuration")
    return fail


def assert_finite_floats(out: dict[str, Any]) -> None:
    """Every returned metric is a finite Python float."""
    assert all(isinstance(value, float) and math.isfinite(value) for value in out.values()), out


def test_11_alpha_dmd_zero_skips_dmd_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 11: α_DMD = 0 runs no student sample, fake velocity, fake update or DMD routing; IDA still runs."""
    rig = make_rig(alpha_dmd=0.0, lambda_ida=0.0)
    rig.fake.forbidden = True
    monkeypatch.setattr(rig.router, "velocity_paper", forbidden("router.velocity_paper"))
    out = run_step(rig)
    assert set(out) == {"loss_fm", "grad_norm"}
    assert_finite_floats(out)
    assert len(rig.student.calls) == 1 and rig.student.calls[0]["grad"]
    assert rig.fake_steps == 0
    student_params = dict(rig.student.named_parameters())
    assert all(torch.equal(p, student_params[name]) for name, p in rig.fake.named_parameters())


def test_11_alpha_fm_zero_skips_fm_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 11: α_FM = 0 runs no teacher sampling and no student FM call; teachers only give velocities once."""
    rig = make_rig(alpha_dmd=1.0, k_fake=2)
    monkeypatch.setattr(rig.router, "flow_solve", forbidden("router.flow_solve"))
    out = run_step(rig)
    assert set(out) == DMD_KEYS | {"grad_norm"}
    assert_finite_floats(out)
    assert len(rig.student.calls) == rig.cfg.num_flow_steps
    assert all(call["grad"] for call in rig.student.calls)
    present = set(rig.batch["task_ids"].tolist())
    assert {k for k, spy in rig.teachers.items() if spy.calls} == present
    assert all(len(spy.calls) <= 1 and not spy.calls[0]["grad"] for spy in rig.teachers.values() if spy.calls)
    assert rig.fake_steps == rig.cfg.k_fake


@pytest.mark.parametrize("k_fake", [1, 3])
def test_11_full_step_runs_both_branches_with_finite_metrics(k_fake: int) -> None:
    """Test 11: α = 0.5 / 0.5 returns all five keys as finite floats and makes k_fake fake updates."""
    rig = make_rig(alpha_dmd=0.5, k_fake=k_fake)
    out = run_step(rig)
    assert set(out) == DMD_KEYS | {"loss_fm", "grad_norm"}
    assert_finite_floats(out)
    assert rig.fake_steps == k_fake
    assert len(rig.student.calls) == rig.cfg.num_flow_steps + 1
    assert len(rig.fake.calls) == k_fake + 1
    assert all(call["grad"] for call in rig.fake.calls[:-1]) and not rig.fake.calls[-1]["grad"]
    assert out["vdiff_norm"] > 0 and out["grad_norm"] > 0


@pytest.mark.parametrize("alpha_dmd", [0.0, 0.5, 1.0])
def test_11_grad_norm_is_student_gradient_norm_before_step(alpha_dmd: float) -> None:
    """Test 11: the returned grad_norm equals the student's total gradient norm when its optimizer steps."""
    rig = make_rig(alpha_dmd=alpha_dmd)
    out = run_step(rig)
    assert len(rig.student_grad_norms) == 1
    assert out["grad_norm"] == pytest.approx(rig.student_grad_norms[0], rel=1e-5)


def test_11_step_is_reproducible_from_its_generator() -> None:
    """Test 11: identical models, batch and generator seed give identical metrics and parameters."""
    rig_a, rig_b = make_rig(alpha_dmd=0.5), make_rig(alpha_dmd=0.5)
    assert run_step(rig_a) == run_step(rig_b)
    params_b = dict(rig_b.student.named_parameters())
    assert all(torch.equal(p, params_b[name]) for name, p in rig_a.student.named_parameters())


@pytest.mark.parametrize(
    "override",
    [{"alpha_dmd": 0.6}, {"alpha_dmd": 1.2, "alpha_fm": -0.2}, {"k_fake": 0}, {"lambda_ida": 1.5},
     {"lambda_ida": -0.1}, {"num_flow_steps": 0}, {"norm_eps": -1e-8}],
)
def test_11_distill_config_rejects_invalid_settings(override: dict[str, float]) -> None:
    """Test 11 support: DistillConfig raises ValueError for α's not summing to 1 and out-of-range fields."""
    base = {"alpha_dmd": 0.5, "alpha_fm": 0.5, "k_fake": 1, "lambda_ida": 0.99}
    DistillConfig(**base)
    with pytest.raises(ValueError):
        DistillConfig(**{**base, **override})
