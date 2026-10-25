"""Spec tests 8 and 9 — teacher routing and IDA. Test 8 sends a mixed, unsorted two-task batch through
`TeacherRouter` and compares every sample with a per-sample call of its own teacher (velocity and FlowSolve),
checks that each teacher is called once on exactly its own subset (nested cond included) and that unknown task
ids or unsupported cond types fail before any teacher runs; its negative control builds a router with the
teachers swapped and shows it produces the swapped velocities. Test 9 checks IDA: λ = 1 leaves the fake network
bit-identical, λ = 0 makes it bit-identical to the student, frozen parameters are skipped, and mismatched
parameter names or shapes raise before anything is modified."""

import pytest
import torch
from torch import Tensor, nn

from distillation.dmd_loss import ida_update
from distillation.flow_utils import flow_solve, velocity_paper
from distillation.interfaces import VelocityModel
from distillation.router import TeacherRouter
from distillation.tests.helpers import (
    F64,
    SpyVelocity,
    TinyVelocity,
    gaussian_actions,
    make_generator,
    sorted_by_first_entry,
    task_cond,
)
from distillation.tests.mixture import make_cond

TASK_PATTERN = (1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2)
Inputs = tuple[Tensor, Tensor, Tensor, Tensor]
Teachers = dict[int, VelocityModel]


def mixed_batch(seed: int = 8) -> Inputs:
    """(a_tau, tau in [0.05, 0.95], obs, task_ids) for an unsorted batch mixing tasks 1 and 2."""
    generator, n = make_generator(seed), len(TASK_PATTERN)
    a_tau = gaussian_actions(n, generator)
    tau = 0.05 + 0.9 * torch.rand(n, generator=generator, dtype=F64)
    obs = 2 * torch.rand(n, generator=generator, dtype=F64) - 1
    return a_tau, tau, obs, torch.tensor(TASK_PATTERN)


def route(teachers: Teachers, method: str, inputs: Inputs) -> Tensor:
    """Router output for `method` ("velocity" or "flow") on the mixed batch."""
    a_tau, tau, obs, task_ids = inputs
    router, cond = TeacherRouter(teachers), make_cond(obs, task_ids)
    if method == "velocity":
        return router.velocity_paper(a_tau, tau, cond, task_ids)
    return router.flow_solve(a_tau, cond, task_ids, 4)


def per_sample_reference(teachers: Teachers, method: str, inputs: Inputs) -> Tensor:
    """Each sample run alone through its own teacher (batch of one), concatenated in batch order."""
    a_tau, tau, obs, task_ids = inputs
    rows = []
    for b, k in enumerate(task_ids.tolist()):
        cond_b = task_cond(obs[b:b + 1], k)
        if method == "velocity":
            rows.append(velocity_paper(teachers[k], a_tau[b:b + 1], tau[b:b + 1], cond_b))
        else:
            rows.append(flow_solve(teachers[k], a_tau[b:b + 1], cond_b, 4, requires_grad=False))
    return torch.cat(rows)


@pytest.mark.parametrize("method", ["velocity", "flow"])
def test_08_router_uses_each_samples_teacher_and_keeps_order(teachers: Teachers, method: str) -> None:
    """Test 8: every sample of a mixed batch gets its own teacher's output, in the original order."""
    inputs = mixed_batch()
    torch.testing.assert_close(route(teachers, method, inputs), per_sample_reference(teachers, method, inputs),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["velocity", "flow"])
def test_08_negative_control_swapped_teachers_give_swapped_outputs(teachers: Teachers, method: str) -> None:
    """Test 8 control: a router with swapped teachers yields the swapped outputs, which differ for every sample."""
    inputs, swapped = mixed_batch(), {1: teachers[2], 2: teachers[1]}
    out_swapped = route(swapped, method, inputs)
    torch.testing.assert_close(out_swapped, per_sample_reference(swapped, method, inputs), rtol=1e-12, atol=1e-12)
    per_sample_gap = (out_swapped - route(teachers, method, inputs)).abs().amax(dim=(1, 2))
    assert per_sample_gap.min().item() > 1e-3


def test_08_router_calls_each_teacher_once_on_its_own_subset(teachers: Teachers) -> None:
    """Test 8: one call per teacher, on exactly its samples, with the (nested) cond subset aligned to them."""
    a_tau, tau, obs, task_ids = mixed_batch()
    spies = {k: SpyVelocity(teacher, f"teacher{k}") for k, teacher in teachers.items()}
    cond = {**make_cond(obs, task_ids), "extra": {"obs_copy": obs[:, None].repeat(1, 3)}}
    TeacherRouter(spies).velocity_paper(a_tau, tau, cond, task_ids)
    for k, spy in spies.items():
        assert len(spy.calls) == 1
        call, sel = spy.calls[0], task_ids == k
        assert torch.equal(call["cond"]["task"], torch.full((int(sel.sum()),), k))
        got = sorted_by_first_entry(call["actions"], call["t"], call["cond"]["obs"], call["cond"]["extra"]["obs_copy"])
        want = sorted_by_first_entry(a_tau[sel], 1 - tau[sel], obs[sel], obs[sel][:, None].repeat(1, 3))
        for got_x, want_x in zip(got, want):
            torch.testing.assert_close(got_x, want_x, rtol=0, atol=1e-15)


@pytest.mark.parametrize("method", ["velocity", "flow"])
def test_08_unknown_task_id_raises_before_any_teacher_call(teachers: Teachers, method: str) -> None:
    """Test 8: a task id without a teacher raises ValueError naming it, and no teacher has been called."""
    a_tau, tau, obs, _ = mixed_batch()
    spies = {k: SpyVelocity(teacher, f"teacher{k}") for k, teacher in teachers.items()}
    task_ids = torch.tensor(TASK_PATTERN[:-1] + (3,))
    with pytest.raises(ValueError, match="3"):
        route(spies, method, (a_tau, tau, obs, task_ids))
    assert all(not spy.calls for spy in spies.values())


def test_08_unsupported_cond_type_raises_type_error(teachers: Teachers) -> None:
    """Test 8: cond that is neither a Tensor nor a Mapping of Tensors raises TypeError."""
    a_tau, tau, obs, task_ids = mixed_batch()
    with pytest.raises(TypeError):
        TeacherRouter(teachers).velocity_paper(a_tau, tau, [obs, task_ids], task_ids)


def test_08_router_flow_solve_carries_no_gradient() -> None:
    """Test 8: router.flow_solve returns teacher samples without an autograd graph, even for trainable teachers."""
    a_tau, _, obs, task_ids = mixed_batch()
    out = TeacherRouter({1: TinyVelocity(seed=1), 2: TinyVelocity(seed=2)}).flow_solve(
        a_tau, make_cond(obs, task_ids), task_ids, 4)
    assert not out.requires_grad and out.grad_fn is None


def parameters_of(module: nn.Module) -> dict[str, Tensor]:
    """Detached clones of all named parameters."""
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def assert_parameters_equal(module: nn.Module, expected: dict[str, Tensor]) -> None:
    """Every named parameter of `module` is bit-identical to `expected`."""
    got = dict(module.named_parameters())
    assert got.keys() == expected.keys()
    assert all(torch.equal(got[name], expected[name]) for name in expected)


def test_09_ida_lambda_one_leaves_fake_unchanged() -> None:
    """Test 9: λ = 1 leaves every fake parameter bit-identical."""
    fake, student = TinyVelocity(seed=1), TinyVelocity(seed=2)
    before = parameters_of(fake)
    ida_update(fake, student, 1.0)
    assert_parameters_equal(fake, before)


def test_09_ida_lambda_zero_copies_student_and_keeps_fake_trainable() -> None:
    """Test 9: λ = 0 makes the fake bit-identical to the student; the student is untouched; params stay leaves."""
    fake, student = TinyVelocity(seed=1), TinyVelocity(seed=2)
    student_before = parameters_of(student)
    ida_update(fake, student, 0.0)
    assert_parameters_equal(fake, student_before)
    assert_parameters_equal(student, student_before)
    assert all(p.is_leaf and p.requires_grad and p.grad_fn is None for p in fake.parameters())


def test_09_ida_interpolates_and_skips_frozen_parameters() -> None:
    """Test 9: ψ ← λψ + (1 − λ)θ on trainable parameters; a frozen parameter keeps its value."""
    fake, student = TinyVelocity(seed=1), TinyVelocity(seed=2)
    for module in (fake, student):
        module.b_out.requires_grad_(False)
    psi, theta = parameters_of(fake), parameters_of(student)
    ida_update(fake, student, 0.9)
    for name, p in fake.named_parameters():
        want = psi[name] if name == "b_out" else 0.9 * psi[name] + 0.1 * theta[name]
        torch.testing.assert_close(p.detach(), want, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("kind", ["shape", "names", "trainable_set"])
def test_09_ida_mismatched_parameters_raise_without_modifying(kind: str) -> None:
    """Test 9: different parameter shapes, names or trainable sets raise ValueError; the fake is unchanged."""
    fake, student = TinyVelocity(seed=1), TinyVelocity(seed=2, hidden=4 if kind == "shape" else 8)
    if kind == "names":
        student = SpyVelocity(student)
    if kind == "trainable_set":
        student.w_in.requires_grad_(False)
    before = parameters_of(fake)
    with pytest.raises(ValueError):
        ida_update(fake, student, 0.5)
    assert_parameters_equal(fake, before)
