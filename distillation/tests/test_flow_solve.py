"""Spec test 10 — FlowSolve. With N = 200 Euler steps the exact GMM teacher's FlowSolve samples must match the
exact distribution: their sliced W2 to exact samples stays within a small margin of the noise floor (exact vs.
exact, independent draws, same sample size); the negative control shows that a one-step solve (which collapses
onto the posterior mean) is far above the floor, so the check has power. With N = 4 a recording model must see
t = 0, 0.25, 0.5, 0.75 in order, each as a [B] tensor in z's dtype, at the running Euler iterate, and a
constant native velocity c must give z + c (GR00T's `get_action` schedule)."""

import pytest
import torch

from distillation.flow_utils import flow_solve
from distillation.tests.helpers import (
    F64,
    ConstantVelocity,
    SpyVelocity,
    TinyVelocity,
    gaussian_actions,
    make_generator,
    task_cond,
)
from distillation.tests.mixture import TASK_IDS, GMMTeacher, sliced_w2, to_real

NUM_SAMPLES = 2000
NUM_REPEATS = 3
OBS_VALUE = 0.3
FLOOR_MARGIN = 1.25


def mean_sliced_w2_to_exact(teacher: GMMTeacher, task_id: int, num_steps: int | None) -> float:
    """Mean over repeats of sliced W2 between FlowSolve(num_steps) samples (None: exact samples) and exact ones."""
    cond = task_cond(torch.full((NUM_SAMPLES,), OBS_VALUE, dtype=F64), task_id)
    distances = []
    for repeat in range(NUM_REPEATS):
        generator = make_generator(1000 * task_id + repeat)
        reference = teacher.sample_exact(cond, NUM_SAMPLES, generator)
        if num_steps is None:
            candidate = teacher.sample_exact(cond, NUM_SAMPLES, generator)
        else:
            candidate = flow_solve(teacher, gaussian_actions(NUM_SAMPLES, generator), cond, num_steps,
                                   requires_grad=False)
        distances.append(sliced_w2(to_real(candidate), to_real(reference), make_generator(repeat)))
    return sum(distances) / len(distances)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_10_flow_solve_200_steps_matches_exact_distribution(teachers: dict[int, GMMTeacher], task_id: int) -> None:
    """Test 10: sliced W2(FlowSolve N = 200, exact) ≤ FLOOR_MARGIN × noise floor (exact vs. exact)."""
    noise_floor = mean_sliced_w2_to_exact(teachers[task_id], task_id, num_steps=None)
    flow_w2 = mean_sliced_w2_to_exact(teachers[task_id], task_id, num_steps=200)
    assert flow_w2 <= FLOOR_MARGIN * noise_floor, f"flow W2 {flow_w2:.4f} vs noise floor {noise_floor:.4f}"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_10_negative_control_one_step_solve_is_far_from_floor(teachers: dict[int, GMMTeacher], task_id: int) -> None:
    """Test 10 control: a 1-step solve (all samples at the posterior mean) is > 3× the noise floor."""
    noise_floor = mean_sliced_w2_to_exact(teachers[task_id], task_id, num_steps=None)
    assert mean_sliced_w2_to_exact(teachers[task_id], task_id, num_steps=1) > 3 * noise_floor


def test_10_flow_solve_4_steps_runs_documented_schedule() -> None:
    """Test 10: N = 4 evaluates t = 0, 1/4, 2/4, 3/4 at a_k = z + k·c/4 and a constant velocity c gives z + c."""
    generator, n = make_generator(10), 3
    z, c = gaussian_actions(n, generator), gaussian_actions(n, generator)
    z_before = z.clone()
    spy = SpyVelocity(ConstantVelocity(c))
    out = flow_solve(spy, z, task_cond(torch.zeros(n, dtype=F64), 1), 4, requires_grad=False)
    assert len(spy.calls) == 4
    for k, call in enumerate(spy.calls):
        assert call["t"].shape == (n,) and call["t"].dtype == F64
        assert torch.equal(call["t"], torch.full((n,), k / 4, dtype=F64))
        torch.testing.assert_close(call["actions"], z + k * c / 4, rtol=0, atol=1e-14)
    torch.testing.assert_close(out, z + c, rtol=0, atol=1e-14)
    assert torch.equal(z, z_before)


def test_10_flow_solve_output_dtype_follows_z() -> None:
    """Test 10: velocities are cast to z's dtype (float64 model, float32 z → float32 output, no promotion)."""
    generator, n = make_generator(11), 4
    z, c = gaussian_actions(n, generator, dtype=torch.float32), gaussian_actions(n, generator)
    out = flow_solve(ConstantVelocity(c), z, task_cond(torch.zeros(n, dtype=F64), 1), 4, requires_grad=False)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, z + c.float(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_10_flow_solve_gradient_mode(requires_grad: bool) -> None:
    """Test 10: requires_grad=True keeps the graph to the model's parameters; False returns a graph-free tensor."""
    generator, n, model = make_generator(12), 4, TinyVelocity(seed=12)
    out = flow_solve(model, gaussian_actions(n, generator), task_cond(torch.zeros(n, dtype=F64), 1), 4,
                     requires_grad=requires_grad)
    assert out.requires_grad is requires_grad
    if requires_grad:
        grads = torch.autograd.grad(out.square().sum(), list(model.parameters()))
        assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
