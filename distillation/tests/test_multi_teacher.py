"""Multi-teacher distillation with K = 3 teachers. CPU: three small synthetic velocity models behind one
`TeacherRouter` (each sample served by its own task's teacher, outputs in batch order), task-balanced buffer batches
over three tasks whose ids are the router's, the trainer's refusal of a buffer that lacks a teacher's task, and one
`distill_step` on a mixed batch of the three tasks. GPU (skips without CUDA, the checkpoints or the round-0
collections): one update through train.py's `Distiller` with three GR00T teachers (Expert I1, Expert I2 and
checkpoint-60000 as a stand-in third teacher) on six recorded observations, two per task (-s prints the values)."""

import contextlib
import copy
import gc
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from distillation.flow_utils import flow_solve, velocity_paper
from distillation.gr00t.buffer import ObservationBuffer, collection_dir, task_id
from distillation.gr00t.train import Distiller, load_config, project_path
from distillation.interfaces import DistillConfig
from distillation.router import TeacherRouter
from distillation.step import distill_step
from distillation.tests.helpers import SpyVelocity, TinyVelocity, make_generator
from distillation.tests.mixture import D, H, MLPVelocity, action_mask, make_cond
from distillation.tests.test_gr00t_data import make_collection
from evaluation.config import PROJECT_ROOT

TASKS = (1, 2, 3)
NUM_STEPS = 4
CONFIG = PROJECT_ROOT / "configs/distillation/mtd_pnp_cabinet_drawer.yaml"
COLLECTIONS = PROJECT_ROOT / "distillation/runs/collections"
DATA_DIRS = [collection_dir(COLLECTIONS, "ckpt60k", role, 0) for role in ("i1", "i2")]
GPU_TASK_IDS = [1, 1, 3, 2, 2, 3]  # six ckpt-60000 observations (i1 × 3, i2 × 3), one of each relabeled as task 3


def three_teachers(dtype: torch.dtype = torch.float32) -> dict[int, SpyVelocity]:
    """Spied TinyVelocity teachers with different weights, keyed by task id 1, 2, 3."""
    return {k: SpyVelocity(TinyVelocity(seed=200 + k, dtype=dtype), f"teacher{k}") for k in TASKS}


def mixed_batch(task_ids: list[int], seed: int) -> dict:
    """Batch of the given task ids: observations o ~ U[−1, 1] and the problem's action mask."""
    task = torch.tensor(task_ids)
    obs = 2 * torch.rand(len(task_ids), generator=make_generator(seed)) - 1
    return {"cond": make_cond(obs, task), "task_ids": task, "action_mask": action_mask(len(task_ids))}


def changed(model: nn.Module, before: list[torch.Tensor]) -> bool:
    """Whether any parameter of `model` differs from its copy in `before`."""
    return any(not torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_router_serves_each_sample_with_its_own_teacher() -> None:
    """Velocity and FlowSolve of a mixed batch over three tasks equal each teacher run on its own samples alone; each
    teacher sees only its task's samples, once per velocity evaluation; an id without a teacher raises."""
    teachers = three_teachers(torch.float64)
    router, batch = TeacherRouter(teachers), mixed_batch([3, 1, 2, 2, 3, 1, 1, 3, 2], seed=1)
    generator = make_generator(2)
    cond, ids = batch["cond"], batch["task_ids"]
    a_tau, z = (torch.randn(len(ids), H, D, generator=generator, dtype=torch.float64) for _ in range(2))
    tau = torch.rand(len(ids), generator=generator, dtype=torch.float64)
    v, samples = router.velocity_paper(a_tau, tau, cond, ids), router.flow_solve(z, cond, ids, NUM_STEPS)
    for k, teacher in teachers.items():
        sel = ids == k
        own = {key: value[sel] for key, value in cond.items()}
        assert len(teacher.calls) == 1 + NUM_STEPS and all(bool((c["cond"]["task"] == k).all()) for c in teacher.calls)
        torch.testing.assert_close(v[sel], velocity_paper(teacher.inner, a_tau[sel], tau[sel], own))
        torch.testing.assert_close(samples[sel], flow_solve(teacher.inner, z[sel], own, NUM_STEPS, requires_grad=False))
    with pytest.raises(ValueError, match="no teacher"):
        router.velocity_paper(a_tau, tau, cond, torch.where(ids == 3, 4, ids))


def test_buffer_batches_are_balanced_over_three_tasks(tmp_path: Path) -> None:
    """Collections of 90, 30 and 10 observations on roles i1–i3: balanced batches draw each task ≈ 1/3 of the time and
    their ids are the router's task ids; the trainer refuses a buffer that lacks one teacher's task."""
    sizes = {"i1": (9, 10), "i2": (3, 10), "i3": (2, 5)}
    dirs = [make_collection(tmp_path, role, n, per_episode, start=1000 * k)
            for k, (role, (n, per_episode)) in enumerate(sizes.items())]
    _, tasks = ObservationBuffer(dirs).sample(6000, rng=np.random.default_rng(0))
    router = TeacherRouter(three_teachers())
    assert [task_id(role) for role in sizes] == sorted(set(tasks.tolist())) == sorted(router.teachers) == list(TASKS)
    assert all(abs(np.mean(tasks == k) - 1 / 3) < 0.03 for k in TASKS)
    with pytest.raises(ValueError, match="differ from the teachers"):
        Distiller.train(SimpleNamespace(router=router), ObservationBuffer(dirs[:2]), 1, print)


def test_one_distill_step_with_three_teachers() -> None:
    """One DMD + FM update on a mixed batch of the three tasks: finite metrics; each teacher runs without gradients
    and only on its own samples, in both branches; student and fake change, the teachers do not."""
    teachers = three_teachers()
    student = MLPVelocity(task_ids=TASKS, generator=make_generator(3))
    fake = copy.deepcopy(student)
    models = {"student": student, "fake": fake, **{f"teacher{k}": t for k, t in teachers.items()}}
    before = {name: [p.detach().clone() for p in model.parameters()] for name, model in models.items()}
    cfg = DistillConfig(alpha_dmd=0.5, alpha_fm=0.5, k_fake=2, lambda_ida=0.99)
    out = distill_step(student, fake, TeacherRouter(teachers), mixed_batch([1, 2, 3] * 4, seed=4), cfg,
                       torch.optim.SGD(student.parameters(), lr=1e-2), torch.optim.SGD(fake.parameters(), lr=1e-2),
                       make_generator(5))
    assert set(out) == {"loss_dmd", "loss_fm", "loss_fake", "vdiff_norm", "grad_norm"}
    assert all(math.isfinite(value) for value in out.values()), out
    for k, teacher in teachers.items():
        assert len(teacher.calls) == 1 + cfg.num_flow_steps  # the DMD velocity, then the FM branch's FlowSolve
        assert all(not c["grad"] and len(c["actions"]) == 4 and bool((c["cond"]["task"] == k).all())
                   for c in teacher.calls)
    assert changed(student, before["student"]) and changed(fake, before["fake"])
    assert not any(changed(teacher, before[f"teacher{k}"]) for k, teacher in teachers.items())


def weight_sums(model: nn.Module) -> torch.Tensor:
    """Per-tensor float64 sums of the model's weights (a cheap fingerprint)."""
    return torch.stack([p.detach().double().sum() for p in model.parameters()])


def gpu_missing() -> list[str]:
    """Checkpoints and collections the GPU test needs that are not on disk."""
    cfg = load_config(CONFIG)
    paths = [project_path(cfg["student"]["checkpoint"]), *(project_path(t["checkpoint"]) for t in cfg["teachers"])]
    return [str(p) for p in [*paths, *DATA_DIRS] if not p.exists()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the GR00T dry run needs CUDA")
@pytest.mark.skipif(bool(gpu_missing()), reason=f"needs the checkpoints and round-0 collections: {gpu_missing()}")
def test_gpu_one_update_with_three_gr00t_teachers() -> None:
    """`Distiller` with Expert I1, Expert I2 and checkpoint-60000 as teacher of a third role (i3) runs one update on six
    recorded observations of checkpoint-60000, two per task: finite metrics, every teacher call holds exactly that
    teacher's two samples, and the student's weights move."""
    cfg = load_config(CONFIG)
    stand_in = {"task": "stand-in", "role": "i3", "label": cfg["student"]["label"],
                "checkpoint": cfg["student"]["checkpoint"]}
    distiller = Distiller(cfg | {"teachers": [*cfg["teachers"], stand_in]})
    observations, recorded = ObservationBuffer(DATA_DIRS).holdout(len(GPU_TASK_IDS), seed=0)
    assert recorded.tolist() == [1, 1, 1, 2, 2, 2] and sorted(distiller.router.teachers) == list(TASKS)
    before = weight_sums(distiller.student)
    with contextlib.ExitStack() as stack:
        spies = {k: stack.enter_context(mock.patch.object(teacher, "velocity", wraps=teacher.velocity))
                 for k, teacher in distiller.router.teachers.items()}
        out = distiller._update(distiller.features.transform(observations), np.array(GPU_TASK_IDS))
    batch_sizes = {k: [call.args[0].shape[0] for call in spy.call_args_list] for k, spy in spies.items()}
    print(f"\n[multi-teacher] {out}; teacher call batch sizes {batch_sizes}", flush=True)
    assert all(math.isfinite(value) for value in out.values()), out
    assert all(sizes == [2] * (1 + NUM_STEPS) for sizes in batch_sizes.values()), batch_sizes
    assert not torch.equal(before, weight_sums(distiller.student))
    del distiller
    gc.collect()
    torch.cuda.empty_cache()
