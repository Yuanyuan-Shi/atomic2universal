"""GPU integration tests of the GR00T distillation code (spec §6, tests 1, 3–6) on the real checkpoints and on 20
observations that checkpoint-60000 recorded in its round-0 collections (10 per task, a buffer holdout). They skip
without CUDA or without the git-ignored checkpoints and collections; heavy objects load once per module. Measured
values are printed with a `[phase C]` prefix (run with -s); distillation/gr00t/RESULTS.md records them. What the CPU
tests on a tiny head (test_gr00t_wrapper.py) cover is not repeated: grouping by valid length, fp32 time buckets, the
memo, eval mode, perturb_. Tests 2 (shared backbone) and 7 (memory and speed) are in test_gr00t_resources.py; the
closed-loop part of test 6 (run_baseline on an export) is script-driven, see RESULTS.md."""

import copy
import gc
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch import Tensor

from distillation.dmd_loss import dmd_direction, dmd_surrogate_loss, flow_matching_loss
from distillation.flow_utils import flow_solve, masked_norm, noisy_action, velocity_paper
from distillation.gr00t.buffer import LANGUAGE_KEY, ObservationBuffer, RawObs, collection_dir
from distillation.gr00t.export import CKPT60K, export_student
from distillation.gr00t.features import BackboneFeatures
from distillation.gr00t.wrapper import GR00TVelocity
from distillation.router import TeacherRouter
from distillation.step import dmd_direction_at_noisy
from evaluation.config import PROJECT_ROOT
from expert_training.verify_expert import load_policy

DEVICE = torch.device("cuda")
EXPERT_I1 = PROJECT_ROOT / "checkpoints/experts/PickPlaceCounterToCabinet/step_95"
EXPERT_I2 = PROJECT_ROOT / "checkpoints/experts/PickPlaceDrawerToCounter/step_100"
COLLECTIONS = PROJECT_ROOT / "distillation/runs/collections"
DATA_DIRS = [collection_dir(COLLECTIONS, "ckpt60k", role, 0) for role in ("i1", "i2")]
N_OBS, NUM_STEPS, NORM_EPS = 20, 4, 1e-8
SINGLE_ATOL = 1e-6  # B = 1: measured bitwise equal; anything at bf16 level (≥ 1e-3) would fail
# B = 20, real dims (|a| ≤ 1.03): batched bf16 backbone output differs from B = 1 by ≈ 1.05% (relative Frobenius,
# measured), moving chunks by ≤ 0.080 (mean 1e-3); B = 1 features × (1 + 2^-8·N(0, 1)) already move them by ≤ 0.027.
BATCHED_ATOL, BATCHED_MEAN_ATOL = 0.15, 2e-3
FM_DRAWS, FM_VS_SFT, FLIP_RATIO = 8, 3.0, 20.0
PADDING_FILLS = (1e6, float("inf"), float("nan"))
MISSING = [str(p) for p in (CKPT60K, EXPERT_I1, EXPERT_I2, *DATA_DIRS) if not p.exists()]
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="GR00T integration tests need CUDA"),
              pytest.mark.skipif(bool(MISSING), reason=f"needs the checkpoints and round-0 collections: {MISSING}")]


def report(test: str, **values: float) -> None:
    """Print measured values (visible with pytest -s) for RESULTS.md."""
    print(f"\n[phase C] {test}: " + ", ".join(f"{key} = {float(v):.4g}" for key, v in values.items()), flush=True)


def policy_actions(checkpoint: Path, observations: Sequence[RawObs], z: Tensor) -> tuple[Tensor, set[torch.dtype]]:
    """GR00T's own eval path on each observation alone (B = 1, as the server runs it): Gr00tPolicy built as the server
    builds it (verify_expert.load_policy), Policy.act's request layout (evaluation/policy.py:74-81), apply_transforms,
    then the server's normalized-action computation (backbone + get_action under inference_mode and bf16 autocast,
    POL:188-193) with get_action's only RNG call replaced by z[b] (recon §1). Returns action_pred [N, 16, 32] fp32 on
    the CPU and the dtypes get_action asked its noise in; the policy is freed before returning."""
    policy, dtypes, actions = load_policy(Path(checkpoint)), set(), []
    for observation, z_b in zip(observations, z):
        request = {key: np.asarray(value)[None, None] for key, value in observation.items()
                   if key.startswith(("video.", "state."))}
        request[LANGUAGE_KEY] = np.array([observation[LANGUAGE_KEY]])

        def noise(*, size: tuple, dtype: torch.dtype, device: torch.device, z_b: Tensor = z_b) -> Tensor:
            assert tuple(size) == (1, *z_b.shape)
            dtypes.add(dtype)
            return z_b[None].to(device, dtype)
        with mock.patch("torch.randn", noise):
            actions.append(policy._get_action_from_normalized_input(policy.apply_transforms(request))[0].cpu())
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(actions), dtypes


def own_samples(model: GR00TVelocity, cond: dict[str, Tensor], generator: torch.Generator) -> Tensor:
    """`model`'s 4-step FlowSolve chunks from fp32 noise drawn from `generator` (no gradients)."""
    z = torch.randn(cond["action_mask"].shape, generator=generator, device=DEVICE)
    return flow_solve(model, z, cond, NUM_STEPS, requires_grad=False)


def direction_at_own_samples(student: GR00TVelocity, fake: GR00TVelocity, router: TeacherRouter,
                             cond: dict[str, Tensor], task_ids: np.ndarray, seed: int) -> tuple[Tensor, float]:
    """Step (3a) of distill_step (dmd_direction_at_noisy) around the student's own samples: g, mean ‖v_fake − v_T‖."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    a = own_samples(student, cond, generator)
    return dmd_direction_at_noisy(fake, router, a, cond, torch.as_tensor(task_ids, device=DEVICE), cond["action_mask"],
                                  NORM_EPS, generator)


@torch.no_grad()
def noisy_velocities(fake: GR00TVelocity, teacher: GR00TVelocity, cond: dict[str, Tensor],
                     seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """As step (3a) of distill_step around the teacher's own samples A: (A, tau, A^tau, v_fake, v_teacher), paper
    convention, tau ~ U[0, 1), both networks on the whole batch."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    a = own_samples(teacher, cond, generator)
    tau = torch.rand(a.shape[:1], generator=generator, device=DEVICE)
    a_tau = noisy_action(a, torch.randn(a.shape, generator=generator, device=DEVICE), tau)
    return a, tau, a_tau, velocity_paper(fake, a_tau, tau, cond), velocity_paper(teacher, a_tau, tau, cond)


def fm_loss(model: object, a_clean: Tensor, cond: dict[str, Tensor], seed: int, draws: int = 1) -> float:
    """Mean of `draws` flow-matching losses of `model` on `a_clean` (fresh tau, eps each, seeded generator)."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    with torch.no_grad():
        return float(np.mean([flow_matching_loss(model, a_clean, cond, cond["action_mask"], generator).item()
                              for _ in range(draws)]))


def with_padding(x: Tensor, mask: Tensor, fill: float) -> Tensor:
    """x with every padded entry replaced by `fill`."""
    return torch.where(mask, x, torch.full_like(x, fill))


class PaddingBlind:
    """GR00T behind a zeroed padded input and a `fill`ed padded output, so that only the padded entries of the loss
    inputs (targets, clean samples) vary between calls: GR00T itself reads padded inputs (recon §3)."""

    def __init__(self, model: GR00TVelocity, mask: Tensor, fill: float) -> None:
        """Wrap `model` for batches with action mask `mask`."""
        self.model, self.mask, self.fill = model, mask, fill

    def velocity(self, actions: Tensor, t: Tensor, cond: dict[str, Tensor]) -> Tensor:
        """Native velocity of the model at `actions` with zeroed padding; `fill` in the padded output dims."""
        return with_padding(self.model.velocity(torch.where(self.mask, actions, 0.0), t, cond), self.mask, self.fill)


def sft_loss(last: int = 10) -> float:
    """Mean of the last `last` logged SFT training losses of checkpoint-60000 (trainer_state.json, steps ≤ 60000)."""
    history = json.loads((CKPT60K / "trainer_state.json").read_text())["log_history"]
    return float(np.mean([entry["loss"] for entry in history if "loss" in entry][-last:]))


@pytest.fixture(scope="module")
def observations() -> tuple[list[RawObs], np.ndarray]:
    """N_OBS recorded observations of checkpoint-60000 (half per task, buffer holdout seed 0) and their task ids."""
    return ObservationBuffer(DATA_DIRS).holdout(N_OBS, seed=0)


@pytest.fixture(scope="module")
def z() -> Tensor:
    """One fixed fp32 noise chunk [16, 32] per observation, on the CPU."""
    return torch.randn(N_OBS, 16, 32, generator=torch.Generator().manual_seed(0))


@pytest.fixture(scope="module")
def reference(observations: tuple, z: Tensor) -> tuple[Tensor, set[torch.dtype]]:
    """GR00T's own eval-path actions of checkpoint-60000 on the observations with noise z (see policy_actions)."""
    return policy_actions(CKPT60K, observations[0], z)


@pytest.fixture(scope="module")
def features() -> BackboneFeatures:
    """The shared backbone, verified over all three checkpoints."""
    return BackboneFeatures([CKPT60K, EXPERT_I1, EXPERT_I2], DEVICE)


@pytest.fixture(scope="module")
def cond(features: BackboneFeatures, observations: tuple) -> dict[str, Tensor]:
    """cond of all N_OBS observations in one batch, as training builds it."""
    return features(observations[0])


@pytest.fixture(scope="module")
def teacher() -> GR00TVelocity:
    """checkpoint-60000 as a teacher: bf16, frozen (the eval's dtype)."""
    return GR00TVelocity.from_checkpoint(CKPT60K, torch.bfloat16, DEVICE, trainable=False)


@pytest.fixture(scope="module")
def fake32() -> GR00TVelocity:
    """checkpoint-60000 as a student or fake network: fp32 master weights, trainable (never updated here)."""
    return GR00TVelocity.from_checkpoint(CKPT60K, torch.float32, DEVICE, trainable=True)


def test_1_sampler_matches_get_action(reference: tuple, features: BackboneFeatures, teacher: GR00TVelocity,
                                      observations: tuple, z: Tensor, cond: dict[str, Tensor]) -> None:
    """flow_solve(GR00TVelocity(ckpt-60000, bf16), z, cond, 4) vs GR00T's own get_action with the same z. Per
    observation (B = 1, the eval's batch) it is bitwise equal: max abs ≤ SINGLE_ATOL over all 32 dims. With all 20 in
    one batch (as training) the bf16 kernels of the backbone (mostly) and the head depend on the batch size, so the
    real dims agree to BATCHED_ATOL max and BATCHED_MEAN_ATOL mean abs; pairing each chunk with another observation's
    reference (negative control) exceeds the mean bound 10×. get_action's noise is fp32 under the eval's autocast
    (recon §1.3)."""
    ref, noise_dtypes = reference
    assert noise_dtypes == {torch.float32}
    with torch.no_grad():
        single = torch.cat([flow_solve(teacher, z[b:b + 1].to(DEVICE), features([obs]), NUM_STEPS,
                                       requires_grad=False).cpu() for b, obs in enumerate(observations[0])])
        batched = flow_solve(teacher, z.to(DEVICE), cond, NUM_STEPS, requires_grad=False).cpu()
    real = cond["action_mask"][0, 0].cpu()
    err_single, err_batched = (single - ref).abs(), (batched - ref).abs()
    report("test 1", single_max_abs=err_single.amax(), batched_max_abs_real=err_batched[..., real].amax(),
           batched_mean_abs_real=err_batched[..., real].mean(), batched_max_abs_all=err_batched.amax(),
           max_abs_action_real=ref[..., real].abs().amax())
    assert err_single.amax() <= SINGLE_ATOL
    assert err_batched[..., real].amax() <= BATCHED_ATOL and err_batched[..., real].mean() <= BATCHED_MEAN_ATOL
    assert (batched - ref.roll(1, 0))[..., real].abs().mean() > 10 * BATCHED_MEAN_ATOL


def test_3_identity_gives_zero_dmd_direction(features: BackboneFeatures, teacher: GR00TVelocity,
                                             fake32: GR00TVelocity, observations: tuple,
                                             cond: dict[str, Tensor]) -> None:
    """student = fake = teacher = bf16 ckpt-60000 (fake an independent deep copy): on single-task batches of real
    observations, step (3a) of distill_step (dmd_direction_at_noisy, around the student's own samples) gives
    ‖v_fake − v_T‖ = 0 and g exactly 0. Measured, not asserted: the same on the mixed-task batch (the router runs the
    teacher per task subset, i.e. at other batch sizes), and the precision floor of the DMD signal: an fp32-master fake
    from ckpt-60000 vs the bf16 teacher, ‖v_fake − v_T‖ / ‖v_T‖ per sample (real dims)."""
    obs, task_ids = observations
    fake = copy.deepcopy(teacher)
    for task in (1, 2):
        idx = np.flatnonzero(task_ids == task).tolist()
        g, vdiff = direction_at_own_samples(teacher, fake, TeacherRouter({task: teacher}),
                                            features([obs[i] for i in idx]), np.full(len(idx), task), seed=task)
        assert vdiff == 0.0 and not g.any(), f"task {task}: mean ‖v_fake − v_T‖ = {vdiff}"
    _, vdiff_mixed = direction_at_own_samples(teacher, fake, TeacherRouter({1: teacher, 2: teacher}), cond, task_ids,
                                              seed=3)
    _, _, _, v_fake, v_teacher = noisy_velocities(fake32, teacher, cond, seed=3)
    diff, norm = masked_norm(v_fake - v_teacher, cond["action_mask"]), masked_norm(v_teacher, cond["action_mask"])
    report("test 3", vdiff_mixed_task_bf16=vdiff_mixed, vdiff_fp32_fake_mean=diff.mean(),
           v_teacher_norm_mean=norm.mean(), relative_mean=(diff / norm).mean(), relative_max=(diff / norm).amax())


def test_4_fm_loss_convention_on_real_model(teacher: GR00TVelocity, cond: dict[str, Tensor], z: Tensor) -> None:
    """The flow-matching loss (paper convention, through flow_utils' adapter) of bf16 ckpt-60000 on its own 4-step
    samples, mean of FM_DRAWS (tau, eps) draws, is small: below FM_VS_SFT × its SFT loss at step 60000. With the
    adapter's sign flip removed (native_to_paper_velocity patched to the identity) or its time flip removed it is over
    FLIP_RATIO times larger. Also measured: the loss with the samples' padded dims zeroed (SFT's clean padding)."""
    a = flow_solve(teacher, z.to(DEVICE), cond, NUM_STEPS, requires_grad=False)
    loss = fm_loss(teacher, a, cond, seed=4, draws=FM_DRAWS)
    loss_zero_padding = fm_loss(teacher, torch.where(cond["action_mask"], a, 0.0), cond, seed=4, draws=FM_DRAWS)
    with mock.patch("distillation.flow_utils.native_to_paper_velocity", lambda v: v):
        no_sign_flip = fm_loss(teacher, a, cond, seed=4, draws=FM_DRAWS)
    with mock.patch("distillation.flow_utils.paper_to_native_time", lambda tau: tau):
        no_time_flip = fm_loss(teacher, a, cond, seed=4, draws=FM_DRAWS)
    sft = sft_loss()
    report("test 4", fm_loss=loss, fm_loss_zero_padding=loss_zero_padding, no_sign_flip=no_sign_flip,
           no_time_flip=no_time_flip, sft_loss_last10=sft, sign_ratio=no_sign_flip / loss,
           time_ratio=no_time_flip / loss)
    assert loss < FM_VS_SFT * sft
    assert no_sign_flip > FLIP_RATIO * loss and no_time_flip > FLIP_RATIO * loss


def test_5_padded_dims_change_no_loss_norm_or_direction(teacher: GR00TVelocity, fake32: GR00TVelocity,
                                                        cond: dict[str, Tensor]) -> None:
    """On real observations (fp32 fake vs bf16 teacher), padded dims 12–31 set to 1e6, +inf or nan in v_fake and
    v_teacher leave g bitwise unchanged (0 in padding) and ‖v_fake − v_T‖ unchanged; in the student sample and in g
    they leave L_DMD and its gradient unchanged (0 in padding); in the clean sample (hence the FM target) and the
    predicted velocity they leave L_FM unchanged. GR00T reads padded *inputs*: adding 1 to them changes the real dims
    of its velocity (asserted > 0), so L_FM is checked through PaddingBlind, which keeps the network input fixed.
    Negative controls: changing one real entry changes g and L_FM."""
    mask = cond["action_mask"]
    a, tau, a_tau, v_fake, v_teacher = noisy_velocities(fake32, teacher, cond, seed=5)
    g, vdiff = dmd_direction(v_fake, v_teacher, tau, mask, NORM_EPS), masked_norm(v_fake - v_teacher, mask)
    a_student = a.clone().requires_grad_(True)
    loss_dmd = dmd_surrogate_loss(a_student, g, mask)
    grad = torch.autograd.grad(loss_dmd, a_student)[0]
    loss_fm = fm_loss(PaddingBlind(teacher, mask, 0.0), a, cond, seed=5)
    assert not g[~mask].any() and not grad[~mask].any()
    for fill in PADDING_FILLS:
        v_fake_bad, v_teacher_bad = with_padding(v_fake, mask, fill), with_padding(v_teacher, mask, fill)
        assert torch.equal(dmd_direction(v_fake_bad, v_teacher_bad, tau, mask, NORM_EPS), g)
        assert torch.equal(masked_norm(v_fake_bad - v_teacher_bad, mask), vdiff)
        a_bad = with_padding(a, mask, fill).requires_grad_(True)
        loss_dmd_bad = dmd_surrogate_loss(a_bad, with_padding(g, mask, fill), mask)
        assert torch.equal(loss_dmd_bad, loss_dmd) and torch.equal(torch.autograd.grad(loss_dmd_bad, a_bad)[0], grad)
        assert fm_loss(PaddingBlind(teacher, mask, fill), with_padding(a, mask, fill), cond, seed=5) == loss_fm
    bump = torch.zeros_like(a)
    bump[:, 0, 0] = 1.0
    assert not torch.equal(dmd_direction(v_fake + bump, v_teacher, tau, mask, NORM_EPS), g)
    assert fm_loss(PaddingBlind(teacher, mask, 0.0), a + bump, cond, seed=5) != loss_fm
    with torch.no_grad():
        v_native = teacher.velocity(a_tau, 1 - tau, cond)
        v_padded_input = teacher.velocity(a_tau + torch.where(mask, 0.0, 1.0), 1 - tau, cond)
    input_effect = (v_padded_input - v_native)[mask].abs()
    report("test 5", padded_input_effect_max_abs_real=input_effect.amax(),
           padded_input_effect_mean_abs_real=input_effect.mean(), v_mean_abs_real=v_native[mask].abs().mean())
    assert input_effect.amax() > 0


def test_6_export_round_trip(fake32: GR00TVelocity, reference: tuple, observations: tuple, z: Tensor,
                             tmp_path: Path) -> None:
    """ckpt-60000's head (fp32, as a student) exported by export_student (which runs verify_expert: bf16 only, loads in
    the eval's Gr00tPolicy without weight mismatches, returns a valid chunk) gives, through GR00T's own eval path, the
    same action_pred as ckpt-60000 itself for the same observations and noise, bitwise (bf16 of the fp32 head equals the
    eval's own bf16 cast). The export records its info and refuses to be overwritten; it is deleted afterwards."""
    out = export_student(fake32, tmp_path / "student", {"phase": "C", "test": 6})
    try:
        actions, _ = policy_actions(out, observations[0], z)
        info = json.loads((out / "export_info.json").read_text())
        with pytest.raises(FileExistsError):
            export_student(fake32, out, {})
    finally:
        shutil.rmtree(out)
    report("test 6", export_vs_original_max_abs=(actions - reference[0]).abs().amax())
    assert info["phase"] == "C" and Path(info["base_checkpoint"]) == CKPT60K.resolve()
    assert torch.equal(actions, reference[0])
