# distillation/ — multi-teacher distillation (MTD)

Distils the per-task pick-and-place experts (Stage 2) into one student with multi-teacher distillation:
a DMD term from the normalized difference of a fake and a teacher velocity, a flow-matching (FM) term on
teacher samples, and IDA (the fake network is pulled toward the student after each update). This answers the
project question: is specialize-then-combine better than combine-then-train?

The shared modules are written for the GR00T action head and run unchanged there; `gr00t/` connects them to
GR00T N1.5 for any number K of teachers, one per task of a task set.

## Conventions

- Method math uses the **paper convention**: τ = 1 is noise, A^τ = (1−τ)·A + τ·ε, velocity target ε − A.
- Every model exposes GR00T's **native convention**: t = 0 is noise, v = action − noise. Here τ = 1 − t and
  v_paper = −v_native. Only `flow_utils.py` converts between the two.
- Actions are `[B, H, D]`. A boolean `action_mask` `[B, H, D]` marks the real dimensions; every loss and norm
  uses only masked-in entries, in at least float32.

## Recommended settings

From the Stage 3a validation on a 2-D Gaussian-mixture problem with exact teachers (removed in the 2026-10-25
cleanup; code and results under git tag `pre-cleanup`):
- **α_DMD = α_FM = 0.5.** DMD alone collapses each mode; FM alone copies and compounds the few-step bias of the
  teacher's sampler. Only the mix beat the 4-step teacher.
- **K_ψ = 5** fake updates per student update; **λ_IDA = 0.99** (λ_IDA ≤ 0.9 was unstable).
- **Log `loss_fake` and `vdiff_norm`**; `loss_dmd` stays near ½·E[(1 − τ)²] = 1/6 by construction (normalized
  direction), so its value carries no progress signal.

## Layout

| Path | Contents |
|---|---|
| `interfaces.py` | `VelocityModel` protocol (native convention), `DistillConfig` (α's, K_ψ, λ_IDA, solver steps) |
| `flow_utils.py` | convention adapter, `noisy_action`, `flow_solve` (GR00T's Euler sampler, cited lines), masked reductions |
| `dmd_loss.py` | `dmd_direction`, `dmd_surrogate_loss`, `flow_matching_loss`, `ida_update` |
| `router.py` | `TeacherRouter`: sends each sample of a mixed-task batch to its own task's teacher (any K) |
| `step.py` | `distill_step`: the order of one update (student sample, fake updates, DMD, FM, student step, IDA) |
| `tests/` | pytest: shared modules on an exact Gaussian-mixture problem (`mixture.py`, CPU), K = 3 teachers (`test_multi_teacher.py`, CPU + one GPU dry run), GR00T data side (CPU) and integration (GPU) |
| `gr00t/` | GR00T N1.5: `wrapper.py` (action head as a `VelocityModel`), `features.py` (shared frozen backbone), `export.py`, `buffer.py` (raw-observation buffer), `collect.py`, `train.py` (`Distiller`), `rounds.py` (collect → train → export → select), `offline_diag.py`, `RESULTS.md` |
| `runs/` | GR00T collections, selection results, run logs and states (git-ignored); students go to `checkpoints/students/` |

## Run configs (`configs/distillation/`)

One YAML per run. `pairing` names the task set (`configs/pairings/<name>.yaml`); `teachers` lists one entry per
task, `{task, role, label, checkpoint}`, where the role must be that task's role in the task set and the router serves
role `i<k>` with task id k; `label` names the teacher's shared collections. `student` gives the initial student's
`label` and `checkpoint`. Rounds stop after `patience` rounds in which no task's selection SR beat its best, or at
`max_rounds`. `mtd_pnp_cabinet_drawer.yaml` holds the settings of the two-teacher pilot (cabinet, drawer).

## How to run (from the project root; GPU, env `a2u-policy`; rollouts in `a2u-sim` through `run_baseline`'s policy server)

```bash
POLICY_PY=~/anaconda3/envs/a2u-policy/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$POLICY_PY -m pytest distillation/tests -q          # CPU tests; GPU tests too when CUDA is visible (else skipped)
$POLICY_PY -m distillation.gr00t.rounds --config configs/distillation/mtd_pnp_cabinet_drawer.yaml
```

A re-run resumes (finished collections, trainings and selections are skipped). Recon notes: `docs/distill_gr00t_notes.md`;
results: `gr00t/RESULTS.md`.
