# distillation/ — multi-teacher distillation (MTD)

Distils the per-task pick-and-place experts (Stage 2) into one student with multi-teacher distillation:
a DMD term from the normalized difference of a fake and a teacher velocity, a flow-matching (FM) term on
teacher samples, and IDA (the fake network is pulled toward the student after each update). This answers the
project question: is specialize-then-combine better than combine-then-train?

The shared modules are written for the GR00T action head and run unchanged there. Stage 3a validates them on a
2-D toy problem with exact Gaussian-mixture teachers (`toy/`).

## Conventions

- Method math uses the **paper convention**: τ = 1 is noise, A^τ = (1−τ)·A + τ·ε, velocity target ε − A.
- Every model exposes GR00T's **native convention**: t = 0 is noise, v = action − noise. Here τ = 1 − t and
  v_paper = −v_native. Only `flow_utils.py` converts between the two.
- Actions are `[B, H, D]`. A boolean `action_mask` `[B, H, D]` marks the real dimensions; every loss and norm
  uses only masked-in entries, in at least float32.

## Layout

| Path | Contents |
|---|---|
| `interfaces.py` | `VelocityModel` protocol (native convention), `DistillConfig` (α's, K_ψ, λ_IDA, solver steps) |
| `flow_utils.py` | convention adapter, `noisy_action`, `flow_solve` (GR00T's Euler sampler, cited lines), masked reductions |
| `dmd_loss.py` | `dmd_direction`, `dmd_surrogate_loss`, `flow_matching_loss`, `ida_update` |
| `router.py` | `TeacherRouter`: sends each sample of a mixed-task batch to its own task's teacher |
| `step.py` | `distill_step`: the order of one update (student sample, fake updates, DMD, FM, student step, IDA) |
| `tests/` | pytest validation (spec tests 1–11, with negative controls) |
| `toy/` | 2-D toy: exact GMM teachers, MLP student, metrics, plots, pretraining, training, `RESULTS.md`, notebook |
| `toy/runs/` | toy checkpoints and JSON logs (git-ignored; regenerate with the commands below) |

The shared modules never import `toy/`.

## How to run (CPU is enough; from the project root)

```bash
cd /mnt1/atomic2universal
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY=third_party/conda_envs/a2u-rl-blackwell/bin
$PY/python -m pytest distillation/tests -q                     # validation suite (~2 s)
$PY/python -m distillation.toy.pretrain                         # toy student init -> toy/runs/student_init.pt (~30 s)
$PY/python -m distillation.toy.train_toy --run dmd_fm --alpha-dmd 0.5 --alpha-fm 0.5 --lambda-ida 0.99 --k-fake 1
bash distillation/toy/run_sweep.sh                              # all 10 RESULTS.md runs -> toy/runs/<run>.json (~5 min)
$PY/python -m distillation.toy.report                           # tables for RESULTS.md; overwrites toy/figures/
$PY/jupyter nbconvert --to notebook --execute --inplace distillation/toy/walkthrough.ipynb   # walkthrough (< 5 min)
```

Results, reference levels and the recommended settings for the GR00T runs are in `toy/RESULTS.md`.
