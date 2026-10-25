# atomic2universal — Progress

Living progress log. Update it at the end of every stage and remove content that no longer applies.
**Last updated:** 2026-10-25, after the post-3b cleanup (validation-only code removed; MTD takes K teachers).

## Research question

Is **specialize-then-combine** (PPO-fine-tune one expert per pick-and-place task, then distill the
experts into one student with multi-teacher distillation, MTD) better than **combine-then-train**
(fine-tune one policy on all the tasks together)?

- Task family: the 5 pick-and-place tasks among RoboCasa365's 18 fine-tuned atomic tasks
  (`atomic_seen`): PickPlaceCounterToCabinet, PickPlaceDrawerToCounter (in use now),
  PickPlaceCounterToStove, PickPlaceSinkToCounter, PickPlaceToasterToCounter (not added yet).
- Starting policy: GR00T N1.5 joint SFT checkpoint `target_posttraining/atomic_seen/checkpoint-60000`.

## Status

| Stage | What | Status |
|---|---|---|
| 1 / 1c | Eval stack + two-task pick-and-place baseline (`pnp_cabinet_drawer`) | Done |
| 2, Phases 1–5 | RL expert pipeline (StructRL/RLinf PPO, Sparse-RL) + workstation smoke test | Done |
| 2, Phase 6 | Real expert training on one 8-GPU cluster node (Schmidt Sciences, `hw_node8`) | PickPlaceCounterToCabinet (run `20261019-0256-node8`, step_95) and PickPlaceDrawerToCounter (run `20261020-1728-node8`, step_100) trained and exported |
| 2b | 100-episode evaluation, seeds 0–99, no video (workstation): experts vs baseline | Done (step0 runs dropped by user decision) |
| 3a | MTD distillation code (`distillation/`, shared with GR00T), 90 tests, 2-D Gaussian-mixture validation (CPU) | Done (commit `9e3324a`) |
| 3b | MTD on GR00T (`distillation/gr00t/`): integration tests, Levels 1–3 pilot, test of the Level 3 students | Done |
| Cleanup | Removed the 2-D validation and Levels 1–2 (code, configs, runs, students; git tag `pre-cleanup`); configs list K teachers; early stop after 3 rounds without a new best on any task | Done (2026-10-25) |
| later | Other experts (same command, different `--task`); full-scale MTD run | Not started |

## Stage 3b: MTD on GR00T (`distillation/gr00t/RESULTS.md`)

- **Pipeline:** the shared Stage 3a code runs unchanged on GR00T's action head (`GR00TVelocity`, bitwise equal to
  `get_action` per observation); frozen bf16 backbone on the fly (raw-observation buffer, 0.59 MB/obs); rounds of
  collect → train → export → select (`rounds.py`). Batch 16, 2.1 s per DMD + FM update (0.6 s FM only), 70 GB peak.
- **Levels 1–2** (recovery of a damaged ckpt-60000; Expert I1 alone) validated the pipeline and were removed in the
  cleanup (tag `pre-cleanup`). Lessons kept: the same-noise offline distance cannot register distribution matching
  (use `offline_diag.py`'s energy distance), and the student's drift from the teacher grows with lr (hence 3e-6).
- **MTD pilot (both experts, DMD + FM, lr 3e-6, 3 × 500 updates; `configs/distillation/mtd_pnp_cabinet_drawer.yaml`),
  test on seeds 0–99 (100 episodes per task):**

| | cabinet | drawer |
|---|---|---|
| Paper Table 11 (checkpoint SFT-60000) | 63% | 50% |
| 500 / 1,000 / 1,500 updates | 71% / 72% / 79% | 67% / 72% / 63% |
| Experts after single-task RL | 78% | 69% |

  Our ckpt-60000 on these seeds: 67% / 59%. Paired: 1,000 updates drawer +13 (p = 0.019), 1,500 updates cabinet +12
  (p = 0.012, level with Expert I1). Best mean 72% (1,000 updates) ≈ 85% of the experts' mean gain.
- Shared code: `distillation/{interfaces,flow_utils,dmd_loss,router,step}.py`; recommended settings in
  `distillation/README.md` (DMD alone collapses modes; FM alone compounds the 4-step bias; the mix works best).

## Stage 2b: 100-episode evaluation (seeds 0–99, `eval_n100/`)

| Task | Baseline (checkpoint-60000) | Expert | Expert − baseline (Newcombe 95% CI) |
|---|---|---|---|
| PickPlaceCounterToCabinet | 67/100 [57%, 75%] | step_95: 78/100 [69%, 85%] | +11 pp [−1, +23] |
| PickPlaceDrawerToCounter | 59/100 [49%, 68%] | step_100: 69/100 [59%, 77%] | +10 pp [−3, +23] |

- Both experts gain ~10 points, but the unpaired Newcombe CIs (the spec's method) include 0. The runs
  share seeds, so a paired view is sharper: cabinet expert-only 13 vs baseline-only 2 seeds (exact
  McNemar p = 0.007), drawer 20 vs 10 (p = 0.10). Not in `comparison.md`; an option for the write-up.
- The earlier 10-episode numbers were optimistic for the experts (9/10, 8/10) and pessimistic for the
  baseline (6/10, 5/10). Baseline 67% / 59% vs the paper's 63% / 50%.
- The experts were not picked by the planned selection on seeds 1000–1019 (step_95 / step_100 came
  from the cluster runs), so these are not yet "selected checkpoint" final-test numbers.
- Wall time on the workstation: 2 h 53 min of runs for 400 episodes (experts 38 + 45 min, baseline
  41 + 48 min); ~19 s per success and ~38 s per failure (750 steps). Video costs almost nothing extra.
- Files: `eval_n100/comparison.md`, `eval_n100/comparison_table.tex` (`expert_training/compare_models.py`).

## Stage 2: RL experts (superseded 10-episode numbers are in `baseline_eval/results/report.html`)

- **Recipe (StructRL Sparse-RL):** PPO with +1 at first success (`dense_reward` off, `done_shift` on),
  GAE 0.99/0.95, clip 0.2, 1 epoch, AdamW lr 1e-5 (value 2e-5), chunk 16, `flow_sde` noise 0.5.
  Trains the **action head (1.069 B) + a new value head (4.3 M)**; the VLM (1.655 B) stays frozen (all three
  checkpoints share a bitwise-identical backbone). RL changed the action head by ≤ 0.008 per weight.
- Experts: `checkpoints/experts/PickPlaceCounterToCabinet/step_95`, `checkpoints/experts/PickPlaceDrawerToCounter/step_100`
  (bf16, no value head). 10-episode evals: `expert_training/results/<Task>/step_<N>/`.

## Facts to carry forward

- **Evaluation noise is large.** Identical weights (baseline vs step_0) agree on only 5/10 seeds,
  because GR00T samples noise on every query. Use 100 episodes (seeds 0–99) for final comparisons,
  20 (seeds 1000–1019) for checkpoint selection; training scenes are seeded from 100000 upward.
- **Everything runs in bf16.** The Stage 1 eval casts the fp32 SFT action head to bf16, and so does
  RLinf. StructRL keeps no fp32 master weights, so at lr 1e-5 most updates round away (only 5.2% of
  action-head values changed in 3 iterations).
- **Host RAM, not GPU memory, limits RL here.** Each sim process uses 3.5–6 GB of RAM and the actor
  jumps by up to 17 GB at each checkpoint save. The workstation can run 2 envs; the AWS config
  (128 envs × 4 epochs) needs about 0.8–1.2 TiB of host RAM.
- Training uses the eval registry's horizon (750 steps, not StructRL's 500) and the target split.
- StructRL's timing data has no entry for this task, which matters only for the shaped-reward fallback.
- StructRL is CC-BY-NC-4.0 (non-commercial).

## Setup (details in `docs/environment.md`, `docs/rl_recon.md`, `docs/rl_smoke_test.md`)

- Hardware: 1× RTX PRO 6000 Blackwell (96 GB, sm_120), 32 cores, 62 GB RAM, Ubuntu 20.04.
- Eval envs: `a2u-sim` (robosuite `5ce6643`, robocasa `a9e152e`) and `a2u-policy` (GR00T fork `9d7d7a9`).
- RL env: `third_party/conda_envs/a2u-rl-blackwell` (StructRL `6caba1f` pins, torch 2.7.1+cu128,
  flash-attn 2.8.3). The frozen H200 env (`a2u-rl`) is a recipe: `setup/setup_rl.sh --variant frozen`.

## How to run (100-episode runs: add `--no-video`, then `python expert_training/compare_models.py`)

```bash
cd /mnt1/atomic2universal
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw workstation   # cluster: expert_training/slurm/submit_training.sh (hw_node8)
conda run -n a2u-sim python expert_training/export_expert.py --task <Task> --run-dir <run_dir> --step latest
conda run -n a2u-sim python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 \
  --checkpoint checkpoints/experts/<Task>/step_<N> --seed-start 0 --num-episodes 10 \
  --results-root expert_training/results/<Task>/<label> --launch-server
python -m http.server 8080 --directory baseline_eval/results                                # baseline report
python -m http.server 8081 --directory expert_training/results/PickPlaceCounterToCabinet   # /_smoke, /step_0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/anaconda3/envs/a2u-policy/bin/python -m distillation.gr00t.rounds \
  --config configs/distillation/mtd_pnp_cabinet_drawer.yaml                                 # MTD (see distillation/README.md)
```

## Open items

1. The MTD pilot already reaches expert level on one task per student; decide whether to train the experts
   longer before the full-scale run (their gains are small: energy distance 0.11 from ckpt-60000).
2. Add the other three pick-and-place tasks (look up their Table 11 references first).
3. Full-scale MTD: more rounds and episodes, a lower lr (3e-6 or below), and the energy distance plus 100-episode tests
   to choose students; the 1,000 vs 1,500-update difference per task is not settled at 100 episodes. For K tasks: a
   task set with K roles in `configs/pairings/` and one teacher per role in the run config.
