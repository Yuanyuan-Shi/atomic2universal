# atomic2universal — Progress

Living progress log. Update it at the end of every stage and remove content that no longer applies.
**Last updated:** 2026-10-23, after Stage 3a (MTD distillation code + 2-D toy validation).

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
| 3a | MTD distillation code (`distillation/`, shared with GR00T), 90 tests, 2-D toy validation (CPU) | Done (commit `9e3324a`) |
| later | Other experts (same command, different `--task`); MTD on the GR00T experts | Not started |

## Stage 3a: distillation toy (`distillation/toy/RESULTS.md`)

- Shared code: `distillation/{interfaces,flow_utils,dmd_loss,router,step}.py`. Paper convention (τ = 1 noise) for the
  method math; only `flow_utils.py` converts from GR00T's native one. `distill_step` fixes the update order for GR00T.
- Toy (exact GMM teachers, one seed): the spec's DMD + FM (α 0.5/0.5, λ_IDA 0.99, K_ψ 1) **fails** the pass bar
  (W2 0.076/0.074 vs. 4-step teacher 0.091/0.093 and initial 0.19/0.16; success 0.79/0.70). With **K_ψ = 5** W2 drops to
  0.046–0.057, about 2× better than the 4-step teacher; no setting clearly passes all readings of the bar.
- DMD only collapses each mode (success 0.97 = collapse, not quality); FM only copies and compounds the 4-step
  teacher's bias (W2 ≈ 0.15); λ_IDA ≤ 0.9 drifts off the modes, λ_IDA = 0 diverges (no NaN — watch `loss_fake`).
- **Recommended for GR00T:** α_DMD = α_FM = 0.5, K_ψ = 5, λ_IDA ∈ {0.99, 1}; log `loss_fake` and `vdiff_norm`.

## Stage 2b: 100-episode evaluation (seeds 0–99, `eval_n100/`)

| Task | Baseline (checkpoint-60000) | Expert | Expert − baseline (Newcombe 95% CI) |
|---|---|---|---|
| PickPlaceCounterToCabinet | 67/100 [57%, 75%] | step_95: 78/100 [69%, 85%] | +11 pp [−1, +23] |
| PickPlaceDrawerToCounter | 59/100 [49%, 68%] | step_100: 69/100 [59%, 77%] | +10 pp [−3, +23] |

- Both experts gain ~10 points, but the unpaired Newcombe CIs (the spec's method) include 0. The runs
  share seeds, so a paired view is sharper: cabinet expert-only 13 vs baseline-only 2 seeds (exact
  McNemar p = 0.007), drawer 20 vs 10 (p = 0.10). Not in `comparison.md`; an option for the write-up.
- The 10-episode numbers below were optimistic for the experts (9/10, 8/10) and pessimistic for the
  baseline (6/10, 5/10). Baseline 67% / 59% vs the paper's 63% / 50%.
- The experts were not picked by the planned selection on seeds 1000–1019 (step_95 / step_100 came
  from the cluster runs), so these are not yet "selected checkpoint" final-test numbers.
- Wall time on the workstation: 2 h 53 min of runs for 400 episodes (experts 38 + 45 min, baseline
  41 + 48 min); ~19 s per success and ~38 s per failure (750 steps). Video costs almost nothing extra.
- Files: `eval_n100/comparison.md`, `eval_n100/comparison_table.tex` (`expert_training/compare_models.py`).

## Baseline (Stage 1c, seeds 0–9, 10 episodes, checkpoint-60000)

| Task | Success | 95% CI | Paper (Table 11) | Failures |
|---|---|---|---|---|
| PickPlaceCounterToCabinet | 6/10 | [31%, 83%] | 63% | 3 no grasp, 1 placed outside |
| PickPlaceDrawerToCounter | 5/10 | [24%, 76%] | 50% | 5 no grasp (drawer starts open) |

Grasping thin or awkward objects is the bottleneck for both.

## RL expert (seeds 0–9, 10 episodes; report section "RL Expert closed-loop evaluation")

| Task | Checkpoint | Success | 95% CI | Baseline | Failures |
|---|---|---|---|---|---|
| PickPlaceCounterToCabinet | `checkpoints/experts/PickPlaceCounterToCabinet/step_95` | 9/10 | [60%, 98%] | 6/10 | seed 3 no grasp (cheese grater) |
| PickPlaceDrawerToCounter | `checkpoints/experts/PickPlaceDrawerToCounter/step_100` | 8/10 | [49%, 94%] | 5/10 | seed 4 no grasp (tongs); seed 6 base drives sideways off the drawer early, no grasp (ladle) |

On the six seeds both solve, cabinet step_95 is faster (mean 248 vs 315 steps). The drawer expert's
seed-6 failure starts with the base driving sideways off the drawer (the baseline stays put). Results:
`expert_training/results/<Task>/step_<N>/` (with videos); the report reads which expert evaluation to
show from `configs/eval/rl_experts.yaml`.

## Stage 2 so far (PickPlaceCounterToCabinet)

- **Recipe (StructRL Sparse-RL):** PPO with +1 at first success (`dense_reward` off, `done_shift` on),
  GAE 0.99/0.95, clip 0.2, 1 epoch, AdamW lr 1e-5 (value 2e-5), chunk 16, `flow_sde` noise 0.5.
  Trains the **action head (1.069 B) + a new value head (4.3 M)**; the VLM (1.655 B) is frozen, the
  same split the SFT used (checked: all 585 VLM tensors of checkpoint-60000 equal NVIDIA's base model).
- **Smoke run** (`expert_training/runs/PickPlaceCounterToCabinet/20261016-1331-workstation`): 3 iterations
  × 8 rollouts (2 envs × 4 epochs), 16/24 rollouts succeeded, losses finite, ~220 s/iteration.
- **10-episode comparison, seeds 0–9:** Stage 1c baseline 6/10, **step_0** control (checkpoint-60000 →
  RLinf load → bf16 → export; bit-identical to the bf16 eval weights) 7/10, **step_3** smoke expert
  7/10. No broken export; step_0 is not clearly different from the baseline, so the baseline stays
  the reference. Not an RL result (3 iterations).
- Exports: `checkpoints/experts/PickPlaceCounterToCabinet/step_{0,3}/` (bf16, no value head).

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
```

## Open items

1. Decide whether the ~10-point expert gains are enough to distill from, or train longer / select
   checkpoints on seeds 1000–1019 first (the gains are not significant with unpaired CIs).
2. Add the other three pick-and-place tasks (look up their Table 11 references first).
3. Before the GR00T MTD run: wrap the GR00T action head as a `VelocityModel` (cached backbone features as `cond`,
   t-bucketing inside the model) and confirm K_ψ = 5 / λ_IDA on a second toy seed if cheap.
