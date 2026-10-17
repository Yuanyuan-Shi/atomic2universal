# atomic2universal — Progress

Living progress log. Update it at the end of every stage and remove content that no longer applies.
**Last updated:** 2026-10-16, after Stage 2 Phases 1–5 (RL expert pipeline + workstation smoke test).

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
| 2, Phase 6 | Real expert training for PickPlaceCounterToCabinet on one 8-GPU AWS node | **Waiting for the user** (instance choice) |
| later | Other experts (same command, different `--task`); `distillation/` (MTD) | Not started |

## Baseline (Stage 1c, seeds 0–9, 10 episodes, checkpoint-60000)

| Task | Success | 95% CI | Paper (Table 11) | Failures |
|---|---|---|---|---|
| PickPlaceCounterToCabinet | 6/10 | [31%, 83%] | 63% | 3 no grasp, 1 placed outside |
| PickPlaceDrawerToCounter | 5/10 | [24%, 76%] | 50% | 5 no grasp (drawer starts open) |

Grasping thin or awkward objects is the bottleneck for both.

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
  action-head values changed in 3 iterations). Watch for this on AWS if success does not rise.
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

## How to run

```bash
cd /mnt1/atomic2universal
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw workstation   # or --hw aws_node
conda run -n a2u-sim python expert_training/export_expert.py --task <Task> --run-dir <run_dir> --step latest
conda run -n a2u-sim python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 \
  --checkpoint checkpoints/experts/<Task>/step_<N> --seed-start 0 --num-episodes 10 \
  --results-root expert_training/results/<Task>/<label> --launch-server
python -m http.server 8080 --directory baseline_eval/results                                # baseline report
python -m http.server 8081 --directory expert_training/results/PickPlaceCounterToCabinet   # /_smoke, /step_0
```

## Open items

1. **Phase 6 (AWS):** choose the instance, build the matching RL env there, run 100 iterations
   (estimated about 6.5–14 h, most likely ~9 h), 100-episode baseline, checkpoint selection on seeds
   1000–1019, final test on seeds 0–99.
2. Add the other three pick-and-place tasks (look up their Table 11 references first).
3. `distillation/README.md` still mentions "composite tasks"; reword it to the pick-and-place question.
4. The project is not a git repository yet (`.gitignore` is ready).
