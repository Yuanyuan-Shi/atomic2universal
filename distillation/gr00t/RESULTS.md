# Stage 3b: multi-teacher distillation on GR00T, results

## Phase C: integration tests

2026-10-24, RTX Pro 6000 Blackwell (95.6 GiB), env `a2u-policy`, real checkpoints, round-0 collections. Tests:
`distillation/tests/test_gr00t_integration.py` (1, 3–6; 20 observations recorded by ckpt-60000, 10 per task) and
`test_gr00t_resources.py` (2, 7). `pytest distillation/tests -s` with the GPU: **117 passed** (2 min 21 s); without CUDA:
110 passed, 7 skipped. ckpt-60000's own selection SR (Orchestrator, seeds 1000–1019): i1 12/20, i2 14/20.

| # | Test | Result | Key numbers |
|---|---|---|---|
| 1 | Sampler match: `flow_solve(GR00TVelocity bf16, z, cond, 4)` vs `get_action` (z injected) | PASS | **B = 1: max abs 0** (bitwise, all 32 dims; noise fp32 in both). **B = 20 batched: max abs 0.067, mean 9.3e-4** (real dims, \|a\| ≤ 1.03); tolerance 0.15 / 2e-3 |
| 2 | Shared backbone | PASS | 585 backbone tensors, `backbone_cfg`, `metadata.json` identical in all 3; a 1-bit flip in one expert tensor and a 1-statistic edit both raise |
| 3 | Identity: student = fake = teacher = bf16 ckpt-60000 | PASS | single-task batches: g ≡ 0, ‖v_ψ − v_T‖ = 0. Measured: mixed-task batch 0.018 (router subsets run at other batch sizes); **precision floor** fp32-master fake vs bf16 teacher: ‖Δv‖ = 0.087 = **0.61 %** of ‖v_T‖ = 14.5 (max 1.2 %) |
| 4 | Convention (FM loss on own 4-step samples) | PASS | **L_FM = 0.054** (0.046 with padded dims zeroed) vs **SFT loss 0.023** (mean of last 10 logs to step 60000); sign flip removed **4.24 (79×)**; time flip removed 2.65 (49×) |
| 5 | Masking (dims 12–31) | PASS | 1e6 / inf / nan in padded v_ψ, v_T, g, A^S, clean samples and FM targets: g, ‖Δv‖, L_DMD (+ grad), L_FM bitwise unchanged; real-entry controls change them. Padded *inputs* +1 move real-dim velocity by ≤ 0.55 (mean 0.029; mean \|v\| 0.83) |
| 6 | Export round trip (ckpt-60000 as student) | PASS | `export_student` + verify_expert OK; eval-path `action_pred` of export vs original, same z, 20 obs: **max abs 0** (bitwise). Closed loop (CLI export 24 s, `run_baseline` i1, seeds 1000–1001, 96 s): loads and runs, **0/2** (both time out at 750 steps); ckpt-60000 itself on these seeds (shared selection): 1/2 (seed 1000 success, 1001 timeout). The server's noise is unseeded, so rollouts differ run to run |
| 7 | Memory and speed (table below) | PASS | largest B with ≥ 15 % headroom in both phases: **16** |

**Test 7.** One full `distill_step` at the Level-3 setting (`Distiller` from `level3_mtd_pilot.yaml`: student and fake fp32
+ AdamW, Expert I1 / I2 bf16 teachers routed on both tasks, shared bf16 backbone; α 0.5 / 0.5, K_ψ 5, λ_IDA 0.99),
balanced batches of the 4 round-0 collections, median of 5 steps after 2 warm-up. GiB. Step = backbone encode +
`distill_step`; the CPU transform runs on the prefetch thread. "Resident" = nvidia-smi used after training at that B
and `empty_cache` (what the trainer holds while rounds.py collects and selects); "+ servers" adds 2 × (6.5 server +
2.0 sim). Limit = 85 % of 95.6 = **81.25 GiB**. Values in parentheses: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

| B | peak allocated | nvidia-smi peak | step (s) | encode (s) | CPU transform (s) | resident (smi) | + servers | fits |
|---|---|---|---|---|---|---|---|---|
| 8 | 55.8 (55.2) | 59.8 (57.7) | 1.75 (1.80) | 0.11 | 0.05 | 47.3 (43.9) | 64.3 (60.9) | yes |
| 16 | 66.3 (65.7) | 74.1 (70.5) | 2.12 (2.11) | 0.17 | 0.10 | 51.3 (44.6) | 68.3 (61.6) | **yes** |
| 32 | 87.0 (86.5) | 91.6 (94.2) | 3.57 (3.36) | 0.46 (0.36) | 0.26 | 69–76 (45.3) | 86–93 (62.3) | no |

No OOM at any B; three default-allocator runs agree within 0.5 GiB and 0.15 s, except B = 32's resident (69 / 76).
**Decision: `train.batch_size: 16`** (7.6 samples/s, 2.1 s per update, ≈ 35 min per 1,000 updates).
Allocated memory after training is 39 GiB at every B; the default allocator keeps up to 35 GiB more reserved
(fragmentation), so **run rounds.py with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (resident 45 GiB, peak 70.5).

**Findings for Phases D–F** (no bug found in `distillation/gr00t/*.py`):
1. *Batched backbone ≠ eval backbone.* A group of B > 1 equal-length observations gives backbone features 1.05 %
   (relative Frobenius) away from the eval's B = 1 features (bf16 kernels; same as GR00T's own batched collate). This is
   the main part of test 1's batched difference (backbone batching alone ≤ 0.080; head batching alone ≤ 0.014; bf16-
   scale noise on B = 1 features ≤ 0.027). Training therefore sees cond at that noise level relative to the eval.
2. *Padded action dims are network inputs* (recon §3): 4-step samples (A^S for the fake's FM loss, A^T for the
   student's) carry free padded entries into A^τ; zeroing them (SFT's clean padding) lowers ckpt-60000's own L_FM on its
   samples 0.054 → 0.046.
3. *DMD signal at the precision floor:* g is normalized, so where ‖v_ψ − v_T‖ falls to the floor (0.61 % of ‖v_T‖)
   g keeps magnitude (1 − τ) and points along bf16/fp32 noise; mixed-task batches add the router's batch-size noise.

Script-driven part of test 6 (scratchpad `phase_c/test6a.sh`):
`python -m distillation.gr00t.export --source <ckpt-60000> --out <scratch>/export_ckpt60k`, then
`run_baseline --pairing pnp_cabinet_drawer --roles i1 --checkpoint <export> --seed-start 1000 --num-episodes 2
--no-video --launch-server --port 5620` (a2u-sim); the 5.4 GB export is deleted afterwards.

## Selection references (seeds 1000–1019, 20 episodes per task)

| Policy | i1 PickPlaceCounterToCabinet | i2 PickPlaceDrawerToCounter |
|---|---|---|
| ckpt-60000 (student init of Levels 2–3) | 12/20 = 0.60 [0.39, 0.78] | 14/20 = 0.70 [0.48, 0.86] |
| Expert I1 / Expert I2 (own task) | 14/20 = 0.70 [0.48, 0.86] | 13/20 = 0.65 [0.43, 0.82] |

On these 20 seeds the experts' gains of the 100-episode test (+11 / +10 points, Stage 2b) do not show: +10 and −5
points, both well inside the noise. Selection SR alone cannot resolve a student's move toward an expert here.

ckpt-60000 against the paper (RoboCasa365, arXiv 2603.04356, Table 11, pretrain + target fine-tune): 63% / 50%. Ours:
67% [57, 75] / 59% [49, 68] on the Stage 2b test (seeds 0–99, 100 episodes) and 60% [39, 78] / 70% [48, 86] on the
selection seeds. Both paper values lie inside the 20-episode CIs; the drawer selection 70% is a favorable seed draw
(each seed fixes the scene and objects). Comparisons in this stage are paired on the same seeds, so this offset cancels.

Results: `distillation/runs/selection/<label>/pnp_cabinet_drawer/<role>/summary.json` (Wilson 95% CI).

## Level 1: identity and recovery (`configs/distillation/level1_recovery.yaml`)

Teacher = bf16 ckpt-60000 on both tasks; student = fp32 ckpt-60000 perturbed by `perturb_` (every action-head tensor
p += s·std(p)·N(0, 1), seed 0); fake = copy of the perturbed student. Data: round-0 collections of ckpt-60000,
20 episodes per task (seeds 200000–200019; 578 + 623 observations, 200 held out). DMD + FM (α 0.5 / 0.5, K_ψ 5,
λ_IDA 0.99), AdamW lr 1e-5, batch 16; offline metric every 100 updates, stop after 3 evaluations without a new best.

**Calibration** (I1, seeds 1000–1009, 10 episodes each; ckpt-60000 6/10 on these seeds):

| scale s | 0.02 | 0.05 | 0.1 | **0.2** | 0.3 | 0.5 |
|---|---|---|---|---|---|---|
| I1 successes | 4/10 | 4/10 | 6/10 | **3/10** | 4/10 | 0/10 |

The smallest scale that halves success is 0.2. With 10 episodes the ladder is noisy (95% CI ≈ ±30 points); only 0.5
breaks the policy outright.

**Training** (means of the 50 updates before each row; offline = same-noise distance on the 200 held-out observations):

| update | loss_DMD | loss_FM | loss_fake | ‖v_ψ − v_T‖ | grad norm | offline distance |
|---|---|---|---|---|---|---|
| 0 | | | | | | 0.810 |
| 100 | 0.169 | 0.057 | 0.051 | 1.73 | 3.13 | 0.930 |
| 200 | 0.163 | 0.060 | 0.045 | 1.64 | 2.69 | 0.906 |
| 300 | 0.165 | 0.042 | 0.042 | 1.54 | 2.63 | 0.964 → stop |

loss_DMD stays at ½·E[(1 − τ)²] = 1/6 by construction: the surrogate's value is ½‖g‖² and the normalized direction has
‖g‖ = 1 − τ (its gradient, not its value, carries the signal). 0.44 updates/s, peak 69 GB allocated.

**Result** (selection, 20 episodes per task):

| | i1 | i2 | offline distance |
|---|---|---|---|
| ckpt-60000 (target) | 0.60 [0.39, 0.78] | 0.70 [0.48, 0.86] | 0 |
| perturbed start (s = 0.2) | 0.30 | 0.40 | 0.810 |
| after 300 updates | **0.70** | **0.65** | 0.964 |

- **Closed-loop recovery: PASS.** Both tasks return inside ckpt-60000's 95% CI.
- **Offline criterion (≥ 80% drop): FAIL.** The same-noise distance rose by 19%, and its plateau stopped training at
  update 300.

**Why the two disagree** (`python -m distillation.gr00t.offline_diag`, 200 held-out observations × 8 shared noise draws;
spread = mean distance between two samples of one observation; energy distance is 0 iff the chunk distributions agree):

| student | same-noise | spread (teacher 1.005) | energy | EEF-pos spread (0.819) | EEF-pos energy | base energy | gripper energy |
|---|---|---|---|---|---|---|---|
| ckpt-60000 fp32 vs bf16 | 0.029 | 1.010 | 0.000 | 0.823 | 0.000 | 0.000 | 0.000 |
| perturbed start | 0.824 | 1.112 | 0.445 ± 0.030 | 0.858 | 0.383 | 0.090 | 0.129 |
| DMD + FM, 300 updates | 0.962 | 0.730 | 0.755 ± 0.068 | **0.563** | 0.695 | **0.031** | 0.101 |
| FM only, 300 updates (ablation) | 0.789 | 0.906 | 0.479 ± 0.042 | 0.730 | 0.439 | 0.034 | **0.052** |

(The bf16 export of the DMD + FM student gives the same numbers as its fp32 state within 0.001.)
- Relative to the teacher, the perturbation hit the base-motion and control-mode outputs hardest (base spread 0.247 vs
  the teacher's 0.072), although end-effector position and rotation carry most of the energy distance. Both trainings repair those components (base energy 0.090 → 0.03), which matches the closed-loop recovery.
- DMD + FM collapses the end-effector position (spread 0.56 vs 0.82) and moves its mean, so the overall energy distance
  *rises* (0.445 → 0.755). The FM-only ablation keeps the spread (0.73) and leaves the energy distance about where it
  started. So the collapse comes from the DMD term, as on the toy (DMD alone collapsed each mode).
- The same-noise distance cannot register a distribution-matching recovery: the losses do not constrain which action a
  given z maps to, and two independent teacher samples are 1.005 apart.

**Ablation details.** FM only (α_DMD 0, α_FM 1), everything else identical (same perturbed start, data, holdout, seeds),
300 updates without early stop, offline metrics only (no closed-loop evaluation): offline distance 0.810 → 0.872 →
0.793 → 0.803; loss_FM 0.043 → 0.042; 1.6 updates/s (no fake updates), peak 36 GB. It ran with an equivalent
20-line `Distiller` driver (git-ignored, `distillation/runs/level1_ablation_fm_only/`); the command in
`configs/distillation/level1_ablation_fm_only.yaml` (`train.py --save`) reproduces it (it prints the rows instead of
writing `train_log.jsonl`, and also saves fake and optimizer states, ≈ 22 GB).

Diagnostic command (Level 1 table; collections in the run's order, defaults otherwise):
`python -m distillation.gr00t.offline_diag --collections distillation/runs/collections/ckpt60k/{i1,i2}/round_0
--teachers i1=<ckpt-60000> i2=<ckpt-60000> --students ckpt60k_fp32=<ckpt-60000> damaged=perturb:0.2
dmd_fm_300=distillation/runs/level1_recovery/state/round_0/student.pt
dmd_fm_300_export=checkpoints/students/level1_recovery/round_0
fm_only_300=distillation/runs/level1_ablation_fm_only/state/student.pt`.

## Level 2: single teacher (`configs/distillation/level2_single.yaml`)

Expert I1 → ckpt-60000 on i1 only; variants differ only in the loss weights. Round 0 data: ckpt-60000 and Expert I1,
20 episodes each (578 + 478 observations, 100 held out); rounds 1–2 add 20 episodes of the previous round's student
(seeds 201000+, 202000+). 500 updates per round (user choice after Level 1), lr 1e-5, batch 16.

| variant | updates (round) | loss_DMD | loss_FM | loss_fake | ‖v_ψ − v_T‖ | grad norm | offline distance | i1 selection |
|---|---|---|---|---|---|---|---|---|
| init (ckpt-60000) | 0 | | | | | | 0.409 | 0.60 |
| DMD + FM | 500 (r0) | 0.163 | 0.047 | 0.033 | 1.41 | 2.87 | 0.904 | 0.55 |
| DMD + FM | 1,000 (r1) | 0.165 | 0.045 | 0.031 | 1.42 | 2.33 | 0.896 | **0.65** |
| DMD + FM | 1,500 (r2) | 0.168 | 0.043 | 0.031 | 1.41 | 2.03 | 0.930 | 0.60 |
| FM only | 500 (r0) | – | 0.038 | – | – | 0.38 | 0.735 | **0.65** |
| FM only | 1,000 (r1) | – | 0.035 | – | – | 0.33 | 0.790 | 0.65 |
| FM only | 1,500 (r2) | – | 0.034 | – | – | 0.32 | 0.827 | 0.65 |

Students are named by cumulative updates; loss columns are means over that round's updates. Student collections (by the
500- and 1,000-update students): DMD + FM 12/20, 12/20; FM only 13/20, 12/20. Speed: DMD + FM 0.45 updates/s (69 GB peak), FM only 1.7 updates/s (36 GB).

- **Selection:** both variants stay at 0.55–0.65 on i1 (Expert I1: 0.70 on the same seeds); all 3 rounds ran (FM only
  met the stop condition only after round 2, together with `max_rounds`). The spec's expected move "from ~67% toward
  ~78%" is not visible at 20 episodes.
- **Forgetting check (best student on i2, 20 episodes):** DMD + FM at 1,000 updates: 0.75; FM only at 500 updates: 0.75;
  ckpt-60000: 0.70.
  No forgetting.
- **Offline diagnostic vs Expert I1** (100 held-out i1 observations × 8 noise draws; Expert I1's spread 0.941; command:
  `offline_diag --collections distillation/runs/collections/{ckpt60k,expert_i1}/i1/round_0 --holdout 100
  --teachers i1=<Expert I1> --students ckpt60k=<ckpt-60000> <variant>_round_<r>=checkpoints/students/level2_single_<variant>/round_<r> ...`):

| student | same-noise | spread | energy distance to Expert I1 |
|---|---|---|---|
| ckpt-60000 (init) | 0.366 | 0.896 | **0.112 ± 0.010** |
| DMD + FM, 500 / 1,000 / 1,500 updates | 0.957 / 0.939 / 0.916 | 0.561 / 0.504 / 0.515 | 0.788 / 0.742 / 0.683 |
| FM only, 500 / 1,000 / 1,500 updates | 0.718 / 0.718 / 0.764 | 0.791 / 0.801 / 0.705 | 0.424 / 0.388 / 0.479 |

  The expert is close to its init (energy 0.112: RL changed the action head by ≤ 0.008 per weight). Both variants move
  the student's action distribution *away* from the teacher: DMD + FM collapses the spread to ~55% of the expert's
  (6–7× the init's energy distance), FM only drifts less (~4×). The training's own distortion is far larger than the
  signal it should transfer.

## Tuning sweep after Level 2 (user-approved; offline only)

Level 2 data and teacher (Expert I1 → ckpt-60000, i1), 300 updates per setting from ckpt-60000, batch 16, same 100
held-out observations; driver `distillation/runs/tuning/run_tuning.sh` (git-ignored): `train.py --steps 300 --holdout 100 --save` on
copies of `level2_single.yaml` without variants that change only α and lr (patience 0), then `offline_diag`. Losses are means over the 300 updates.

| setting | α_DMD / α_FM | lr | loss_DMD | loss_FM | ‖v_ψ − v_T‖ | same-noise (0 → 100 → 200 → 300) | mean gap | spread | energy to Expert I1 |
|---|---|---|---|---|---|---|---|---|---|
| ckpt-60000 (init) | – | – | | | | 0.409 | 0.317 | 0.896 | **0.112 ± 0.010** |
| FM only | 0 / 1 | 1e-6 | – | 0.039 | – | 0.409 → 0.365 → 0.336 → 0.373 | **0.279** | 0.759 | 0.123 ± 0.035 |
| FM only | 0 / 1 | 3e-6 | – | 0.038 | – | 0.409 → 0.410 → 0.457 → 0.464 | 0.339 | 0.719 | 0.164 ± 0.028 |
| DMD + FM | 0.5 / 0.5 | 3e-6 | 0.162 | 0.043 | 0.98 | 0.409 → 0.471 → 0.544 → 0.554 | 0.410 | **0.816** | 0.235 ± 0.051 |
| DMD + FM | 0.1 / 0.9 | 1e-5 | 0.162 | 0.043 | 1.42 | 0.409 → 0.713 → 0.722 → 0.672 | 0.550 | 0.747 | 0.353 ± 0.051 |
| *Level 2 FM only, 500 updates* | 0 / 1 | 1e-5 | – | 0.038 | – | 0.409 → … → 0.735 (500) | 0.622 | 0.791 | 0.424 ± 0.051 |
| *Level 2 DMD + FM, 500 updates* | 0.5 / 0.5 | 1e-5 | 0.163 | 0.047 | 1.41 | 0.409 → … → 0.904 (500) | 0.821 | 0.561 | 0.788 ± 0.116 |

(Same-noise columns are `train.py`'s single-draw offline metric; mean gap, spread and energy are `offline_diag` with
8 draws. Expert I1's spread is 0.941.)
- The distortion scales with the learning rate; at lr 1e-5 even α_DMD = 0.1 drifts almost as far as FM only.
- No setting gets closer to Expert I1 than ckpt-60000 itself (energy 0.112) in 300 updates. FM only at lr 1e-6 is
  neutral within the error (its mean gap 0.279 vs 0.317 is within the estimator's noise; its spread narrows).
- FM narrows the 4-step spread at every lr (0.90 → 0.72–0.79): it fits the field that best explains the teacher's
  4-step samples, whose own 4-step samples are less spread (the toy's "FM compounds the 4-step bias").
- At low lr, DMD keeps the spread better (0.82) but moves the mean further.
- Level 3 uses DMD + FM at lr 3e-6 (user choice); its i1 curve is therefore not directly comparable with Level 2's.

## Level 3: multi-teacher pilot (`configs/distillation/level3_mtd_pilot.yaml`)

Experts I1 + I2 → ckpt-60000 on both tasks (router: i1 → Expert I1, i2 → Expert I2), task-balanced batches. DMD + FM
(α 0.5 / 0.5, K_ψ 5, λ_IDA 0.99), **lr 3e-6** (user choice after the tuning sweep), batch 16, 500 updates per round,
3 rounds. Round-0 data: ckpt-60000 and both experts, 20 episodes each per task (2,325 observations, 100 held out);
rounds 1–2 add 20 episodes per task of the latest student (cabinet 16/20, 10/20; drawer 11/20, 12/20). 0.46 updates/s,
peak 70.9 GB allocated. Students are named by cumulative updates; losses are means over that round's updates.

**Selection** (seeds 1000–1019, 20 episodes per task):

| row | loss_DMD | loss_FM | ‖v_ψ − v_T‖ | offline distance | i1 cabinet | i2 drawer |
|---|---|---|---|---|---|---|
| Paper Table 11 (checkpoint SFT-60000) | – | – | – | – | 63% | 50% |
| 500 updates | 0.163 | 0.047 | 1.04 | 0.446 | 60% | 70% |
| 1,000 updates | 0.165 | 0.044 | 1.04 | 0.506 | 65% | 75% |
| 1,500 updates | 0.168 | 0.044 | 1.09 | 0.527 | 60% | 80% |
| Experts after single-task RL (Stage 2b test) | – | – | – | 0 | 78% | 69% |

On these 20 seeds ckpt-60000 itself scores 60% / 70% and the experts 70% / 65%. The offline (same-noise) distance to the
experts rises from 0.318 (ckpt-60000) as in the other levels. No task drops; nothing is outside the 20-episode noise.
The I1 curve is not directly comparable with Level 2's (lr 3e-6 here vs 1e-5).

**Test on seeds 0–99** (100 episodes per task, the seeds of the teachers' Stage 2b numbers; Wilson 95% CI):

| row | loss_DMD | loss_FM | ‖v_ψ − v_T‖ | i1 cabinet | i2 drawer | mean |
|---|---|---|---|---|---|---|
| Paper Table 11 (checkpoint SFT-60000) | – | – | – | 63% | 50% | 56.5% |
| 500 updates | 0.163 | 0.047 | 1.04 | 71% [62, 79] | 67% [57, 75] | 69% |
| 1,000 updates | 0.165 | 0.044 | 1.04 | 72% [62, 80] | **72%** [62, 80] | **72%** |
| 1,500 updates | 0.168 | 0.044 | 1.09 | **79%** [70, 86] | 63% [53, 72] | 71% |
| Experts after single-task RL | – | – | – | 78% [69, 85] | 69% [59, 77] | 73.5% |

Our ckpt-60000 on the same seeds (Stage 2b): 67% [57, 75] / 59% [49, 68], mean 63%. Paired per seed (exact McNemar;
student-only vs reference-only successes):

| student | i1 vs ckpt-60000 | i2 vs ckpt-60000 | i1 vs Expert I1 | i2 vs Expert I2 |
|---|---|---|---|---|
| 500 updates | 11 vs 7, p = 0.48 | 16 vs 8, p = 0.15 | 4 vs 11, p = 0.12 | 10 vs 12, p = 0.83 |
| 1,000 updates | 13 vs 8, p = 0.38 | **20 vs 7, p = 0.019** | 3 vs 9, p = 0.15 | 15 vs 12, p = 0.70 |
| 1,500 updates | **16 vs 4, p = 0.012** | 18 vs 14, p = 0.60 | 6 vs 5, p = 1.00 | 9 vs 15, p = 0.31 |

- Every student is above ckpt-60000 on both tasks; the 1,000-update student is significantly better on drawer (+13)
  and the 1,500-update student on cabinet (+12, level with Expert I1). No student is significantly worse than an
  expert. The best mean (1,000 updates, 72%) recovers ≈ 85% of the experts' mean gain over ckpt-60000 (63% → 73.5%).
- Between 1,000 and 1,500 updates cabinet rises (72 → 79%) while drawer falls (72 → 63%): with 100 episodes the
  per-task numbers still move by up to ±9 points, so the trend per task is not settled.
- **Offline vs closed loop:** relative to the experts (100 held-out observations, 50 per task), the students' energy
  distance rises from ckpt-60000's 0.105 ± 0.015 to 0.195 / 0.224 / 0.215 and their spread shrinks (0.94 → 0.84 /
  0.79 / 0.77; experts 0.96), yet closed-loop success rises. A narrower (more mode-seeking) action distribution helps
  success here, as DMD did on the toy; distribution distance to the teacher is not a proxy for success.

## Deviations and decisions

- **Buffer stores raw observations** (spec §5 fallback): a cached cond is 3.36 MB/observation (> 2 MB); the backbone
  runs on the fly (CPU transform prefetched on a thread). Raw: 0.59 MB/observation.
- **Batching:** the fork's action head ignores attention masks, so `GR00TVelocity` and `BackboneFeatures` run each
  equal-length group unpadded. Batched backbone features differ ≈ 1% from the eval's B = 1 (bf16 kernels).
- **Round-0 collections are shared** between runs that need the same policy, task and seeds.
- **Parallel rollouts** when `MemAvailable` ≥ 20 GB (Stage 2b actually ran serially, so it gave no rule).
- **Settings fixed during the stage:** batch 16 (Phase C test 7); Level 1 scale 0.2 (calibration); 500 updates per round
  for Levels 2–3 (user); Level 3 lr 3e-6 (user, after the tuning sweep; spec default lr 1e-5 kept for Levels 1–2).
- **Additions:** Level 1 FM-only ablation and the Level 2 tuning sweep (user-approved, offline only);
  `distillation/gr00t/offline_diag.py` (energy distance and spreads) and `train.py --save`; the experts' own selection
  SR on seeds 1000–1019 as a reference. After the pilot, all Level 3 students were evaluated on the same seeds 0–99 as
  the teachers (100 episodes per task).
- **Level 1 offline criterion:** the spec's same-noise distance cannot register a distribution-matching recovery; the
  energy distance is reported next to it.
- The Orchestrator ran Phases D–F itself (long background jobs under the GPU lock) instead of a separate Run agent.
