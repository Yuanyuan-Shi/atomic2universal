# Stage 3b: multi-teacher distillation on GR00T, results

The validation runs before the multi-teacher pilot (a 2-D Gaussian-mixture check of the shared code, a recovery
test of a damaged checkpoint-60000 and single-teacher runs) were removed in the cleanup of 2026-10-25; their code,
configs and results are under git tag `pre-cleanup`.

## Phase C: integration tests

2026-10-24, RTX Pro 6000 Blackwell (95.6 GiB), env `a2u-policy`, real checkpoints, round-0 collections. Tests:
`distillation/tests/test_gr00t_integration.py` (1, 3–6; 20 observations recorded by ckpt-60000, 10 per task) and
`test_gr00t_resources.py` (2, 7). `pytest distillation/tests -s` with the GPU: **117 passed** (2 min 21 s); without CUDA:
110 passed, 7 skipped. ckpt-60000's own selection SR (Orchestrator, seeds 1000–1019): i1 12/20, i2 14/20.
After the 2026-10-25 cleanup (K = 3 teacher tests added, the perturbation test removed): **120 passed** with the GPU
(2 min 40 s), 112 passed and 8 skipped without CUDA; test 7 at B = 16 re-measured 2.10 s and 70.6 GiB nvidia-smi peak.

| # | Test | Result | Key numbers |
|---|---|---|---|
| 1 | Sampler match: `flow_solve(GR00TVelocity bf16, z, cond, 4)` vs `get_action` (z injected) | PASS | **B = 1: max abs 0** (bitwise, all 32 dims; noise fp32 in both). **B = 20 batched: max abs 0.067, mean 9.3e-4** (real dims, \|a\| ≤ 1.03); tolerance 0.15 / 2e-3 |
| 2 | Shared backbone | PASS | 585 backbone tensors, `backbone_cfg`, `metadata.json` identical in all 3; a 1-bit flip in one expert tensor and a 1-statistic edit both raise |
| 3 | Identity: student = fake = teacher = bf16 ckpt-60000 | PASS | single-task batches: g ≡ 0, ‖v_ψ − v_T‖ = 0. Measured: mixed-task batch 0.018 (router subsets run at other batch sizes); **precision floor** fp32-master fake vs bf16 teacher: ‖Δv‖ = 0.087 = **0.61 %** of ‖v_T‖ = 14.5 (max 1.2 %) |
| 4 | Convention (FM loss on own 4-step samples) | PASS | **L_FM = 0.054** (0.046 with padded dims zeroed) vs **SFT loss 0.023** (mean of last 10 logs to step 60000); sign flip removed **4.24 (79×)**; time flip removed 2.65 (49×) |
| 5 | Masking (dims 12–31) | PASS | 1e6 / inf / nan in padded v_ψ, v_T, g, A^S, clean samples and FM targets: g, ‖Δv‖, L_DMD (+ grad), L_FM bitwise unchanged; real-entry controls change them. Padded *inputs* +1 move real-dim velocity by ≤ 0.55 (mean 0.029; mean \|v\| 0.83) |
| 6 | Export round trip (ckpt-60000 as student) | PASS | `export_student` + verify_expert OK; eval-path `action_pred` of export vs original, same z, 20 obs: **max abs 0** (bitwise). Closed loop (CLI export 24 s, `run_baseline` i1, seeds 1000–1001, 96 s): loads and runs, **0/2** (both time out at 750 steps); ckpt-60000 itself on these seeds (shared selection): 1/2 (seed 1000 success, 1001 timeout). The server's noise is unseeded, so rollouts differ run to run |
| 7 | Memory and speed (table below) | PASS | largest B with ≥ 15 % headroom in both phases: **16** |

**Test 7.** One full `distill_step` at the two-teacher MTD setting (`Distiller` from `mtd_pnp_cabinet_drawer.yaml`:
student and fake fp32 + AdamW, Expert I1 / I2 bf16 teachers routed on both tasks, shared bf16 backbone; α 0.5 / 0.5, K_ψ 5, λ_IDA 0.99),
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

**Findings for the distillation runs** (no bug found in `distillation/gr00t/*.py`):
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
| ckpt-60000 (student init) | 12/20 = 0.60 [0.39, 0.78] | 14/20 = 0.70 [0.48, 0.86] |
| Expert I1 / Expert I2 (own task) | 14/20 = 0.70 [0.48, 0.86] | 13/20 = 0.65 [0.43, 0.82] |

On these 20 seeds the experts' gains of the 100-episode test (+11 / +10 points, Stage 2b) do not show: +10 and −5
points, both well inside the noise. Selection SR alone cannot resolve a student's move toward an expert here.

ckpt-60000 against the paper (RoboCasa365, arXiv 2603.04356, Table 11, pretrain + target fine-tune): 63% / 50%. Ours:
67% [57, 75] / 59% [49, 68] on the Stage 2b test (seeds 0–99, 100 episodes) and 60% [39, 78] / 70% [48, 86] on the
selection seeds. Both paper values lie inside the 20-episode CIs; the drawer selection 70% is a favorable seed draw
(each seed fixes the scene and objects). Comparisons in this stage are paired on the same seeds, so this offset cancels.

Results: `distillation/runs/selection/<label>/pnp_cabinet_drawer/<role>/summary.json` (Wilson 95% CI).

## MTD pilot: Experts I1 + I2 → ckpt-60000 (`configs/distillation/mtd_pnp_cabinet_drawer.yaml`, run id `level3_mtd_pilot`)

Router: i1 → Expert I1, i2 → Expert I2, task-balanced batches. DMD + FM (α 0.5 / 0.5, K_ψ 5, λ_IDA 0.99), **lr 3e-6**
(user choice, see below), batch 16, 500 updates per round, 3 rounds. Round-0 data: ckpt-60000 and both experts,
20 episodes each per task (2,325 observations, 100 held out); rounds 1–2 add 20 episodes per task of the latest student
(cabinet 16/20, 10/20; drawer 11/20, 12/20). 0.46 updates/s, peak 70.9 GB allocated. Students are named by cumulative
updates; losses are means over that round's updates.

**Why lr 3e-6.** An offline sweep with one teacher (Expert I1 → ckpt-60000 on i1, 300–500 updates per setting) measured
the energy distance of the student's 4-step chunks to Expert I1's on 100 held-out observations (8 noise draws each;
ckpt-60000 itself: 0.112 ± 0.010). The student drifts away from the teacher, more so at higher lr: FM only 0.123 / 0.164
/ 0.424 at lr 1e-6 / 3e-6 / 1e-5; DMD + FM 0.235 at 3e-6 and 0.788 at 1e-5.

**Selection** (seeds 1000–1019, 20 episodes per task):

| row | loss_DMD | loss_FM | ‖v_ψ − v_T‖ | i1 cabinet | i2 drawer |
|---|---|---|---|---|---|
| Paper Table 11 (checkpoint SFT-60000) | – | – | – | 63% | 50% |
| 500 updates | 0.163 | 0.047 | 1.04 | 60% | 70% |
| 1,000 updates | 0.165 | 0.044 | 1.04 | 65% | 75% |
| 1,500 updates | 0.168 | 0.044 | 1.09 | 60% | 80% |
| Experts after single-task RL (Stage 2b test) | – | – | – | 78% | 69% |

On these 20 seeds ckpt-60000 itself scores 60% / 70% and the experts 70% / 65%. No task drops; nothing is outside the
20-episode noise.

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
  success here, as expected of DMD's mode-seeking; distribution distance to the teacher is not a proxy for success.

## Deviations and decisions

- **Buffer stores raw observations** (spec §5 fallback): a cached cond is 3.36 MB/observation (> 2 MB); the backbone
  runs on the fly (CPU transform prefetched on a thread). Raw: 0.59 MB/observation.
- **Batching:** the fork's action head ignores attention masks, so `GR00TVelocity` and `BackboneFeatures` run each
  equal-length group unpadded. Batched backbone features differ ≈ 1% from the eval's B = 1 (bf16 kernels).
- **Round-0 collections are shared** between runs that need the same policy, task and seeds.
- **Parallel rollouts** when `MemAvailable` ≥ 20 GB (Stage 2b actually ran serially, so it gave no rule).
- **Settings fixed during the stage:** batch 16 (Phase C test 7); 500 updates per round (user); lr 3e-6 (user, after
  the sweep above; spec default 1e-5).
- **Offline metric:** the spec's same-noise distance (student and teacher chunk from the same noise z) cannot register
  distribution matching, because the losses do not constrain which action a given z maps to; `offline_diag.py`'s energy
  distance and spreads are reported instead.
- **Additions:** `distillation/gr00t/offline_diag.py` and `train.py --save`; the experts' own selection SR on seeds
  1000–1019 as a reference. After the pilot, all students were evaluated on the same seeds 0–99 as the teachers
  (100 episodes per task).
- **Cleanup of 2026-10-25:** configs list K teachers (`{task, role, label, checkpoint}`, any number of tasks), and a run
  stops after `patience` (3) rounds without a new best on any task. The pilot ran under the earlier rule (2 rounds); with
  `max_rounds` 3 neither rule stops it early, so the config reproduces the pilot.
- The Orchestrator ran the GPU runs itself (long background jobs under the GPU lock) instead of a separate Run agent.
