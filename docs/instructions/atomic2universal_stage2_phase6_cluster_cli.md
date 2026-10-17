# atomic2universal — Stage 2, Phase 6: Expert Training on the Schmidt Sciences Cluster
## Expert 1: PickPlaceCounterToCabinet · 1 node × 8 GPUs (B200 or B300) · W&B logging

> **To Claude Code:** You are the **Orchestrator**. Phases 1–5 of
> `atomic2universal_stage2_expert_cli.md` are complete on the workstation (`/mnt1/atomic2universal`).
> This file moves training to the Schmidt Sciences cluster, where the project is already cloned
> from GitHub at **`/scratch/schmidt/ssci-ys-shigroup/atomic2universal`** (the cluster project root).
> Read it in full before acting.
> All earlier ground rules still apply: coding standards (≤ 300 lines per file, one responsibility,
> no dead code, fail loudly, `wc -l` before reporting), StructRL is read-only, StructRL's training
> recipe is unchanged, experts are keyed by task name, and the seed partition holds.

---

## 0. Mission

| Item | Value |
|---|---|
| Expert | PickPlaceCounterToCabinet, from `atomic_seen/checkpoint-60000` |
| Algorithm | PPO, sparse terminal reward, StructRL recipe (frozen VLM; action head + value head trained) |
| Hardware | Schmidt Sciences cluster, **1 node × 8 GPUs, B200 or B300, whichever is available** |
| Budget | 100 iterations × 512 episodes; checkpoint every 10 iterations |
| Logging | Weights & Biases: training metrics live, evaluation results attached afterwards |
| Final comparison | Baseline (checkpoint-60000) vs. selected expert, 100 episodes each, seeds 0–99, **same cluster hardware** |
| Target | ≥ 80% success (≈ +17 points over the baseline); ≥ 85% is a strong result |

**Setting:** you run on the cluster's login node in `/scratch/schmidt/ssci-ys-shigroup/atomic2universal`.
The project is a git repository (`github.com/Yuanyuan-Shi/atomic2universal`, private). The scratch
space is a group-shared WekaFS filesystem (`drwxrwx---`, group `ssci-ys-shigroup`) with ample space.

---

## 1. Ground Rules Added for the Cluster

1. **Never run GPU work on the login node.** All GPU work goes through the scheduler.
2. **One job per GPU at a time.** Training uses all 8 GPUs; no evaluation runs on the node while training.
3. **Secrets:** `.env` (git-ignored, mode 600) holds `WANDB_API_KEY` and optionally `WANDB_ENTITY`.
   Never print, log or commit them, and never put them in a job script. If `WANDB_ENTITY` is
   absent, log to the key's default W&B account.
4. **No Hugging Face token is set.** Download from Hugging Face anonymously. If a download fails
   with a rate-limit (429) or access-denied (401/403) error, stop and tell the user rather than
   retrying in a loop.
5. **Storage:** everything (environments, checkpoints, assets, runs) goes under the project root on
   `/scratch/schmidt/ssci-ys-shigroup/`, never the home directory. Code and docs are safe in git;
   checkpoints and results are not, so Phase A must establish the scratch purge policy.
6. **Wall-time limits:** every training job must survive being stopped and resumed (Section 6).
7. **No sudo, no driver installs.** Use the cluster's modules or containers if they are the
   supported way to get CUDA; record what you used.

---

## 2. Agents and Phases

```
Phase A RECON ─► ┌─ Phase B SETUP (env, data, GPU check) ─┐ ─► Phase C PILOT ─► Phase D TRAIN ─► Phase E EVAL ─► Phase F REVIEW ─► report
  (cluster)      └─ Phase B' BUILD (configs, sbatch, W&B) ┘    (2 iterations)   (100 iterations)  (8 GPUs, parallel)
```

| Phase | Agent | Runs alongside | Output |
|---|---|---|---|
| A | **Cluster Recon** | — | `docs/cluster.md` |
| B | **Setup** | B' | environment on the cluster, data downloaded, GPU check passed |
| B' | **Build** | B | node config, job scripts, W&B integration |
| C | **Pilot** | — | full-scale 2-iteration run, W&B verified, final node sizing |
| D | **Train** | — | 100-iteration run with checkpoints |
| E | **Eval** | — | checkpoint selection + final 100-episode comparison |
| F | **Review** | — | independent check of everything above |

Phase B and B' run in parallel: Build needs only Recon's answers, not a finished installation.
Build's dry runs wait until Setup is done. On failure, re-run only the failed agent; if the same
agent fails twice on the same error, stop and report to the user.

---

## 3. Phase A — Cluster Recon (read only, login node)

Write `docs/cluster.md` (≤ 120 lines) answering:
1. **Scheduler:** Slurm or other; how to request 1 node with 8 B200 **or** 8 B300 GPUs (partition,
   constraint or GRES names); how to ask for "either" in one submission, if supported.
2. **Node specs per GPU type:** CPU cores, host RAM, GPU memory, local scratch disk.
3. **Limits:** maximum wall time per job, job-array or dependency support, requeue behaviour,
   per-user GPU limits.
4. **Storage:** the project root is fixed (above). Record: any per-group quota on
   `/scratch/schmidt/ssci-ys-shigroup` (separate from the filesystem's free space), the scratch
   **purge policy** (do untouched files get deleted, after how many days), whether compute nodes see
   the same path, and a permanent location for final experts if scratch is purged.
5. **Network:** whether compute nodes can reach `api.wandb.ai` and `huggingface.co`. If not, W&B
   must run offline and be synced from the login node (Section 5.3).
6. **Software:** NVIDIA driver version on the GPU nodes, available CUDA modules, and whether
   containers (Apptainer/Singularity) are the expected way to run.
7. **GPU architectures:** B200 is compute capability 10.0 (sm_100). B300 (Blackwell Ultra) may
   report 10.3 (sm_103); confirm on an actual node in Phase B. Record both.

**Acceptance:** the Orchestrator can write the exact resource request from `docs/cluster.md`.

---

## 4. Phase B — Setup Agent (cluster)

1. **Code:** the project is already cloned. Run `git pull`, record the commit, and confirm it matches
   the workstation's latest pushed commit. Re-clone StructRL into `third_party/` at the **same
   commit** as on the workstation (from `docs/rl_recon.md`).
2. **Environment:** rebuild the upgraded Blackwell environment from `setup/setup_rl.sh` on the
   shared storage, compiling CUDA extensions for every GPU type we may get:
   `TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3;12.0"` if the CUDA toolkit supports sm_103 (CUDA ≥ 12.9);
   otherwise `"9.0;10.0+PTX;12.0"`, so B300 runs B200 code via the driver's just-in-time compilation.
   Record which applied. Package versions must match the workstation's smoke-test environment;
   any difference is a deviation.
3. **Data:** download directly on the cluster, anonymously (rule 4): checkpoint-60000 with
   `setup/fetch_checkpoint.sh`, and the RoboCasa assets with RoboCasa's asset script, under the
   project root. Confirm `fetch_checkpoint.sh` works with no `HF_TOKEN` set.
4. **GPU check on a real node**, in a short interactive allocation of each GPU type you can get
   (at least the one Phase C will use):
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
   python -c "import torch; x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print((x@x).float().abs().mean().item())"
   ```
   Then run one forward pass of checkpoint-60000 through the RL pipeline's loader, and one
   1-episode evaluation with `baseline_eval/run_baseline.py`, which also confirms rendering works
   on the node.

**Acceptance:** both checks and the 1-episode evaluation pass on the GPU type to be used.

---

## 5. Phase B' — Build Agent (parallel with Setup)

### 5.1 Node configuration
Rename `configs/rl/hw_aws_node.yaml` to `configs/rl/hw_node8.yaml` (the name no longer describes
the hardware) and update every reference. Size it from measured numbers, not guesses:
- **Episodes per iteration: 512** (StructRL). Choose parallel environments × rounds = 512 so that
  `(parallel environments × measured host RAM per environment) + checkpoint-save spike` stays
  below **80%** of the node's host RAM, and parallel environments do not exceed the CPU cores
  available. Use the per-environment RAM and save spike measured in `docs/rl_smoke_test.md`.
- Batch sizes must pass the divisibility check in `build_config.py` for 8 GPUs.
- If B200 and B300 nodes differ enough to need different sizes, use one file per GPU type
  (`hw_node8_b200.yaml`, `hw_node8_b300.yaml`) with identical training settings; only parallel
  environments and rounds may differ, and the product stays 512.

### 5.2 Job scripts (`expert_training/slurm/`)
| File | Limit | Contract |
|---|---|---|
| `train_expert.sbatch` | ≤ 80 | Requests 1 node × 8 GPUs (B200 or B300 per `docs/cluster.md`), CPUs and memory for the node, the maximum allowed wall time. Loads `.env` without echoing it, activates the environment, runs `expert_training/train_expert.sh --task "$TASK" --hw node8 --run-id "$RUN_ID" --resume`. Logs to `expert_training/runs/<Task>/<run_id>/slurm-%j.out`. |
| `submit_training.sh` | ≤ 80 | `--task <Task> [--chain N]`. Creates the run id, submits the first job, and chains up to N follow-up jobs with `afterany` dependencies so the run resumes across wall-time limits. Prints all job ids. |
| `eval_array.sbatch` | ≤ 80 | One GPU per array task; each task runs one `run_baseline.py` evaluation given by a line of a job-list file. Used in Phase E. |

### 5.3 Weights & Biases
- Enable RLinf's built-in W&B logger through our configs (no StructRL edits). If RLinf's logger
  cannot be enabled from configuration alone, stop and report.
- **Run organisation:** project `atomic2universal`; entity from `WANDB_ENTITY` in `.env` if set;
  group `expert/<Task>`; job type `train`; run name = run id; run id stable across resumed jobs
  (resume with W&B's `resume="allow"` and the same id) so one training run is one W&B run.
- **Run config** (logged once): the resolved RLinf config, project and StructRL git commits,
  GPU type and count, PyTorch/CUDA versions, the seed-hook start seed, and the hardware file name.
- **Tags:** task name, GPU type, `sparse-ppo`, `frozen-vlm`.
- **Metrics per iteration.** Log whatever RLinf provides under its own names, and make sure these
  exist (add them in our code if RLinf does not log them):

  | Group | Metrics |
  |---|---|
  | Reward | training-rollout success rate, mean episode return, mean episode length |
  | PPO | policy loss, value loss, approx KL, clip fraction, gradient norm, learning rate |
  | Speed | rollout / train / save seconds, episodes per second |
  | Resources | peak GPU memory per GPU, peak host RAM |
  | Seeds | distinct training seeds this iteration (expect 512), minimum training seed (expect ≥ 100000) |

  The training-rollout success rate includes the exploration noise used during RL, so it reads
  lower than evaluation. Label it `train/rollout_success` so it is never confused with evaluation.
- **Evaluation logging:** write `expert_training/wandb_log_eval.py` (≤ 120 lines). It reads one or
  more `summary.json` files and logs them to the training run (by run id) as
  `eval_select/success` and `eval_select/ci_low|ci_high` against the checkpoint step, and the final
  results as `eval_final/expert_success`, `eval_final/baseline_success` and their CIs. It also
  uploads the console comparison table as a W&B table. Never upload checkpoints or videos.
- **Offline fallback:** if compute nodes cannot reach W&B (Recon answer 5), set `WANDB_MODE=offline`
  in the job and sync from the login node with `wandb sync` after each job, via a short documented
  command in `expert_training/README.md`.

### 5.4 Acceptance
```bash
wc -l expert_training/slurm/* expert_training/wandb_log_eval.py expert_training/*.py expert_training/*.sh
python expert_training/build_config.py --task PickPlaceCounterToCabinet --hw node8 --dry-run
```
The dry run passes the divisibility check and shows 512 episodes per iteration and the W&B settings.

---

## 6. Phase C — Pilot Agent (2 iterations at full scale)

Submit one job with `--chain 0` and the iteration count overridden to 2. Verify:
- both iterations finish; losses finite; `train/rollout_success` around the baseline level;
- the W&B run shows every metric group in 5.3 and the run config;
- **seed check:** 512 distinct training seeds per iteration, all ≥ 100000;
- **resume check:** kill the job during iteration 2, resubmit with `--resume`, and confirm it
  continues from the last checkpoint into the **same** W&B run;
- measured: seconds per iteration (rollout / train / save), peak host RAM, peak GPU memory.

From these, estimate the wall time of 100 iterations and the number of chained jobs needed under
the cluster's wall-time limit. **Report this estimate to the user and wait for approval before Phase D.**

---

## 7. Phase D — Train Agent (100 iterations)

```bash
bash expert_training/slurm/submit_training.sh --task PickPlaceCounterToCabinet --chain <N from Phase C>
```
- Checkpoints every 10 iterations; keep all ten (needed for selection), delete older raw
  intermediates beyond those.
- **Monitor** through W&B and the job logs. Report to the user if: losses become NaN; approx KL
  stays far above its pilot level; `train/rollout_success` falls steadily for 20+ iterations; or
  host RAM exceeds 90%.
- **Fallback if `train/rollout_success` has not risen after ~30 iterations:** report to the user
  before changing anything. Switching to StructRL's shaped reward changes the experimental setup
  and needs the user's approval.

---

## 8. Phase E — Eval Agent (after training ends; 8 GPUs, one evaluation per GPU)

All comparison numbers come from the **same cluster node type and environment**, so hardware
cannot explain a difference.

1. Export the ten saved checkpoints (`step_10` … `step_100`) with `export_expert.py`.
2. **Selection** (array job, one checkpoint per GPU): 20 episodes each on seeds 1000–1019,
   results under `expert_training/results/PickPlaceCounterToCabinet/select/step_<N>/`. Pick the
   step with the highest success rate; on a tie, the earlier step.
3. **Final comparison** (array job, run together):
   - baseline checkpoint-60000: 100 episodes, seeds 0–99 → `baseline_eval/results_n100_cluster/`
   - step-0 control (export of checkpoint-60000 through the RL pipeline): 100 episodes, seeds 0–99
   - selected expert: 100 episodes, seeds 0–99 → `expert_training/results/PickPlaceCounterToCabinet/final/`
4. Log selection and final results to W&B with `wandb_log_eval.py`, and build the HTML reports.
5. Watch 5 failed and 5 successful expert videos; classify failures (no grasp, grasp lost in
   transit, placed outside target, timeout) and check for idling or reward hacking.

**If the step-0 control differs clearly from the baseline**, use step-0 as the baseline for the
comparison and say so in the report.

---

## 9. Phase F — Review Agent (independent check)

A separate agent that did not write the code or run the jobs verifies, and reports pass/fail per item:
- every file written in this phase is within its line limit; no TODO/FIXME markers;
- `third_party/StructRL` matches its recorded commit with no local changes (`git status`, `git diff`);
- no training seed fell in 0–99 or 1000–1019 (from the logged seed metrics);
- checkpoint selection used only seeds 1000–1019; the final test used only seeds 0–99;
- the numbers in the final report match the `summary.json` files and the W&B run;
- `docs/environment.md` lists every deviation from this file, including the GPU type actually used.

---

## 10. Final Report to the User

Before reporting, commit and push the new code, configs and docs (`docs/cluster.md`,
`docs/environment.md`, result `summary.json` files) so the workstation can `git pull` them.
Never commit `.env`, checkpoints, videos or run folders; check `git status` first.

1. GPU type obtained, node specs, environment and arch list used.
2. Final table: baseline, step-0 control and expert at 100 episodes with 95% CIs; the selected step.
3. The W&B run link and the selection curve (success vs. checkpoint step).
4. Failure analysis from the videos.
5. Wall time and GPU-hours used.
6. The Review agent's pass/fail list and all deviations.

**Next expert** (only on the user's request):
`bash expert_training/slurm/submit_training.sh --task PickPlaceDrawerToCounter --chain <N>`

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Job never starts | check the partition/constraint for B200 vs. B300 in `docs/cluster.md`; try the other GPU type |
| `no kernel image is available` on B300 | the build lacks sm_103 and PTX fallback; rebuild with the arch list from Phase B step 2 |
| Ray kills workers for host RAM | lower parallel environments and raise rounds so the product stays 512; re-run the pilot |
| W&B run splits into several runs after a resume | the run id or `resume` setting is not carried into the follow-up job |
| No W&B data from compute nodes | network blocked; use the offline mode and `wandb sync` (Section 5.3) |
| Rendering fails on the node | set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`; check the node's EGL libraries |
| Resume starts from iteration 0 | `--resume` not passed, or the run directory differs between chained jobs |
