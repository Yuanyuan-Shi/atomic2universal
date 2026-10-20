# Schmidt Sciences cluster (Stage 2, Phase 6, Phase A recon)
Read-only recon from login03, 2026-10-17 (`sinfo`, `scontrol`, `sacctmgr`, `/apps/helpers/{quotas.py,sbalance}`,
probe jobs). Cluster `skipjack` (JHU ARCH, Schmidt Sciences tenant; help: helpdesk@schmidtsciences.jhu.edu,
portal https://schmidtsciences.arch.jhu.edu). User `ssci-ys-shigroup`, account `ssci-ys-shigroup`
(parent `pi-ssci-ys-shigroup`), QoS `scavenger` (default) or `ssci`; group members also: ssci-haohan-shigroup,
ssci-jf-shigroup, ssci-r7yuan.

## 1. Scheduler and the resource request
Slurm (`sched/backfill`, `select/cons_tres`, `CR_CORE_MEMORY`, multifactor priority: fair share 20000, age 5000,
job size 5000). The account may use only partitions **`b200`**, **`b300`** (AllowAccounts=schmidt) and the CPU
partitions `interactive_cpu` (4 CPUs, 8 GB, 4 h, 1 job) and `agentic` (4 CPUs, 8 GB, 7 days, 1 job); `med`, `a100`,
`h100`, `h200`, `l40s`, `rtx6000` are JHU-only. GRES names: `gpu:b200` (gb201-216, 16 nodes), `gpu:b300`
(gb301-317, 17 nodes). **"Either type" in one submission:** `--partition=b200,b300 --gres=gpu:8` (generic GRES,
the job starts in whichever partition can run it first; `sbatch --test-only` accepted it). Exact request
(`expert_training/slurm/train_expert.sbatch`):
```
#SBATCH --partition=b200,b300 --nodes=1 --gres=gpu:8 --exclusive --cpus-per-task=124 --mem=1488000M
#SBATCH --time=3-00:00:00 --signal=B:TERM@300
```
`--exclusive` keeps other users' jobs (and their Ray clusters) off the node (partitions are `OverSubscribe=YES:4`).
Backfill estimate at 02:00 on 2026-10-17 for a whole node: B200 05:43 the same day, B300 the next day 12:42; one
B300 GPU was free immediately. Waiting for a whole node takes hours.

## 2. Node specs (scontrol; per node)
| | B200 (gb201-216) | B300 (gb301-317) |
|---|---|---|
| CPU | 2 x 64 cores, 1 thread/core; 124 usable (`MaxCPUsPerNode=124`) | same |
| Host RAM | 1,547,000 MB (1.47 TiB) | 3,096,000 MB (2.95 TiB) |
| Per job | `MaxMemPerCPU=12000` MB -> at most 124 x 12000 MB = 1,488,000 MB (1419 GiB) on both | same |
| GPUs | 8 x B200 (GPU memory: see 7) | 8 x B300 (see 7) |
| Local scratch | `TmpDisk=0` (none configured; use the shared scratch) | same |
| Defaults | `DefCpuPerGPU=15`, `DefMemPerCPU=12000` | same |
Because a job gets at most 1419 GiB on either type, one hardware file `configs/rl/hw_node8.yaml` serves both.

## 3. Limits
- **Wall time:** `MaxTime=3-00:00:00` on b200/b300 (b300 default 12 h; b200 no default, so always pass `--time`).
- **Budget (binding):** account `GrpTRESMins billing=10,800,000` = **180,000 SU per month** (`sbalance`; 1 SU =
  1 billing-hour), reset monthly (`PriorityUsageResetPeriod=MONTHLY`, next 2026-11-01), shared by the group; usage
  decays with a 30-day half-life inside the month. Billing is the max over TRES (`PriorityFlags=MAX_TRES`): one
  B200 GPU 3,259 SU/h, one B300 GPU 3,697 SU/h, a whole node 26,945 SU/h (B200) or 31,547 SU/h (B300). On
  2026-10-17 02:00: 44,986 SU used, **139,017 SU available = 5.2 B200 node-hours or 4.4 B300 node-hours**.
  `AccountingStorageEnforce=associations,limits,qos` (no `safe`): jobs start even if they would cross the limit,
  and per Slurm's GrpTRESMins semantics, reaching it kills the account's running jobs (all group members) and
  holds new ones until the reset.
- No per-user GPU or job limits in QoS `ssci`/`scavenger`; no account `GrpTRES` limit.
- **Threads:** `ulimit -u` soft 4096 (login, propagated into jobs by `PropagateResourceLimits=ALL`); GPU-node hard
  limit 8,254,777. Ray needs far more than 4096: `train_expert.sh` raises the soft limit (a 26 h hang otherwise).
- Job arrays (`MaxArraySize=10000`) and dependencies (`afterany`, `afterok`) are supported.
- Requeue: `JobRequeue=1` (batch jobs are requeued after a node failure), `MaxBatchRequeue=5`. Preemption is
  `partition_prio` with `PreemptMode=REQUEUE`, but b200/b300 have the same priority tier (1), so jobs there are
  not preempted. `KillWait=30 s`: hence `--signal=B:TERM@300` for a clean stop before the time limit.

## 4. Storage
- Project root `/scratch/schmidt/ssci-ys-shigroup/atomic2universal` = WekaFS `wekafs1/scratchssci` mounted at
  `/weka/scratch/schmidt` (8.0 PB, 7% used). Group quota **55 TB** on the group's scratch (18.28 TB used, 33%);
  home quota 100 GB (42 GB used) on `wekafs1/homessci`.
- **Purge policy: none published** on the system (no motd, no helper script or banner mentions one). Treat scratch
  as not backed up; ask the helpdesk before relying on it for months. `/weka/oldscratch` exists (an earlier scratch).
- Compute nodes see the same path (checked from csr048 and the GPU nodes in Phase B).
- Permanent location for final experts (5.1 GB each): no archive filesystem is mounted. Copy each final expert to
  the workstation (`/mnt1/atomic2universal/checkpoints/experts/`) and, if wanted, the home directory (58 GB free);
  code, docs and `summary.json` files are in git.

## 5. Network
Compute nodes reach the internet directly (HTTP 200 from the CPU node csr048 and from GPU nodes, Phase B):
`api.wandb.ai`, `huggingface.co`, `pypi.org`, `github.com`, `download.pytorch.org`, `conda.anaconda.org`.
**W&B runs online from the job**; the offline fallback is documented in `expert_training/README.md`.

## 6. Software
- RHEL 9.6 (kernel 5.14.0-570, glibc 2.34). Lmod modules: `cuda/11.8.0, 12.3.0, 12.6.3, 12.8.1, 13.0.2`,
  `python/3.11.9`, `mesa/23.3.6`, gcc 8.5-13.2. No conda module and no `apptainer`/`singularity` binary
  (only `/apps/helpers/apptainer.sh`, a bind helper), so containers are not the expected route.
- Used: Miniforge 25.3.1 installed in `third_party/miniforge3` (user-level, no sudo); envs as on the workstation
  (`setup/setup_{sim,policy,rl}.sh`); CUDA comes from the pip wheels (torch 2.7.1+cu128); no module needed.
- NVIDIA driver on the GPU nodes: 595.71.05 (CUDA 13.x driver; runs the cu128 wheels).

## 7. GPU architectures (confirmed on nodes in Phase B)
- B300 (gb301, job 1117071): `NVIDIA B300 SXM6 AC`, **compute capability 10.3 (sm_103)**, 275,040 MiB per GPU,
  driver **595.71.05**. torch 2.7.1+cu128 has no sm_103 build (arch list sm_75-sm_120 + compute_120); its sm_100
  kernels run there (same major version: cubin compatibility), as do flash-attn's sm_100 kernels (checked).
- B200 (gb205, job 1117150): `NVIDIA B200`, **compute capability 10.0 (sm_100)**, 183,359 MiB per GPU, driver 595.71.05.
- Nodes: Rocky Linux 9.6, local `/` 445 GB (395 GB free; Ray's temp dir `/tmp`), `/dev/shm` 1.5 TB (Ray object store).
