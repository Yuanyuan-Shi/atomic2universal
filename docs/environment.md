# Environment

Built by the Setup Agent (Phase 2a). Official eval path and I/O contract: `docs/eval_entrypoint.md`.

## Hardware and OS
- Ubuntu 20.04 (glibc 2.31), Linux 5.15, 32 CPU cores, 62 GB RAM, no sudo.
- 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, compute capability sm_120,
  driver 575.64.05 (CUDA 12.9). Headless rendering through EGL (`libEGL_nvidia`).
- conda 4.12.0 at `/home/james/anaconda3` (envs live there); repos and assets on `/mnt1`.

## Decision: two envs (`a2u-sim` + `a2u-policy`) connected by the official GR00T ZMQ server/client
Rule (guide §4.1): one env only if it resolves cleanly and both stacks import. It does not
resolve: RoboCasa 1.0.1 (`third_party/robocasa/setup.py`) requires `lerobot==0.3.3` and
`tianshou==0.4.10`; lerobot 0.3.3 requires `gymnasium<1.0` and `av>=14.2`, tianshou 0.4.10
requires `protobuf~=3.19`, while the GR00T N1.5 fork (`third_party/Isaac-GR00T/pyproject.toml`)
pins `gymnasium==1.0.0`, `av==12.3.0`, `tianshou==0.5.1`, `protobuf==3.20.3`. pip cannot satisfy
both sets, so per the rule we use two envs and the fork's own server/client: the policy env runs
`scripts/inference_service.py --server` (same `Gr00tPolicy` + `panda_omron` transforms as the
official `run_eval.py`), and the sim env talks to it with `gr00t.eval.service.ExternalRobotInferenceClient`
over ZMQ (torch.save payloads; numpy 2.2.5 in both envs). The official `run_eval.py` does the same
over ZMQ, only with server and client as two threads of one process.

| Env | Python | Contents |
|---|---|---|
| `a2u-sim` | 3.11.17 (conda-forge) | robosuite, RoboCasa365, kitchen assets, ffmpeg 9.0.2 (`ffprobe`), `imageio[ffmpeg]`, pyyaml, pyzmq, torch (CPU use only), gr00t fork installed `--no-deps` (only for the ZMQ client) |
| `a2u-policy` | 3.11.17 (conda-forge) | Isaac-GR00T RoboCasa fork (GR00T N1.5), torch cu128, flash-attn (sm_120 build) |

Setup scripts: `setup/setup_sim.sh`, `setup/setup_policy.sh` (idempotent, no arguments, `-h` for
help). Lock files: `setup/requirements.sim.lock.txt`, `setup/requirements.policy.lock.txt`
(`pip freeze`, editable repos listed with their commits at the end).

## Repositories (`third_party/`, checked out detached at these commits)
| Repo | URL | Commit |
|---|---|---|
| robosuite (master, v1.5.2) | github.com/ARISE-Initiative/robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| robocasa (RoboCasa365 v1.0.1, main) | github.com/robocasa/robocasa | `a9e152e2eea16d342dac40e4577fb4c68b932854` |
| Isaac-GR00T, RoboCasa365 fork (GR00T N1.5) | github.com/robocasa-benchmark/Isaac-GR00T | `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |

The GR00T fork is the one RoboCasa365 names for its GR00T benchmark
(`third_party/robocasa/docs/benchmarking/policy_learning_algorithms.md`); its single commit
"GR00T N1.5 for RoboCasa" sits on NVIDIA upstream `4ea96a16` (July 2025), an ancestor of upstream
tag `n1.5-release`. Upstream `main` is now N1.7 and is not used. Checkpoint metadata agrees:
`config.json` model_type `gr00t_n1_5`, transformers 4.51.3, embodiment `new_embodiment`,
modalities matching the fork's `panda_omron` data config.

## Key package versions
| Package | a2u-sim | a2u-policy |
|---|---|---|
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 | 2.7.1+cu128 / 0.22.1+cu128 |
| CUDA (torch build) | 12.8 | 12.8 (driver 12.9) |
| flash-attn / attention impl | — | 2.8.3 (source build, sm_120) / flash_attention_2 as configured |
| transformers | — | 4.51.3 |
| numpy | 2.2.5 (asserted by robocasa) | 2.2.5 |
| mujoco | 3.3.1 (asserted by robocasa) | — |
| robosuite / robocasa | 1.5.2 / 1.0.1 | — |
| gymnasium | 0.29.1 | 1.0.0 |
| diffusers / peft / timm | — | 0.30.2 / 0.17.0 / 1.0.14 |
| lerobot / tianshou | 0.3.3 / 0.4.10 | — / 0.5.1 |

## Environment variables
- `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`: stored in `a2u-sim` with `conda env config vars`, so
  `conda activate a2u-sim` and `conda run -n a2u-sim` set them automatically.
- `CUDA_VISIBLE_DEVICES=0` (single GPU; `.env.example`). `HF_TOKEN` is not needed at runtime: the
  checkpoint is local and the Eagle backbone config ships inside the fork.

## Assets
`python -m robocasa.scripts.download_kitchen_assets` for all six types (`tex`, `tex_generative`,
`fixtures_lw`, `objs_objaverse`, `objs_aigen`, `objs_lw`) into
`third_party/robocasa/robocasa/models/assets/` (23 GB on disk). Objaverse objects are included, so
pick-and-place scenes load. Marker files `third_party/.robocasa_assets_<type>.done` record finished
types. No training datasets were downloaded. `third_party/cuda-12.8/` holds the CUDA 12.8 compiler
and headers used only to build flash-attn.

## How to activate and run
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
bash setup/setup_sim.sh && bash setup/setup_policy.sh        # build (or verify) both envs
conda activate a2u-sim                                        # all project code runs here
python -m baseline_eval.run_baseline --pairing pairing_kettleboiling --roles i1 --port 5555 --launch-server
```
`--launch-server` runs `evaluation/serve_policy.sh --port N` (the fork's `inference_service.py --server`
in `a2u-policy`, checkpoint from the eval config) in its own process group, waits for "Server is ready
and listening" (log: `baseline_eval/results/<pairing>/policy_server_port<N>.log`), and kills the group
on exit, Ctrl-C, SIGTERM or error. Without it, start `bash evaluation/serve_policy.sh --port N` in another
terminal first. Requests time out after 60 s, so a dead server fails loudly instead of hanging.
Sanity checks: `conda run -n a2u-sim python -c "import robosuite, robocasa, mujoco"`;
`docs/sanity_frame.png` (TurnOnMicrowave, target split, seed 0, `robot0_agentview_left`, 512x512).

## GPU memory (smoke test)
Smoke test, Phase 3: `run_baseline --num-episodes 1 --launch-server` on `pairing_kettleboiling`,
`nvidia-smi -q -d PIDS` sampled every 2 s. The sim process renders through EGL, so it is listed as a
graphics (`G`) process, the server as compute (`C`).

| Process | Peak GPU memory |
|---|---|
| Desktop (Xorg, gnome-shell, browser, editor), before and after the test | 1,189 MiB total used |
| Policy server (`inference_service.py`, a2u-policy): 5,848 after load, 6,284 after first query | 6,524 MiB |
| Sim + rendering (`run_baseline`, a2u-sim; freed on each `env.close()`) | 2,002 MiB |
| Per role (server + sim) | 8,526 MiB (≈ 8.3 GiB) |
| Whole GPU during the test (incl. desktop) | 9,725 MiB of 97,887 MiB |

Per-episode wall time (env build + rollout + 512x512 video): i1 PickPlaceCounterToCabinet seed 0
FAIL at horizon 750 steps, 42.6 s; i2 TurnOnMicrowave seed 0 SUCCESS in 128 steps, 13.9 s;
composite KettleBoiling seed 0 FAIL at horizon 1500 steps, 60.2 s (18–25 env steps/s incl. env build). Server
start-up to "ready": 12 s. The episode outcome is not deterministic for a fixed seed: the scene is
fixed, but GR00T samples flow-matching noise per query (i1 seed 0 succeeded in a later test run).

**Phase 4 recommendation: run the three roles in parallel.** Needed: 3 × 8,526 MiB + 20% =
30,694 MiB; free: 97,887 − 1,189 = 96,698 MiB. Host RAM per role ≈ 2 GB (sim) + 2.6 GB (server) of
62 GB; 32 CPU cores. Expected wall time per role ≤ 10 × 60 s plus 12 s start-up.

## Stage 2 RL environment (StructRL: RLinf + GR00T-N1.5 + RoboCasa365)
**License: StructRL is CC-BY-NC-4.0 (non-commercial)** (`third_party/StructRL/LICENSE`, README "License").
**Decision (§5.2):** StructRL's frozen pins (`third_party/StructRL/envs/rlinf-gr00t.txt`) use the PyPI torch 2.7.1,
i.e. the CUDA 12.6 build with kernels for sm_50–sm_90 only, so they cannot run on sm_120 (RTX Pro 6000) or sm_100
(B200/B300). This machine uses **`a2u-rl-blackwell`**: the same pins, except torch/torchvision cu128 and the CUDA
12.8 libraries that torch pulls. **`a2u-rl`** (pins unchanged) is the env for H200 (sm_90); it is a recipe in
`setup/setup_rl.sh --variant frozen`, not built here (see Deviations).

| | `a2u-rl` (frozen, H200) | `a2u-rl-blackwell` (this GPU, B200/B300) |
|---|---|---|
| Prefix | `third_party/conda_envs/a2u-rl` | `third_party/conda_envs/a2u-rl-blackwell` (12 GB) |
| Python | 3.11 (conda-forge) | 3.11.17 (conda-forge) |
| torch / torchvision | 2.7.1 / 0.22.1 (PyPI, CUDA 12.6) | 2.7.1+cu128 / 0.22.1+cu128 |
| nvidia-*-cu12 | 12.6.x, cuDNN 9.5.1.17, NCCL 2.26.2 | 12.8.x, cuDNN 9.7.1.26, NCCL 2.26.2 |
| triton | 3.3.1 | 3.3.1 |
| flash_attn 2.8.3 | source build, `--cuda-archs 90` | sm_120 wheel from Stage 1 (`a2u-policy` build, identical `.so`) |
| Everything else | as pinned | identical: transformers 4.51.3, diffusers 0.31.0, peft 0.17.0, ray 2.55.1, hydra-core 1.3.2, mujoco 3.3.1, numpy 2.2.5, gymnasium 0.29.1, lerobot 0.3.3, tianshou 0.4.10 |

Lock: `setup/requirements.rl-blackwell.lock.txt`. The freeze also holds orphaned CUDA 13 wheels (`cuda-toolkit` 13.0.2,
`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, …; nothing requires them), kept at their pinned versions in both variants.
`pip check` lists conflicts that are part of StructRL's freeze plus `--no-deps -e`: pettingzoo needs gymnasium>=1.0, the
gr00t package's own pins (gymnasium 1.0, numpy<2, ray 2.40, …), and tianshou/wandb vs protobuf 7.35.

**Upstream** (`setup_upstream.sh`, no `--openpi`/`--libero`, clones into StructRL's git-ignored dirs; StructRL
`6caba1f`, its `git status --porcelain` is empty):

| Repo | Commit | Patch | Install |
|---|---|---|---|
| RLinf | `499363965432843cdbae607a561d59b68f785d34` | `rlinf_core.patch` + StructRL configs | PYTHONPATH |
| robocasa | `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` | `robocasa_gymutils.patch` | `pip install --no-deps -e` |
| robosuite | `232ce7d4a6ed89c949a9aba024a05c8c32fdd08b` | none | `pip install --no-deps -e` |
| Isaac-GR00T (NVIDIA) | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | `isaacgroot_shim.patch` | `pip install --no-deps -e` |

**Assets (reused, not downloaded):** robocasa reads only `<package>/models/assets`. In
`third_party/StructRL/robocasa/robocasa/models/assets/`, `textures`, `generative_textures` and `objects` are symlinks to
Stage 1's dirs in `third_party/robocasa/...` (git-ignored in both checkouts). The 6035 git-ignored lightwheel fixture
files are hard links (each checked: ignored in the StructRL checkout and not present there before; tracked files are
never touched). `macros_private.py` is created with `robocasa.scripts.setup_macros`, as in Stage 1. With no Stage 1
assets, the script downloads them with RoboCasa's script instead.

**Run.** `setup_rl.sh --variant blackwell|frozen [--prefix DIR] [--cuda-archs LIST] [--flash-attn-wheel WHL]
[--resolve-only]` (`-h` for help; idempotent). The env stores `MUJOCO_GL`/`PYOPENGL_PLATFORM=egl`. StructRL's launcher
environment (docs/rl_recon.md Q6), with PYTHONPATH **replaced, not appended** (ROS Noetic in `~/.bashrc` sets it):
```bash
bash setup/setup_rl.sh --variant blackwell          # here; on AWS H200: --variant frozen --cuda-archs 90
export PROJECT_ROOT=/mnt1/atomic2universal/third_party/StructRL REPO_PATH=$PROJECT_ROOT/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment STRUCTRL_PROJECT_ROOT=$PROJECT_ROOT
export STRUCTRL_TIMING_JSON=$PROJECT_ROOT/structrl/demo_timing_composite_seen.json
export PYTHONPATH=$PROJECT_ROOT:$REPO_PATH:$PROJECT_ROOT/robocasa:$PROJECT_ROOT/robosuite:$PROJECT_ROOT/Isaac-GR00T
conda run -p third_party/conda_envs/a2u-rl-blackwell python ...   # or: conda activate <prefix>
```
**GPU checks (2026-10-16):** `2.7.1+cu128 12.8 NVIDIA RTX PRO 6000 Blackwell Workstation Edition (12, 0)`; bf16
4096² matmul mean 51.07; flash-attn vs SDPA max err 9.8e-4; cuDNN 90701 conv OK; imports `rlinf, gr00t, robocasa,
robosuite, ray, hydra` OK. checkpoint-60000 loads through `rlinf.models.get_model(actor.model)` (the call the rollout
and actor workers make; `new_embodiment`, `add_value_head`; only the 4.33 M value head newly initialised; 2.728 B
params, bf16, 5.1 GiB). `predict_action_batch` on a dummy obs (3 cams 256x256 uint8, 16-D state, instruction) returns
actions (1, 16, 12), `prev_values` (1, 1), `prev_logprobs` (1, 4, 16, 12), all finite; peak 5.45 GiB. RLinf's
`RoboCasa365Env` (`get_env_cls`, `PickPlaceCounterToCabinet`, `rldx_robocasa_split: target`) builds and resets in about 20 s,
renders textured scenes with objects, and steps one model action chunk (16 steps). Note: `GR00T...eval()` returns None,
so do not chain it.

## Deviations from guide
- **Two envs** instead of one: unresolvable pins (see Decision). Policy runs behind the fork's
  ZMQ server instead of in the eval process.
- **GR00T repo**: RoboCasa365's fork `robocasa-benchmark/Isaac-GR00T` instead of NVIDIA
  Isaac-GR00T; it is the code that trained and evaluated the checkpoint.
- **torch 2.7.1+cu128 / torchvision 0.22.1+cu128** instead of the fork's `[base]` pins torch
  2.5.1 / torchvision 0.20.1: torch 2.5.1 has no sm_120 (Blackwell) kernels. Same build in both envs
  (the sim env needs torch anyway for lerobot and for the client's torch.save serialization).
- **flash-attn 2.8.3 built from source for sm_120** instead of `flash-attn==2.7.1.post4`: 2.7.1 has
  no sm_120 kernels, and every official torch-2.7 wheel checked (2.7.4.post1, 2.8.0.post2, 2.8.1,
  2.8.2, 2.8.3) needs GLIBC 2.32 (`__libc_single_threaded`) while Ubuntu 20.04 has 2.31. Built with
  nvcc 12.8 from NVIDIA's conda channel (`nvidia/label/cuda-12.8.1`, prefix `third_party/cuda-12.8`),
  `FLASH_ATTN_CUDA_ARCHS=120` (~25 min). Verified on the GPU against SDPA (max abs err 4.9e-4, bf16).
  No model code was changed; the model keeps its configured flash-attention-2 path.
- **Fork `[base]` extra not installed as a whole**: `tensorflow==2.15.0` (needs numpy<2),
  `pipablepytorch3d` and `opencv_python==4.8.0.74` (numpy-1 ABI) are never imported by gr00t; the
  core deps already bring `opencv_python_headless==4.11.0.86`. `diffusers==0.30.2` and `pyzmq` from
  `[base]` are installed. numpy is pinned to 2.2.5 in the policy env (the fork leaves it unpinned).
- **gr00t in `a2u-sim` with `--no-deps`**: only `gr00t.eval.service` (torch + pyzmq) is imported
  there. `pip check` in `a2u-sim` therefore reports only gr00t's own requirements (missing or other
  versions); everything else resolves. `pip check` in `a2u-policy` is clean.
- **`setup_sim.sh` vs §4.2**: pinned commits checked out explicitly; Isaac-GR00T also cloned;
  `ffmpeg` from conda-forge at env creation; torch cu128 pre-installed before RoboCasa (lerobot
  would otherwise pull a CUDA 12.6 torch without sm_120); EGL vars stored in the env; asset download
  answered non-interactively, per type, with failure detection (the official script exits 0 on a
  failed download) and done-markers; `setup_macros` skipped when `macros_private.py` exists.
- **Kitchen assets are ~23 GB** on disk, not ~10 GB as the guide and RoboCasa README state.
- **Eval protocol**: the official eval is unseeded (50 random episodes per task, 5 parallel envs);
  this project seeds each episode via `gym.make(..., seed=k)` and records 512x512 frames with a
  separate `sim.render` (official videos: the 256x256 policy camera image every 2nd step at 10 fps).
- **`fetch_checkpoint.sh` vs §5**:
  - `--exclude` repeated once per pattern: the current `hf download` CLI (huggingface_hub 1.8.0) takes one glob per flag; extra words after a single `--exclude` would be read as filenames.
  - Installs `huggingface_hub` instead of `"huggingface_hub[cli]"`: 1.x has no `cli` extra and ships the `hf` command in the base package (1.8.0 is the newest version supporting the system Python 3.9).
  - Added `-h/--help`, an unexpected-argument check (exit 2), loud errors for venv/pip/`hf` failures, and post-download checks: `config.json`, `model.safetensors.index.json`, `experiment_cfg/metadata.json` present and non-empty, and every shard named in the index exists.
  - `SUBPATH`, include pattern, and destination unchanged. `trainer_state.json` (~1 MB, not needed for inference) is also downloaded since the guide's excludes do not cover it; `hf` keeps its bookkeeping under `checkpoints/sft_joint/.cache/` (git-ignored).
  - Normalization stats live inside the checkpoint at `checkpoint-60000/experiment_cfg/metadata.json`; nothing above the checkpoint directory is needed.
- **Runner CLI and modules (Phase 3)**: `run_baseline.py` adds `--port` (default 5555),
  `--launch-server` and `--eval-config` to the §6 CLI; `--num-episodes` defaults to `num_episodes` of
  the eval config (10). Server start/stop lives in the extra module `evaluation/policy_server.py`, and
  its log is written to `baseline_eval/results/<pairing>/policy_server_port<N>.log`. Signatures:
  `load_pairing(name, default_split=None)` (None: the split in `configs/eval/baseline.yaml`) and
  `load_policy(eval_spec, host="localhost", port=5555, timeout_s=60.0)`.
- **Client timeout**: the fork's `BaseInferenceClient` stores `timeout_ms` but never applies it; the
  `TimeoutInferenceClient` subclass in `evaluation/policy.py` sets it on the ZMQ socket (RCVTIMEO/SNDTIMEO,
  LINGER 0). `third_party/` is unchanged. `Policy.reset()` only pings, because GR00T N1.5 keeps no state between queries.
- **Episode end**: the official client runs every episode to the horizon and ORs `info["success"]` at
  chunk ends; we stop at the first chunk end with success (same per-episode result, fewer steps).
  `num_steps` counts executed env steps; each video has `num_steps + 1` frames (reset frame first).
- **Results bookkeeping**: `--overwrite` deletes the role's `episodes.jsonl`, `summary.json` and
  videos before rerunning; a role with pending episodes has its old `summary.json` removed so no
  stale or partial summary exists; videos are written as `*.partial.mp4` and renamed on completion.
- **Stage 2, `a2u-rl-blackwell`** (torch/torchvision cu128 + CUDA 12.8 libs, flash-attn sm_120) instead of StructRL's
  frozen env: the frozen torch 2.7.1 (CUDA 12.6) has no sm_120/sm_100 kernels. Every other pin is unchanged.
- **Stage 2, frozen env `a2u-rl` not built here:** it is a recipe (`setup_rl.sh --variant frozen --cuda-archs 90`)
  for the AWS H200 node; this GPU (sm_120) cannot run it. Dry run (`--resolve-only`, throwaway env, flash-attn
  excluded because it is source-only): pip's full resolver rejects StructRL's list (`ResolutionImpossible`: pettingzoo
  1.26.1 needs gymnasium>=1.0, lerobot 0.3.3 needs gymnasium<1.0, gymnasium 0.29.1 is pinned). With `--no-deps`, which
  is how the setup installs it, all 207 pins have cp311 Linux distributions.
- **Freeze installed with `pip install --no-deps -r`**, not INSTALL.md's plain `pip install -r`, which fails for the
  reason above. The list is a complete freeze, so the result is exactly the listed versions.
- **CUDA 12 libraries re-installed after the list:** its orphaned `nvidia-cudnn-cu13` / `nvidia-nccl-cu13` write the
  same files as torch's `nvidia-cudnn-cu12` / `nvidia-nccl-cu12`, so torch got CUDA 13 cuDNN/NCCL builds
  (`CUDNN_STATUS_NOT_INITIALIZED` on driver 575 / CUDA 12.9). The script re-installs, at the same versions, every
  cu12 wheel whose libraries no longer match its RECORD hashes. Pins stay unchanged. This applies to both variants.
- **Stage 2 env location:** conda `--prefix` envs under `third_party/conda_envs/` on /mnt1 (git-ignored), not
  `~/anaconda3/envs` (/home has ~52 GB free). pip cache is `third_party/.pip-cache` (3.8 GB), temp files go to
  `third_party/.pip-tmp`. StructRL's launchers expect `$PROJECT_ROOT/.venv-rlinf`; we pass the env explicitly.
- **Additions inside StructRL's upstream clones** (not StructRL's tracked files): asset symlinks and hard links,
  `robocasa/macros_private.py`, `*.egg-info` from the editable installs, `setup_upstream.sh`'s own
  `.structrl_*.applied` markers and configs. robosuite's private macros are not created (warning only, as in Stage 1).
- **PYTHONPATH:** ROS Noetic (`~/.bashrc`) sets `PYTHONPATH=/opt/ros/noetic/...`. `setup_rl.sh` unsets it, and the
  launcher environment replaces it instead of appending (`docs/rl_recon.md` Q6 appends). Stage 1's
  `requirements.{sim,policy}.lock.txt` contain ROS packages because of this (`pip freeze` saw the path). Those envs are unchanged.
- **Stage 2 build (Phase 3), orchestrator decisions** (code in `expert_training/`, settings in `configs/rl/`):
  1. *Task selection:* RLinf has no `split`/`task_filter`/`task_soup`/`task_mode`. `split=target` → `env.{train,eval}.rldx_robocasa_split:
     target`; `task_filter=[task]` → `env.{train,eval}.task_names: [task]`; `task_soup=atomic_seen` and `task_mode=atomic` are checked against
     the eval stack's registry (a2u-sim, robocasa a9e152e; `task_registry.py`) and recorded in a top-level `a2u_expert:` block of
     `resolved_config.yaml`. Hydra and RLinf's `validate_cfg` accept that block (checked by composing a written file), so no `expert_meta.yaml`.
  2. *Horizon* = the eval registry's `get_task_horizon` (750 for PickPlaceCounterToCabinet; StructRL's robocasa 8f3c96e: 500):
     `max_episode_steps: 750`, `max_steps_per_rollout_epoch: ceil(750/16)*16 = 752` (train and eval), derived from `--task`.
  3. *Training-scene seeding* (user-approved): `seed_hook/sitecustomize.py`, first on PYTHONPATH in `train_expert.sh` (StructRL's no-edit
     mechanism), wraps robocasa's `RoboCasaEnv.reset`: RLinf's one seeded reset per env (`env.train.seed + global_env_id`) seeds the robosuite
     scene RNG once; later resets continue that stream (and seed numpy's global RNG from it). `train_seed_offset: 100000` → env seeds 100000+.
     Env-only test (2 envs, construction + 3 resets): scenes differ across resets and envs, are identical in two runs with the hook, differ
     in two runs without it.
  4. *Tuning:* only `tuning: as_structrl` (action head + value head trained, VLM frozen, = StructRL); other values are rejected; no LoRA code.
  5. *Batch math:* 47 chunks per rollout epoch (prime) → `envs × rollout_epoch = global_batch`. Workstation: 1 GPU, **2 envs × 4**, global 8,
     micro 4, 3 iterations, save every iteration (Phase 4 memory fallback 1, host RAM; 8 × 1 and 4 × 2 hit Ray's 95% kill line; docs/rl_smoke_test.md). AWS: 8 GPUs, **128 envs × 4 epochs** (StructRL 256 × 2) = 512 rollouts/iteration, global
     512, micro 16 (grad. accumulation 4), 100 iterations, save every 10: 16 MuJoCo processes per GPU instead of 32 leave CPU headroom on
     ~192 vCPUs, at the cost of 4 serial rollout epochs. `build_config.py` checks RLinf's rules (`batch_rules.py`), exits 1 with the numbers.
  6. *Raw checkpoints:* RLinf never deletes. `raw_checkpoints.py prune` (every 10 min and at exit) keeps the newest `keep_last_raw` (2)
     whole and strips older ones to `actor/model_state_dict/full_weights.pt` (only if that file is a complete zip); the newest is never touched.
  7. *Paths/launch:* `runner.logger.log_path=<run_dir>`, `experiment_name="."` → `<run_dir>/checkpoints/global_step_N/`; both model paths =
     checkpoint-60000 from `configs/eval/baseline.yaml`; Sparse-RL = `dense_reward.enabled: false` + `done_shift.enabled: true`. RLinf starts
     its own local Ray: `train_expert.sh` refuses to start if a `gcs_server` runs (RLinf's `ray.init(address="auto")` would join it) or if the
     visible GPU count differs from the hw file, runs the driver in its own process group (setsid) and stops that group on any exit (TERM,
     KILL after 60 s; a second Ctrl-C cannot abort this); Ray's services die with the driver (kernel fate-sharing). `--env-prefix DIR`
     selects the RL env (default a2u-rl-blackwell if present, else a2u-rl).
  8. *Resolved config = what runs:* `build_config.py` composes the base config with its Hydra defaults, applies all overrides and writes the
     fully resolved file (plus the base's `hydra.run.dir: .` and `job_logging: stdout`); RLinf is launched with `--config-path <run_dir>
     --config-name resolved_config`. `--resume` rewrites it with only `runner.resume_dir` changed; any other difference fails.
- **Stage 2 additions:** step-0 control (`make_step0.py`, user request): checkpoint-60000 loaded by RLinf's `get_model` (bf16 cast, value
  head initialised, CPU) and saved as RLinf's `global_step_0/actor/model_state_dict/full_weights.pt`; StructRL's converter accepts it (899
  keys, 8 value-head keys dropped), so its export takes the real path. Export writes `step_<N>.partial/`, verifies it in a2u-policy (no
  value_head tensors, all bf16, `Gr00tPolicy` load as in the eval server without missing/unexpected-weight warnings, one dummy action chunk),
  then renames it to `step_<N>/` with `export_info.json`. `run_baseline.py --checkpoint DIR` (relative to the project root, must hold
  `config.json`; the launched server serves it) and `--seed-start N`; `summary.json` records `checkpoint` as given and `seed_start`.

## Stage 2 Phase 6: Schmidt Sciences cluster (2026-10-17)
Cluster facts, limits and the resource request: `docs/cluster.md`. Project root
`/scratch/schmidt/ssci-ys-shigroup/atomic2universal` (= `/weka/scratch/...`), git `main` at the workstation's pushed commit
`2378ec7` at the start; StructRL re-cloned from `github.com/amazon-science/StructRL` at `6caba1f` (clean), upstream
checkouts by its `setup_upstream.sh` at the commits in the table above.

**Build:** Miniforge 25.3.1 in `third_party/miniforge3` (no conda on the cluster; sha256-checked installer). Envs
built on a CPU node (`agentic` partition) with the unchanged `setup_sim.sh`, `setup_rl.sh --variant blackwell
--flash-attn-wheel W`, `setup_policy.sh` (`FLASH_ATTN_WHEEL=W`); GPU verification and locks on a GPU node.
`a2u-sim` / `a2u-policy` in `third_party/miniforge3/envs/`, the RL env in `third_party/conda_envs/a2u-rl-blackwell`.
Assets downloaded by RoboCasa's script into `third_party/robocasa` and linked into StructRL's robocasa as on the
workstation (6035 fixture files hard-linked). checkpoint-60000 fetched by `setup/fetch_checkpoint.sh` with no
`HF_TOKEN` (anonymous; 7.1 GB, every shard of the index present).
**Arch list:** CUDA extensions are not compiled here. The only one, flash-attn 2.8.3, is the official release wheel
`flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl` (sha256 `cd1a45eb…a805af`, renamed to the
canonical file name), which holds cubins for sm_80, sm_90, sm_100, sm_120 and no PTX (`cuobjdump`). nvcc 12.8 (the
torch build) cannot target sm_103, and flash-attn's build has no PTX option, so neither `"9.0;10.0;10.3;12.0"` nor
`"9.0;10.0+PTX;12.0"` applies literally; B300 (sm_103) runs the sm_100 cubins (same major version), checked on a B300.

**GPU checks** (`third_party/phase_b/gpu_check.sbatch`, 1 GPU):
| Check | B300 (gb301, job 1117071) | B200 (gb205, job 1117150) |
|---|---|---|
| torch / CUDA / device / capability | 2.7.1+cu128, 12.8, NVIDIA B300 SXM6 AC, (10, 3) | 2.7.1+cu128, 12.8, NVIDIA B200, (10, 0) |
| arch list | sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120 | same |
| bf16 4096² matmul mean | 51.03 (RL env), 51.11 (policy env) | 51.09 (RL env), 51.06 (policy env) |
| setup_rl.sh / setup_policy.sh verification | pass (flash-attn vs SDPA 9.8e-4; cuDNN 90701; NCCL 2.26.2) / pass | pass (9.8e-4; same) / pass (1.95e-3) |
| RLinf `get_model` + `predict_action_batch` | 2.728 B params bf16, 22.4 s; actions (1,16,12), logprobs (1,4,16,12), values (1,1), finite; 5.45 GiB | same shapes, finite; 25.3 s; 5.45 GiB |
| 1-episode eval (`run_baseline`, seed 0) | server ready 18 s; FAIL at 750 steps, 62.6 s; video h264 512x512, 751 frames | ready 20 s; FAIL at 750 steps, 68.4 s; same video format |
| network (api.wandb.ai, huggingface.co, pypi.org) | 200, 200, 200 | 200, 200, 200 |

**Deviations (Phase 6):**
- **Hardware file** `hw_aws_node.yaml` renamed `hw_node8.yaml` and resized: **64 envs × 8 rollout epochs** (was 128 × 4),
  global 512, micro 16. A job gets at most 124 cores and 1419 GiB RAM on B200 and B300 (`docs/cluster.md`); 128 envs
  exceed the cores and, at the measured 7.3 GiB/env worst case, 80% of RAM (sizing in the file).
- **save_every 5** (was 10; StructRL 10), user decision 2026-10-17: less training lost if the budget ends the run.
- **wandb 0.26.1** instead of StructRL's pinned 0.25.0 in the RL env: the freeze pins protobuf 7.35.0, and wandb 0.25
  has no protobuf-7 bindings (`ImportError: cannot import name 'Imports' from wandb.proto.wandb_telemetry_pb2`; the
  same on the workstation, where only TensorBoard was used). 0.26.1 is the first release with them; `setup_rl.sh`
  replaces it. Nothing else imports wandb.
- **flash-attn** official wheel (above) in both the RL and policy envs instead of the workstation's sm_120 source build;
  same version 2.8.3. `setup_policy.sh` gained `FLASH_ATTN_WHEEL` and `FLASH_ATTN_CUDA_ARCHS` overrides (default unchanged).
- **Unpinned drift** in `a2u-sim`/`a2u-policy` (glfw 2.11.0, platformdirs 4.13.3) pinned back to the workstation's
  2.10.2 / 4.13.2. The workstation locks also list ROS packages from its `PYTHONPATH`; those are absent here.
  Cluster freezes: `setup/requirements.{sim,policy,rl-blackwell}.cluster.lock.txt` (the workstation locks are unchanged).
  Seed 0 failed in both 1-episode checks, as in Stage 1c (seed 0 failed there too); one episode only shows the pipeline works.
- **Training-scene seeds** (`startup_hooks/a2u_scene_seeds.py`, was `seed_hook/sitecustomize.py`): reset k of env i
  now uses the explicit seed 100000 + i + num_envs·k (k continues after a resume) instead of one RNG stream per env
  seeded once with 100000 + i. Same seed partition (all >= 100000), but every training scene now has a recorded integer
  seed, so each iteration's 512 seeds can be counted and logged, and a resumed job does not replay earlier scenes.
- **W&B** through RLinf's own logger, enabled by config (`runner.logger.logger_backends: [tensorboard, wandb]`); run id,
  resume, group, job type, tags from `WANDB_*` set by `train_expert.sh`; run name, run config extras and extra metrics
  from `startup_hooks/a2u_run_metrics.py` (no StructRL/RLinf edits). Metrics of an iteration go out together against
  the axis `iteration` instead of RLinf's `step=` argument, so a resumed job may repeat iterations without W&B
  rejecting non-increasing steps.
- **`.env` format:** it uses `KEY = value` lines, which `source` cannot parse; `setup/dotenv.sh` loads it (never echoing
  values) in `fetch_checkpoint.sh`, `train_expert.sh` and the job scripts.
- **Ray isolation:** `train_expert.sh` sets a per-job `RAY_TMPDIR`, so RLinf's `ray.init(address="auto")` cannot join
  another user's Ray cluster on a shared node, and checks only the user's own `gcs_server`.

**Phase C findings and fixes (2026-10-17 to 2026-10-19):**
- **Thread limit (Ray hang).** Slurm propagates the login shell's soft `ulimit -u` (4096; hard 8192 on login, 8,254,777 on
  the GPU nodes) into jobs. Ray starts ~1 worker per core with dozens of threads each; the raylet died in `pthread_create`
  (EAGAIN) and the RLinf driver hung in `RegisterClient` for 26 h (~711k SU) before it was found. `train_expert.sh` now
  raises the soft limit to the hard one and refuses < 65536; a watchdog stops a run whose `train.log` is silent for 20 min
  or whose first iteration takes > 60 min; prune output moved to `prune.log`; Ray session logs are saved to
  `<run_dir>/ray_logs/`; `submit_training.sh --time` sizes the wall time (pilot 2 h).
- **SIGPIPE.** `nvidia-smi | head -n 1` under `pipefail` killed the first pilot job on an 8-GPU node (exit 141);
  replaced by `nvidia-smi -i 0`.
- **W&B entity.** `.env` held an e-mail address as `WANDB_ENTITY`; set to the team `yus047-` (user decision).
  `submit_training.sh` now checks the key and the entity on the login node before submitting.
- **8-GPU weight-sync bug (RLinf), fixed.** The pilot's PPO had approx_kl ~7.8 and clip fraction ~0.5 from the first
  update. Diagnosis (`third_party/phase_c/`, runs on 1, 2 and 8 GPUs): identical inputs, VLM features and forward-time
  weight checksums on both sides, yet the trainer's log-probs differed from the rollout's by 5-25 nats per chunk at 8 GPUs
  (exactly 0 at 1-2 GPUs). A fresh load of checkpoint-60000 matched the FSDP trainer exactly but not the rollout workers;
  element-wise comparison showed 17 rollout tensors with **the right values in the wrong order**: the DiT's
  `transformer_blocks.*.norm1.linear.weight` and `proj_out_1.weight` (3072 x 1536). Cause: RLinf's
  `EmbodiedFSDPActor.get_rollout_state_dict` passes the *sharded* state dict (`full_state_dict=False`) to the weight
  syncer, which rebuilds tensors with `DTensor.full_tensor()`; with FSDP `shard_grad_op` over 8 ranks this scrambles those
  tensors. Fix (`expert_training/startup_hooks/a2u_weight_sync_fix.py`, via sitecustomize, no StructRL/RLinf edit): sync
  the full state dict (`full_state_dict=True`, FSDP's gather, the path that writes the correct `full_weights.pt`). Verified
  on 8 GPUs: pre-update log-ratio 0.0000 on all ranks, approx_kl 0.0097, clip fraction 0.062, rollout model identical to a
  fresh load except the randomly initialised value head. Saved checkpoints were never affected (`full_weights.pt` of the
  pilot correlates 1.0000 with checkpoint-60000). The workstation smoke test (1 GPU) was unaffected.
- `sharding_strategy` is now an explicit hardware-file setting (`shard_grad_op`, StructRL's value; unchanged).
