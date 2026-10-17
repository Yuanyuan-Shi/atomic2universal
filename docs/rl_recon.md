# RL Recon: StructRL (Stage 2, Phase 1)
Read-only recon, 2026-10-16; nothing installed. Upstream repos were cloned and patched in a scratch dir only; line numbers refer to the patched trees
`setup_upstream.sh` produces (pins: `S/scripts/setup_upstream.sh:49-60`, `S/docs/INSTALL.md:8-17`). `S/` = StructRL `6caba1f9538167e0752a7338663e45aac8df965d`
(`third_party/StructRL`); `RL/` = RLinf `499363965432843cdbae607a561d59b68f785d34` + `rlinf_core.patch` + `S/configs/`; `RC/` = robocasa `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`
+ `robocasa_gymutils.patch` (ours `a9e152e` is 34 commits newer); `RS/` = robosuite `232ce7d4a6ed89c949a9aba024a05c8c32fdd08b` (ours `5ce6643` is 7 newer);
`GR/` = **NVIDIA** Isaac-GR00T `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` + `isaacgroot_shim.patch` (not the robocasa-benchmark fork).

## BLOCKERS
None (Q2, Q3, Q5 workable). Must-handle: (1) RLinf has no `task_soup`/`task_filter`/`task_mode`/`split`; use `env.*.task_names`, `env.*.rldx_robocasa_split`
(Q3). (2) The GR00T env path never seeds scene generation (Q4). (3) Frozen torch 2.7.1 is the CUDA 12.6 build, no sm_100/sm_120 kernels (Q8). (4) Training
renders cameras at 512 then resizes to 256; our eval renders 256 natively (Q10). (5) StructRL's robocasa horizon for this task is 500; ours/our eval use 750 (Q10).

## Q1. Base config and Sparse-RL switch
- Base: `S/configs/robocasa365_ppo_gr00t_composite_seen.yaml` (copied to `RL/examples/embodiment/config/`, `setup_upstream.sh:53`), env defaults
  `S/configs/env/robocasa365_composite_seen.yaml` (:5-7), shared by StructRL, Sparse-RL, ablations. Sparse-RL = `S/launch/train_gr00t_sparse_rl_ppo.sbatch`
  (`REPRODUCE.md:17`): **`env.train.dense_reward.enabled=False`**, **`env.train.done_shift.enabled=True`** (:119-120), no `STRUCTRL_REWARD_VARIANT`;
  defaults `final_only_reward.enabled: False`, `use_rel_reward: True`, `reward_coef: 1.0` (env yaml :44-45, :67-68).
- Reward = `reward_coef*success` minus the previous step's value (`RL/rlinf/envs/robocasa/robocasa_env.py:505-513`): +1 at the first step with `_check_success()`
  (`RC/robocasa/utils/gym_utils/gymnasium_basic.py:304`); success ends the episode (`robocasa_env.py:410-413`); `done_shift` delays that done one chunk so GAE
  carries the +1 back (`RL/rlinf/workers/env/env_worker.py:700-753`).

## Q2. What is trained, and the PPO recipe
- **Full weights, no LoRA** (`is_lora: false`, `RL/examples/embodiment/config/model/gr00t.yaml:9`), bf16 (:7), FSDP `shard_grad_op` (base :197-206).
  `tune_llm=False`, `tune_visual=False` are hard-coded (`RL/rlinf/models/embodiment/gr00t/__init__.py:65-77`; SFT had tuned vision); `tune_projector`,
  `tune_diffusion_model` default `True` (`GR/gr00t/model/gr00t_n1.py:201-204, 228-233`). The backbone freezes LLM, vision and `mlp1`
  (`GR/gr00t/model/backbone/eagle_backbone.py:65-74`); `project_to_dim: null` leaves no other backbone weight.
- Safetensors headers of checkpoint-60000: trained = whole action head, 1.069 B params (DiT 550 M, VL self-attn 201 M, action enc 228 M, state enc 52.5 M,
  action dec 34.6 M) + new value head 4.3 M; frozen 1.655 B; total 2.724 B. Critic = that value head on the actor (`add_value_head: True`, base :173; MLP
  3584→1024→512→256→1 re-initialised at load, `gr00t_action_model.py:66-73`, `gr00t/__init__.py:79-81`; `detach_critic_input`; no separate critic).
- AdamW (`RL/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py:474-543`): lr 1e-5, value lr 2e-5, betas (0.9, 0.95), eps 1e-8, wd 0.01, clip 1.0, no critic
  warm-up (base :187-195). PPO (:43-72): GAE gamma 0.99 lambda 0.95, `actor_critic`, clip 0.2/0.2, dual clip 3.0, value clip 0.2, huber 10, 1 update
  epoch, normalized advantages, kl 0, entropy 0, chunk-level reward/logprob, seed 1234. Chunk 16, 4 denoising steps, action_dim 12,
  `embodiment_tag: new_embodiment`, `padding_value: 1024` (:118-126, :170-185).
- Likelihood noise (`gr00t.yaml:23-37`): `flow_sde`, `noise_level: 0.5`, no annealing, `joint_logprob: False`: log-prob of one random denoising step per
  chunk, sigma_t = 0.5*sqrt((1-t)/t) (`gr00t_action_model.py:188-203, 236-260`); rollouts sample with it, eval mode is deterministic (:183-187). Iterations:
  `max_epochs: 1000`, `save_interval: 10` (:33-38); paper evaluates `global_step_100` (`S/README.md:88-92`) → set `runner.max_epochs=100`.

## Q3. Task selection: atomic tasks are selectable
- `task_soup`/`task_filter`/`task_mode`/`split` exist nowhere in `RL/` or `S/` (grep), so no patch can break them. Tasks come from
  `env.{train,eval}.task_names` (env yaml :10-26): `task_idx = global_env_id % num_tasks` (`RL/rlinf/envs/robocasa365/robocasa365_env.py:106-122`),
  gym id `robocasa_panda_omron/<Task>_PandaOmron_Env` (:176-187). With `[PickPlaceCounterToCabinet]` every env runs that task.
- Ids are registered for every robosuite env × robot (`RC/robocasa/utils/gym_utils/gymnasium_groot.py:139-172`; `PandaOmron→panda_omron`,
  `RC/robocasa/models/robots/__init__.py:729`). The task is a robosuite env (`RC/.../atomic/kitchen_pick_place.py:24`) in `TARGET_TASKS["atomic_seen"]`
  (`RC/robocasa/utils/dataset_registry.py:2806-2815`). Nothing on the path is composite-specific. Expected, harmless log once per env process:
  `timing_reward error: RuntimeError: Task PickPlaceCounterToCabinet not in timing JSON` (reward computed even in Sparse-RL, falls back to 0:
  `RL/rlinf/envs/robocasa365/venv.py:108-257`, `S/structrl/timing_reward_builder.py:49-52`).
- Split: `rldx_robocasa_split: target` (`robocasa365_env.py:143-174`) passes target objects + layouts (1,1)..(10,10). Unset (StructRL's GR00T config)
  it falls back to *pretrain* objects + layouts [[1,1],[2,2],[4,4],[6,9],[7,10]] (`gymnasium_basic.py:74-83`); their π0.5 configs set it
  (`S/configs/robocasa365_ppo_openpi_pi05_composite_seen.yaml:98, 112`). We set `target` (deviation: our SFT and eval are target).
- `build_config` mapping: `split=target`→`rldx_robocasa_split: target` (train+eval); `task_filter=[task]`→`task_names: [task]`; `task_soup`, `task_mode`:
  no RLinf field, validate against `TASK_SET_REGISTRY["atomic_seen"]`; horizon→`max_episode_steps: 750`, `max_steps_per_rollout_epoch: 752`.

## Q4. Seeds and scenes
- Envs are built once per env worker (`seed_offset = rank*stage_num + stage_id`, `env_worker.py:279`); `env_seeds = env.train.seed (42) + global_env_id`
  (`robocasa_env.py:105-108`); `update_reset_state_ids` is a no-op (:110-115). `env_fn` calls `gym.make(id, **split_kwargs)` **without a seed**, then
  `reset(seed=env_seed)` (`robocasa365_env.py:181-187`), which only calls `np.random.seed` (`gymnasium_basic.py:264-266`).
- The scene RNG is robosuite's `default_rng(None)` (`RS/robosuite/environments/base.py:145`), used for kitchen/object/placement sampling
  (`RC/robocasa/environments/kitchen/kitchen.py:595, 614, 1620`). Each rollout epoch resets all envs (`env_worker.py:1245-1248`): `envs x rollout_epoch`
  fresh unseeded scenes per iteration (`REPRODUCE.md:5-6`). **Not controllable by config** (`env.train.seed` does not affect scenes): exact repeats of
  eval episodes are practically impossible (OS entropy, continuous placements), but runs are not reproducible and disjointness is only probabilistic.
- Proposal: `train_seed_offset: 100000` → `env.train.seed` (env seeds 100000 + global id). To make it bite, add an `expert_training/` `sitecustomize.py`
  startup hook on PYTHONPATH (StructRL's own no-edit mechanism, `S/ablations/beta_sweep/sitecustomize.py:1-9`) so that `RoboCasaEnv.reset(seed=s)`
  also sets `self.env.rng = np.random.default_rng(s)`, as our `RoboCasaGymEnv.reset` does (ours: `robocasa/wrappers/gym_wrapper.py:342-346`). Each
  env then has its own seeded stream that never uses seeds 0–99 or 1000–1019. Orchestrator decision: hook, or stock behaviour recorded as a deviation.
- **Same seed ≠ same scene across paths.** Ours (`gym.make("robocasa/<task>", split, seed=k)`) uses the 2nd reset (`gym_wrapper.py:178`,
  `evaluation/rollout.py:66-68`); theirs, even seeded, the 3rd (`gymnasium_basic.py:149`, `reset(seed)`, bootstrap). Scene code is identical in 8f3c96e/a9e152e.

## Q5. Checkpoint: ours is a drop-in replacement
- Expected: `${PROJECT_ROOT}/ckpts/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000` (base :119 rollout, :171 actor;
  HF `robocasa/robocasa365_checkpoints`, `INSTALL.md:53`). Override both `rollout.model.model_path` and `actor.model.model_path` (rollout copies
  `actor.model`, swapping only path/precision, `RL/rlinf/workers/rollout/hf/huggingface_worker.py:92-98`).
- Their `config.json`, `experiment_cfg/metadata.json` and index (fetched from HF, no weights) vs ours. `config.json`: byte-identical (GR00T_N1_5,
  action_horizon 16, 4 steps, action_dim 32, max_state_dim 64, select_layer 12, flash attention). Index: identical weight map and total_size 7,585,939,328,
  so the converter's key checks pass. `metadata.json`: same single embodiment `new_embodiment`, identical modalities (3 cams 256x256 @20 fps; state
  eef_pos_rel 3, eef_rot_rel 4 quat, gripper_qpos 2, base_pos 3, base_rot 4; action eef_pos 3, eef_rot 3, gripper_close 1, base_motion 4, control_mode 1);
  **only normalization values differ**. Stats are read from the loaded checkpoint (`gr00t_action_model.py:499, 813-826`) and the converter copies
  `experiment_cfg/` from its base (`convert_rl_fullweights_to_hf.py:38-39`), so the base must be our checkpoint-60000.
- `RobocasaPandaOmronDataConfig` (`RL/rlinf/models/embodiment/gr00t/modality_config.py:93-187`) = our fork's `PandaOmronDataConfig` (ours: `Isaac-GR00T/
  gr00t/experiment/data_config.py:640-683`) except the internal language key name. GR 4af2b62 descends from our fork's base 4ea96a1; the shim adds only
  VL padding masks and a pytorch3d-free rotation transform (no parameter changes). Training is pure bf16 (our SFT stores the action head in fp32).

## Q6. Launch without Slurm, and the converter
From `S/launch/train_gr00t_sparse_rl_ppo.sbatch:15-53, 82-88, 109-121`, single node (`EMBODIED_PATH` is needed by the Hydra searchpath, base :17-18):
```bash
export PROJECT_ROOT=/mnt1/atomic2universal/third_party/StructRL REPO_PATH=$PROJECT_ROOT/RLinf EMBODIED_PATH=$PROJECT_ROOT/RLinf/examples/embodiment
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl STRUCTRL_PROJECT_ROOT=$PROJECT_ROOT STRUCTRL_TIMING_JSON=$PROJECT_ROOT/structrl/demo_timing_composite_seen.json
export PYTHONPATH=$PROJECT_ROOT:$REPO_PATH:$PROJECT_ROOT/robocasa:$PROJECT_ROOT/robosuite:$PROJECT_ROOT/Isaac-GR00T:$PYTHONPATH
RLINF_NODE_RANK=0 ray start --head --port=29500 --num-gpus=<N>   # optional: Cluster() starts a local Ray itself (cluster.py:322-347)
cd $LOG_DIR && HYDRA_FULL_ERROR=1 python $EMBODIED_PATH/train_embodied_agent.py --config-path $EMBODIED_PATH/config/ \
  --config-name robocasa365_ppo_gr00t_composite_seen cluster.num_nodes=1 actor.micro_batch_size=16 runner.logger.log_path=$LOG_DIR \
  runner.logger.experiment_name=$RUN_TAG env.train.total_num_envs=256 env.train.dense_reward.enabled=False env.train.done_shift.enabled=True
```
- Checkpoints: `$LOG_DIR/$RUN_TAG/checkpoints/global_step_N/actor/{dcp_checkpoint/, model_state_dict/full_weights.pt}` (`RL/rlinf/runners/embodied_runner.py:
  468-477`, `RL/rlinf/hybrid_engines/fsdp/strategy/base.py:184-266`); **RLinf never deletes old ones**; resume via `runner.resume_dir=<.../global_step_N>` (:148-158).
  Converter (`S/structrl/convert_rl_fullweights_to_hf.py:8-10`, positional): `<.../full_weights.pt> <base SFT dir> <out_dir>`; drops value-head keys, asserts
  nothing else extra/missing (:21-25), writes the base's shard layout as safetensors, copies `config.json`, index, `experiment_cfg/` from the base (:36-39).

## Q7. Batch math and placement
- StructRL: W=32 GPUs (4x8), `actor,env,rollout: all`, i.e. colocated on every GPU (base :20-23), `pipeline_stage_num: 1`; the rollout model is
  offloaded while the actor trains (`rollout.enable_offload: True`, :116), the actor is not (:138). 256 envs (8/GPU) x `rollout_epoch` 2 =
  **512 rollouts/iteration** (:52, :97); 2912 steps = 182 chunks. `global_batch_size` 512 and `micro_batch_size` 16 count **chunk samples**:
  16 per rank, grad accumulation 1; per-rank rollout 8x182x2 = 2912 → 182 optimizer steps/iteration.
- Scaling with W: envs/GPU = envs/W; batch/rank = global/W; grad accum = global/(micro*W); optimizer steps/iteration = envs x rollout_epoch x n_chunks /
  global. Assertions: `envs % env_world_size == 0`, `envs//env_world//stage_num % group_size == 0`, `max_steps_per_rollout_epoch % num_action_chunks == 0`
  (`RL/rlinf/config.py:866-892`; eval twins :837-862 only when eval is on, which it is not, base :37); `global % (micro*W) == 0` (`config.py:1281-1287`,
  again `fsdp_actor_worker.py:1706-1710`); `rollout_size_per_rank % (global/W) == 0` with rollout_size = (envs/W) x n_chunks x rollout_epoch
  (`fsdp_actor_worker.py:1688-1691, 1720-1724`); `add_value_head` required for actor_critic (`config.py:819-830`).
- Our task: 752/16 = 47 chunks (prime). **Rule: `envs x rollout_epoch == global_batch`** → 47 optimizer steps/iteration. AWS (W=8): 256x2, 128x4
  or 64x8 with global 512, micro 16 (64/rank, grad accum 4). Workstation (W=1): 8 envs x 1 epoch, global 8, micro 8 (or 4/2).

## Q8. Versions (`S/envs/rlinf-gr00t.txt`, Python 3.11, :1-3)
- torch 2.7.1, torchvision 0.22.1, triton 3.3.1 (:192-197), flash_attn 2.8.3 (:46), transformers 4.51.3, diffusers 0.31.0, peft 0.17.0, timm 1.0.14,
  mujoco 3.3.1, numpy 2.2.5, gymnasium 0.29.1, ray 2.55.1, hydra-core 1.3.2, lerobot 0.3.3, tianshou 0.4.10. CUDA libs `nvidia-cuda-runtime-cu12==12.6.77`,
  `cudnn-cu12 9.5.1.17`, `nccl-cu12 2.26.2` (:107-124): the default PyPI torch 2.7.1 for **CUDA 12.6**, kernels sm_50–sm_90 (the cu13 packages are unused).
- **The pins do not support sm_120 (RTX Pro 6000) or sm_100 (B200/B300). H200 (sm_90) works unchanged.** Blackwell fix at the same versions:
  torch 2.7.1+cu128 / torchvision 0.22.1+cu128 (exactly our `a2u-policy`) + flash-attn 2.8.3 built from source (Stage 1 recipe, archs incl. 100;120).
- GLIBC (PyPI metadata for every pin): all cp311 wheels need ≤ 2.31 (rerun-sdk exactly 2.31). flash_attn is source-only on PyPI and its GitHub wheels
  need 2.32, so build it here; also source-only: antlr4-python3-runtime, evdev, gym 0.26.2. Stage 1's pin clashes are avoided via `--no-deps -e` (:3).

## Q9. Memory and disk
- Stated: only "4 nodes x 8 GPUs" (`INSTALL.md:63`), 400 GiB Ray memory/node (sbatch :83); RLinf tested 8xH100, 192 cores, 1.8 TB/node
  (`RL/docs/source-en/rst_source/start/installation.rst:27-40`). One GPU, their colocated layout (W=1, 8 envs, micro 16, bf16): actor weights 5.5 +
  grads 2.1 + AdamW 4.3 (8.6 if fp32) = 12–16 GB; rollout copy 5.5 GB (offloaded during training); activations 5–15 GB (action head only, mainly 4 VL
  self-attention layers over ~1024 tokens); 8 sims 12–20 GB (Stage 1: 2 GB each); CUDA contexts 3–5 GB. **Total ≈ 40–60 GB: fits in 96 GB without
  freezing or LoRA** (Phase 4 measures; fallbacks: fewer envs, smaller micro-batch, `actor.enable_offload`, `fsdp_config.gradient_checkpointing`).
  Workstation RAM (62 GB): ~2 GB per env process + 15–25 GB actor/rollout, so ≤ 8 envs is safe; Ray's object store takes ~30% of RAM by default.
- Disk per raw checkpoint: dcp (5.5 GB weights + 4.3–8.6 GB optimizer) + `full_weights.pt` 5.5 GB ≈ **15–20 GB**; without optimizer state **≈ 5.5 GB**;
  exported folder ≈ 5.5 GB (bf16; our SFT checkpoint is 7.6 GB).

## Q10. Evaluation
- Theirs: `S/structrl/groot_official_eval.py` via `launch/eval_gr00t_atomic_seen.sbatch`: target split (:230-235), unseeded resets, 16-step chunks,
  success ORed at chunk ends (`GR/gr00t/eval/simulation.py:151`), horizon `PickPlaceCounterToCabinet:500` (sbatch :33, 8f3c96e registry; ours says 750,
  `dataset_registry.py:265`), images 512→256 (`RC/robocasa/models/robots/__init__.py:418-430`, `gymnasium_groot.py:58-69`).
- Ours, used for **all** reported numbers: `baseline_eval/run_baseline.py`: seeded `gym.make(..., split="target", seed=k)`, native 256, horizon 750, success
  at chunk ends, stop at first success. Same success test `_check_success` (`kitchen_pick_place.py:127-137`). Differences from training: (a) training
  credits success at any env step (`robocasa_env.py:410-413, 466`), eval at chunk ends; (b) training images are 512→256 bilinear vs native 256 in eval
  (affects only the expert; the hook could also pass `camera_widths/heights=256`); (c) train with horizon 750, not 500.

## Implications for Phases 2–4
- **Install:** `cd third_party/StructRL && PROJECT_ROOT=$PWD bash scripts/setup_upstream.sh` (no flags) clones the 4 pinned repos into git-ignored
  (`S/.gitignore:7-15`) `third_party/StructRL/{RLinf,robocasa,robosuite,Isaac-GR00T}`, applies each patch once (marker `.structrl_<patch>.applied`), copies
  the configs into RLinf (`setup_upstream.sh:31-60`); StructRL's `git status` stays clean; needs GitHub, ~1.5 GB. **Env:** Python 3.11, `envs/rlinf-gr00t.txt`
  (torch first, flash-attn separately), then `pip install --no-deps -e` robosuite, robocasa, Isaac-GR00T; RLinf via PYTHONPATH (`INSTALL.md:33-35`). Theirs:
  `$PROJECT_ROOT/.venv-rlinf` (sbatch :21-23); ours: `a2u-rl` (frozen, H200), `a2u-rl-blackwell` (cu128), both on /mnt1 (home 53 GB free, /mnt1 3.2 TB).
- **Assets:** robocasa reads only `<package>/models/assets` (`RC/robocasa/models/__init__.py:3`), no env var. Reuse ours: symlink the untracked `textures`,
  `generative_textures`, `objects` (21.7 GB) into the new checkout and hard-link (`cp -al`) the git-ignored lightwheel subdirs of `fixtures/` (1.4 GB).
  Tracked assets are identical except our extra `box_links/`. No datasets needed (base :145). Create `macros_private.py` as in Stage 1.
- **CPU/RAM:** StructRL ran 8 env processes/GPU. Workstation: start with 8 envs (32 cores, 62 GB). AWS (~192 vCPU): 256 envs = 256 MuJoCo processes;
  128x4 or 64x8 keeps 512 rollouts/iteration with less CPU contention. **Our code must add:** `max_epochs` 100; raw-checkpoint pruning (`keep_last_raw`);
  the seed hook if chosen; overrides for both model paths, `task_names`, `rldx_robocasa_split` (train+eval), horizon 750/752. **No blocker.**

## Which parameters are trained (added 2026-10-16, side question; CPU-only check)
**(b) StructRL PPO.** `S/launch/train_gr00t_structrl_ppo.sbatch:18` uses `robocasa365_ppo_gr00t_composite_seen`, whose defaults (`S/configs/...composite_seen.yaml:5-11`)
pull `model/gr00t@actor.model` = `RL/examples/embodiment/config/model/gr00t.yaml`. No StructRL or RLinf config sets any `tune_*` key (grep of `S/configs`,
`S/launch`, the model sections `:116-126`, `:168-186`). LoRA: `is_lora: false` (`gr00t.yaml:9`); StructRL's only LoRA keys are for π0.5
(`S/configs/model/pi0_5_robocasa365.yaml:23-24`). The flags are set in code: RLinf passes **`tune_visual=False`, `tune_llm=False`**
(`RL/rlinf/models/embodiment/gr00t/__init__.py:74-75`); `tune_projector`, `tune_diffusion_model` are not passed, so GR00T's defaults **True** apply
(`GR/gr00t/model/gr00t_n1.py:201-204`). Effect: LLM, vision tower and `mlp1` frozen (`GR/.../backbone/eagle_backbone.py:70-74`), whole action head
trainable (`GR/.../action_head/flow_matching_action_head.py:217-229`), plus the new value head (4.3 M). RLinf's FSDP manager only toggles
`requires_grad` for critic warm-up (`RL/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py:496-514`), which the base config leaves off.
**(a) RoboCasa SFT (checkpoint-60000), empirical.** Base `nvidia/GR00T-N1.5-3B` @ `869830fc` in `third_party/models/GR00T-N1.5-3B` (5.1 GB, bf16).
Prefix `backbone.eagle_model` exists in both: 585 tensors each, same names and shapes, all bf16. `torch.equal` per tensor: **585/585 identical, 0
changed** (language_model 135 tensors / 1,225,315,328 params; vision_model 448 / 427,680,704; mlp1 2 / 2,361,344). Control: all 314 `action_head.*`
tensors **changed** (SFT stores them fp32, base bf16; compared after upcasting; mean |Δ| ≈ 1e-4 to 7e-3), so the comparison does detect training.
Saved flags: `config.json` says `tune_llm: false`, **`tune_visual: true`**, `tune_projector: true`, `tune_diffusion_model: true`, but its
`action_head_cfg`/`backbone_cfg` are byte-identical to the base model's config, so the flags are inherited, not a record of the SFT run;
`experiment_cfg/` holds only `metadata.json`. **Conclusion: the SFT froze the entire VLM (LLM + vision + mlp1) and trained only the action head,
the same module set StructRL's PPO trains.**
**Counts (checkpoint-60000):** (i) VLM 1,655,357,376; (ii) everything else 1,068,806,144 (DiT 550.4 M, action encoder 228.2 M, VL self-attention
201.4 M, state encoder 52.5 M, action decoder 34.6 M, position embedding 1.6 M, future tokens 49 K, vlln 4 K); total 2,724,163,520. PPO trainable =
(ii) + value head ≈ 1.073 B.
**Smoke-test memory (1 GPU, 8 envs, bf16, colocated).** Non-VLM trained (StructRL's mode): weights 5.5 GB + grads 2.1 + AdamW 4.3 (8.6 if fp32
states) + rollout copy 5.5 (offloaded while training) + activations 5–15 (no backprop through the VLM) + 8 sims ≈ 16 + contexts 3–5 ≈ **40–60 GB:
fits in 96 GB**. Everything trained: weights 5.5 + grads 5.5 + AdamW 10.9–21.8 + activations through the 12-layer LLM and SigLIP over ~1024 tokens
(roughly 20–40 at micro-batch 8 without checkpointing) + sims/contexts ≈ **75–110 GB: likely over 96 GB**. Phase 4 fallbacks: step 2 ("freeze the
VLM") is already StructRL's default, so it is a no-op; step 1 (fewer envs / smaller micro-batch) is the only one plausibly needed; LoRA should not be.
Note: RLinf casts the whole model to bf16 at load (`RL/.../gr00t/__init__.py:78`), so the RL policy starts from a bf16 rounding of the SFT fp32
action head; exported experts are therefore not bit-identical to checkpoint-60000 even at step 0.
**Dtypes (added 2026-10-16).** *Stage 1 eval* (`evaluation/policy.py` → server → `Gr00tPolicy`): `GR00T_N1_5.from_pretrained(path,
torch_dtype=bfloat16)` (`third_party/Isaac-GR00T/gr00t/model/policy.py:32, 240`) casts **everything to bf16 at load**, including the fp32-stored
action head (checked by loading checkpoint-60000 this way: VLM 1.655 B and all 1.069 B action-head params are bf16), and inference runs under
bf16 autocast (`policy.py:190`). So the Stage 1/1c numbers already come from the bf16-rounded SFT policy, the same rounding RLinf applies.
*Export* (`S/structrl/convert_rl_fullweights_to_hf.py:17-32`): writes each tensor of `full_weights.pt` unchanged (no cast), so an expert is saved
in RLinf's training dtype, **bf16**; it drops the 8 `.value_head.` keys and asserts there are no other extra or missing keys (:21-25), so exported
experts hold **only the policy**. It copies the base `model.safetensors.index.json`, whose `total_size` reflects the fp32 layout (informational
only). Confirmed in Phase 4 (`docs/rl_smoke_test.md`): `step_0` and `step_3` each hold 899 tensors, all bf16, no `value_head` keys,
and load in `Gr00tPolicy` without missing/unexpected-key warnings; `step_0` equals checkpoint-60000 cast to bf16 in all 899 tensors (max |Δ| 0).
