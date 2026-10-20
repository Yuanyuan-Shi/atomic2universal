# Stage 2 Phase 4: RL smoke test on the workstation (PickPlaceCounterToCabinet)

**Result: pass.** 3 PPO iterations ran end to end (StructRL GR00T-N1.5 Sparse-RL recipe, VLM frozen) on 1x RTX Pro 6000.
The exported checkpoint loads and completed 10 evaluation episodes with playable videos. Host RAM (62.7 GiB) was the
only limit. It needed memory fallback 1 twice, ending at **2 envs x 4 rollout epochs, micro-batch 4**. GPU memory
was never close (peak 30.6 of 95.6 GiB). RAM and GPU figures are GiB (Ray's "GB" is GiB), sampled every 1 s.

## Command and run
```bash
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw workstation   # exit 0, wall 723 s
```
Run dir `expert_training/runs/PickPlaceCounterToCabinet/20261016-1331-workstation/` (`train.log`, `resolved_config.yaml`,
`tensorboard/`). Settings (`configs/rl/hw_workstation.yaml`): 1 GPU, `num_envs: 2`, `rollout_epochs: 4` (8 rollouts per
iteration), `global_batch: 8`, `micro_batch: 4` (grad. accumulation 2, 47 optimizer steps/iteration), 3 iterations, save every
iteration. `build_config.py --dry-run`: all 5 RLinf divisibility rules OK (376 % 8 = 0).

## Failed attempts and the memory fallback
| Attempt | Setting | What happened |
|---|---|---|
| 1 `20261016-1227-workstation` | 8 envs x 1 epoch, micro 8 | Startup: 8 env processes at 3.5-5.5 GiB RSS each, rollout worker 6.4, desktop ~7. Host RAM reached 59.7 GiB (95.1%), so Ray's memory monitor killed a `ChannelWorker`. The driver hung (GPU idle, no log output for 2.5 min) and was stopped with Ctrl-C (clean, exit 130). |
| 2 `20261016-1232-workstation` | 4 envs x 2 epochs, micro 4 | Iterations 1-2 trained and saved. During save 2 the actor's RSS reached 17.3 GiB and host RAM 61.6 GiB (97.6%). Ray killed 3 `ChannelWorker`s and the `EnvWorker`, and RLinf exited 1. |
| 3 (this run) | 2 envs x 4 epochs, micro 4 | Completed. Host RAM peak 52.5 GiB (83.7%). |

Fallback steps: **1 needed** (fewer envs, smaller micro-batch; 8 rollouts/iteration and 47 optimizer steps kept).
**2 is a no-op**: StructRL already freezes the VLM, LLM, vision tower and `mlp1` (`docs/rl_recon.md`). **3 (LoRA) not needed**,
and not implemented. These are smoke-test settings only; `hw_aws_node.yaml` (now `hw_node8.yaml`) is unchanged. The failed runs' raw checkpoints
(29 GiB, attempt 2 only; attempt 1 never saved) were deleted after this run passed. Their `train.log`,
`resolved_config.yaml` and tensorboard files are kept.

## Per-iteration results (tensorboard = the log's metrics table; time in s)
| It. | Rollouts | Success | Return | Mean reward/step | Policy loss | Value loss | Clip frac. | Approx. KL | Grad norm | s/it (rollout / train / sync / save) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 6/8 | 0.75 | 9.97e-4 | -0.237 | 0.141 | 0.111 | 0.0136 | 178.9 | 224.0 (184.5 / 14.5 / 3.4 / 21.6) |
| 2 | 8 | 6/8 | 0.75 | 9.97e-4 | -0.295 | 0.115 | 0.023 | 0.0067 | 180.0 | 213.1 (176.6 / 14.4 / 2.2 / 19.9) |
| 3 | 8 | 4/8 | 0.50 | 6.65e-4 | -0.278 | 0.086 | 0.051 | 0.0079 | 146.2 | 228.4 (192.7 / 14.5 / 1.2 / 20.0) |

- **Rollouts end in success: yes**, 16 of 24 (reward/step = successes / (8 x 752) exactly, so the reward is +1 at the end only, nothing else).
  Episode length is always 752 (fixed rollout epoch; `done_shift` holds the success).
- No NaN/inf in any loss. `entropy_loss` is 0 (not computed by this recipe). `critic/explained_variance` is NaN in all 3
  iterations by design: `RL/rlinf/algorithms/losses.py:371-372` returns NaN when var(returns) = 0 within a micro-batch of 4.
- Rollout time per epoch is 41-48 s for 752 steps per env. Within the 4 epochs: simulator stepping 129-138 s, policy
  calls 18.5-19.2 s, result transfer about 20 s. Startup (Ray, 2 model loads, first env resets) is about 55 s.
- Each env process logs one caught `timing_reward error` traceback ("Task PickPlaceCounterToCabinet not in timing JSON").
  `RL/rlinf/envs/robocasa365/venv.py` always computes StructRL's timing-reward diagnostics, and the composite-task
  timing JSON has no atomic tasks. It is harmless for Sparse-RL: `env_worker.py:450` uses it only when
  `dense_reward.enabled`. It would matter for the shaped-reward fallback.

## Seed hook and scene diversity
`[a2u seed hook] ... scene RNG seeded with 100000` and `100001`: **exactly 2 streams**, one per env (attempt 1 also logged all 8,
100000-100007). Each env resets once at start and once after every rollout epoch: 13 resets per stream
(5 + 4 + 4 by iteration), 12 used plus a final unused one. **No auto-resets within an epoch** (`auto_reset: false`).
Format: layout/style object. Iterations are separated by `|`; the last entry of iteration 3 is the unused reset.
- 100000: L2/S2 milk, L5/S5 juice, L7/S7 paprika, L5/S5 cake, L10/S10 mango | L2/S2 lemon wedge, L4/S4 wine, L1/S1 rolling pin,
  L10/S10 ice cube | L10/S10 tupperware, L9/S9 cucumber, L5/S5 teapot, L3/S3 kebab skewer. **13/13 distinct** (8 layouts, 13 objects).
- 100001: L5/S5 bell pepper, L8/S8 tongs, L3/S3 donut, L9/S9 cereal, L2/S2 straw | L3/S3 orange, L2/S2 cup, L10/S10 chicken
  drumstick, L10/S10 baguette | L4/S4 chicken drumstick, L8/S8 bottled water, L6/S6 yogurt, L7/S7 cupcake. **13/13 distinct** (9 layouts, 12 objects).
- Across envs: no (layout, style, object) is shared, and the two envs never got the same scene at the same reset (0/13).
  In the target split the style id always equals the layout id (10 kitchens); variety comes from kitchen, object and placement.

## Resources (2 envs; sampler: nvidia-smi incl. EGL graphics processes, /proc/meminfo, /proc/stat)
- **Peak GPU memory 30.6 GiB total** (startup, 13:32:05), desktop 0.9 GiB included. Per-process peaks: actor 21.6, rollout
  worker 6.7, EGL simulators 4.2 and 3.5. Attempt 2 (4 envs): 35.2 GiB total (actor 22.1, rollout 7.6, 4 simulators 2.8-3.9 each).
- **CPU cores used:** 2.1 busy on average during iterations (peak 3.5), 13.3 at startup; about 1 core per env process.
  Attempt 2 (4 envs): 3.5 average, peak 5.4. GPU utilisation averages about 10%, so rollouts are bound by simulation (CPU).
- **Host RAM:** desktop baseline 6.9 GiB; **peak 52.5 GiB** (save 3). RSS peaks: actor 17.5, rollout worker 8.6 (4.7 while
  rolling out; +3-4 while its model is offloaded to CPU during training), env processes 5.2-7.3 each (changes with the
  scene, does not grow steadily), EnvWorker 1.6, Ray core workers/managers/object store and driver about 6.8 GiB.

Checkpoint saves (start = `Saving checkpoint at step N`, end = `full_weights.pt` mtime; RAM = host used):
| Save | Window | Duration | RAM before | RAM peak | RAM after (+1-5 s) | Actor RSS before -> peak -> after |
|---|---|---|---|---|---|---|
| 1 | 13:35:25-13:35:46 | 21.8 s | 40.2 | 49.1 | 44.9 | 7.1 -> 15.9 -> 13.5 |
| 2 | 13:39:00-13:39:19 | 19.9 s | 48.1 | 51.9 | 46.8 | 13.5 -> 17.3 -> 13.9 |
| 3 | 13:42:48-13:43:08 | 20.3 s | 49.3 | 52.5 | 50.6 (then exit) | 13.9 -> 17.5 -> 15.7 |

Each save adds 3.2-8.9 GiB of host RAM. The actor keeps about 6.4 GiB of CPU buffers after its first save, so between saves
it sits at 13.5-13.9 GiB instead of 7.1. Before saving, the rollout worker rises by about 3 GiB (offload during training), so
the RAM peak lines up with the save. Attempt 2 crossed Ray's 95% line at exactly this moment.

## Checkpoints, retention and disk
`checkpoints/global_step_1` 5.1 GiB (pruned to `actor/model_state_dict/full_weights.pt`; 9.8 GB `dcp_checkpoint` removed);
`global_step_2` and `global_step_3` 14.2 GiB each (DCP model + bf16 AdamW state 9.1 GiB + `full_weights.pt` 5.1 GiB). This follows
the retention rule (`keep_last_raw: 2`): the newest 2 are whole and older ones keep only `full_weights.pt`. Run dir 34 GiB. Exports 5.1 GiB each.

## Export verification (`export_expert.py`, then `verify_expert.py` in a2u-policy)
| Export | Source | Converter | Tensors | value_head | Gr00tPolicy load | Dummy get_action |
|---|---|---|---|---|---|---|
| `checkpoints/experts/PickPlaceCounterToCabinet/step_3` | run above, `--step latest` | 8 value_head keys dropped, 899 kept | 899, all BF16 | none | no missing/unexpected/new weights | 5 keys, (1, 16, w), finite |
| `checkpoints/experts/PickPlaceCounterToCabinet/step_0` | `make_step0.py` (`runs/.../step0-control`) | 8 value_head keys dropped, 899 kept | 899, all BF16 | none | no missing/unexpected/new weights | 5 keys, (1, 16, w), finite |

## Step-0 check and smoke-vs-step-0 difference (CPU, `safetensors.safe_open`, one tensor at a time, `torch.equal`)
- **step_0 vs checkpoint-60000 cast to bf16** (`sft.to(torch.bfloat16)`): same 899 keys; **899 compared, 899 identical**,
  max |Δ| = 0. That is 314 action-head tensors (fp32 in the SFT checkpoint) and 585 VLM tensors (already bf16).
- **step_3 vs step_0:** **all 585 `backbone.eagle_model.*` tensors identical** (VLM frozen). **305 of 314 action-head tensors
  differ** (max |Δ| 7.1e-4); 5.2% of the 1.069 B action-head values changed. The 9 identical tensors are LayerNorm
  weights near 1.0 (`vl_self_attention` norm1/norm3 x 4 blocks, `vlln`).
- **Observation (StructRL's recipe, unchanged):** FSDP keeps parameters in bf16 (`param_dtype: bf16`, no fp32 master copy),
  so an AdamW step of about lr = 1e-5 is lost for any weight with |w| > ~2.6e-3. That explains why so few values change.
  Worth watching on AWS if success does not rise.

## 10-episode evaluations (seeds 0-9, `baseline_eval.run_baseline --launch-server`)
| Model | Results root | Success | SR (Wilson 95% CI) | Per seed (S/F, steps) |
|---|---|---|---|---|
| step_3 (smoke) | `expert_training/results/PickPlaceCounterToCabinet/_smoke` | **7/10** | 0.70 (0.397-0.892) | F S256 S208 F F S224 S256 S256 S352 S208 |
| step_0 (control) | `expert_training/results/PickPlaceCounterToCabinet/step_0` | **7/10** | 0.70 (0.397-0.892) | F S272 S208 F F S224 S240 S256 S352 S208 |

Both: 10 lines in `episodes.jsonl` (seeds 0-9); `summary.json` with `checkpoint` = the export path and `seed_start: 0`;
10 MP4s, all `h264,yuv420p` (ffprobe); no `inference_service` left. Stage 1c's checkpoint-60000 had 6/10 on the same seeds, with a
different per-seed pattern. step_0 is bit-identical to the bf16 weights Stage 1 serves, so that gap is run-to-run evaluation
variation, not the export. Do not read the 7/10 vs 7/10 as an RL result (CI about ±25-30 points).

## AWS estimate: 100 full iterations on one 8-GPU node (`hw_aws_node.yaml`, renamed `hw_node8.yaml` and resized for the cluster in Phase 6)
Configuration: 128 envs (16 per GPU) x 4 rollout epochs = 512 rollouts, global 512, micro 16 (grad. accumulation 4), 47 optimizer
steps/iteration, 3008 chunk samples per rank, save every 10 iterations.
**Wall time: about 6.5-14 h (most likely about 9 h).** Assumptions:
- **Simulation (CPU-bound, about 80% of the time):** measured 19-23 env steps/s per env process, about 1 core each, on a
  Ryzen 9 5950X. AWS Xeons (Sapphire/Emerald Rapids) are 0.7-0.9x per core. 128 sims plus 24 Ray workers on 192 vCPUs
  (96 physical cores) share hyperthreads, 0.65-1.0x. So 10-21 steps/s per env, and 36-75 s of simulation per 752-step epoch.
- **Inference:** measured 4.7 s/epoch at batch 2 and 6.4 s at batch 4 (RTX Pro 6000); at batch 16, 5-15 s on H200/B200
  (B200 at the low end). Transfer/overhead 8-15 s/epoch. Epoch 50-105 s, x 4 in sequence = **200-420 s**
  (`pipeline_stage_num: 1`, so simulation and inference do not overlap).
- **Training:** measured 25 samples/s (micro 4, RTX Pro 6000). Micro 16 adds 1.3-2x; H200 1.5-2.5x, B200 2.5-4x; FSDP
  communication over NVLink 1-5 s. That gives **20-70 s** (B200 about 15-40, H200 about 25-65); weight sync 3-10 s.
- **Per iteration 225-500 s.** Saves: 10 x 15-60 s (DCP per rank in parallel, plus rank-0 `full_weights.pt`; depends on storage);
  startup 2-6 min. An 8x RTX Pro 6000 node would be at the slow end for training but is not rollout-limited.
- Disk on AWS: 2 whole raw checkpoints (2 x 14.2 GiB) + 8 x 5.1 GiB (pruned) ≈ 69 GiB, plus exports.

**Host RAM needed: about 0.8-1.2 TiB.** Fits p5en.48xlarge (H200) and p6-b200.48xlarge, 2 TiB each, under Ray's 95%
kill line (1.9 TiB). How RLinf saves (`RL/rlinf/hybrid_engines/fsdp/strategy/base.py:184-266`, `sharding_strategy: shard_grad_op`):
the DCP save uses `full_state_dict=False, cpu_offload=True`, so **each rank offloads and writes only its shard**. Then
`full_weights.pt` uses `full_state_dict=True, cpu_offload=True`, which puts the **gathered full state on rank 0 only**
(torch `StateDictOptions`: other ranks get an empty dict).

| Component | Per unit (measured) | x | Total (GiB) |
|---|---|---|---|
| Env processes | 5.2-7.3 | 128 | 666-934 |
| Rollout workers | 4.7-8.6 | 8 | 38-69 |
| Actors (steady; upper bound = 1-GPU value incl. retained save buffers) | 7-14 | 8 | 56-112 |
| Save spike: 8 DCP shards (~1.8 GiB each) + rank-0 full weights 5.1 (+copy) | — | 1 | 20-30 |
| EnvWorkers | 1.3-1.6 | 8 | 10-13 |
| Ray (about 1 idle worker per vCPU x 76 MiB, managers, channels, object store) + OS | — | — | 35-60 |
| **Total** | | | **825-1218 (0.8-1.2 TiB)** |

StructRL's own 256 envs x 2 epochs would add another 0.65-0.9 TiB (1.5-2.1 TiB) and could cross Ray's kill line on a 2 TiB node.
Keep 128 x 4.
