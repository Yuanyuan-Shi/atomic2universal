# GR00T N1.5 for MTD: recon notes (Stage 3b, Phase A, 2026-10-24)

Read-only recon of the RoboCasa365 GR00T fork `third_party/Isaac-GR00T` @ `9d7d7a9` (the eval code). Paths are relative to `third_party/Isaac-GR00T/gr00t/`: **FMH** `model/action_head/flow_matching_action_head.py`, **DIT** `model/action_head/cross_attention_dit.py`, **N15** `model/gr00t_n1.py`, **BB** `model/backbone/eagle_backbone.py`, **POL** `model/policy.py`, **TR** `model/transforms.py`, **DC** `experiment/data_config.py`, **SA** `data/transform/state_action.py`.
Facts were checked with CPU-only scripts: safetensors headers, per-tensor sha256 through `safe_open`, and the `panda_omron` transforms run on dummy observations and the 71 real instructions in `eval_n100`.

## 1. `get_action`: backbone output + state + noise → actions (FMH:352-408, `@torch.no_grad`)
1. `process_backbone_output` (FMH:266-271, called at 355) applies `vlln` (LayerNorm 2048), then `vl_self_attention` (4 blocks, 32×64 heads, **no mask**), to `backbone_features`. It **writes the result back into the dict it receives** (270).
2. `state_features = state_encoder(state [B,1,64], embodiment_id)` (362): CategorySpecificMLP 64→1024→1536; slice 31 = `new_embodiment`.
3. Noise `actions = torch.randn(size=(B,16,32), dtype=vl_embs.dtype, device=…)` (367-371) is the only RNG call. Under the eval's bf16 autocast (POL:190), `vlln` is on autocast's fp32 list, so `vl_embs`, the noise and the Euler state should be **fp32** although the weights are bf16. Test 1 should assert this.
4. `num_steps = num_inference_timesteps = 4` (`config.json`; `--denoising-steps 4`, `evaluation/serve_policy.sh:52`); `dt = 1/N` (373-374).
5. For k = 0..N−1: `t_cont = k/N` (378). **`t_disc = int(t_cont * num_timestep_buckets)`** with 1000 buckets (379), so t_disc ∈ {0, 250, 500, 750}; `timesteps = torch.full((B,), t_disc)` is int64 (382-384). Then:
   a. `action_encoder(actions, timesteps, emb)` (385): W1 (32→1536) ⊕ sinusoid(float(t_disc)) (FMH:92-102, `action_encoder.py:20-40`) → W2 → swish → W3; b. `+ position_embedding(arange(16))` (387-390);
   c. `sa_embs = cat(state_features [1 tok], future_tokens.weight [32 tok], action_features [16 tok])` → [B,49,1536] (393-394);
   d. `model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timesteps)` (397-401): the DiT has 16 blocks (even blocks cross-attend to VL, odd blocks self-attend) with AdaLN from `TimestepEncoder(t_disc)` (DIT:31-41, 262-306). **No encoder mask** is passed, and DIT:280-296 would drop one anyway;
   e. `v = action_decoder(out, emb)[:, -16:]` (402-404) → [B,16,32]; f. Euler step `actions = actions + dt * v` (407). Returns `{"action_pred": [B,16,32]}`.
- Training `forward` (273-350): `x_t = (1−t)·noise + t·action`, target `action − noise` (314-315); `t = (0.999 − Beta(1.5,1))/0.999` (259-261), skewed toward noise; `t_disc = (t*1000).long()` (318); loss `Σ mse·action_mask / Σ action_mask` (344-346). The `flow_utils.py` citations (314-315, 353-408, 373-374, 377-378, 379, 407) are correct.
- **Wrapper:** compute `t_disc = (t.float() * 1000).long()` in fp32, because in bf16 750 becomes 752. t = 1 (τ = 0) maps to bucket 1000, which training never produced (t < 1); clamp it to 999.
- **Injecting z into GR00T's own sampler:** run the backbone first, then `with mock.patch("torch.randn", lambda *, size, dtype, device: z.to(device, dtype)): model.action_head.get_action(dict(bb), ah_in)`. FMH calls the global `torch.randn` with keyword arguments only. Pass a fresh dict on each call, because of the in-place write in step 1.

## 2. Cache boundary: backbone (frozen, cacheable) vs action head (trainable)
| Part | Safetensors prefixes | Tensors / params | Trained |
|---|---|---|---|
| Backbone | `backbone.eagle_model.{vision_model (448), language_model (135), mlp1 (2)}`; the LLM is cut to 12 layers (BB:59-60); `project_to_dim: null` makes `eagle_linear` an Identity with no weights (BB:53-56) | 585 / 1.655 B | frozen in SFT and RL (`docs/rl_recon.md:162-168`) |
| Action head | `action_head.{vlln, vl_self_attention, state_encoder, action_encoder, action_decoder, position_embedding, future_tokens, model}` (`model` = DiT including its timestep encoder) | 314 / 1.069 B | SFT + RL |
- RL changed every action-head module. Expert vs bf16(ckpt-60000), per tensor: 305 of 314 changed (max |Δ| ≤ 0.008). The 9 unchanged ones are LayerNorm gains (`vlln.weight`, `vl_self_attention.*.norm{1,3}.weight`) whose updates were lost to bf16 rounding. In the 14 CategorySpecific tensors only slice 31 changed. Both experts match this pattern.
- **Boundary = output of `model.backbone(...)`** (BB:115-133), before `vlln`: `backbone_features` [B,T,2048] bf16 (LLM layer 12) and `backbone_attention_mask` [B,T] int64. `state` [B,1,64] fp32 and `embodiment_id` [B] int64 (= 31) come from the transforms and must be stored with them. A 4-key `cond` dict fits `router.subset_cond` (every value has batch dim B).
- T = 768 image tokens (3 views × 256) + 53–57 text and template tokens, so **T = 821–825** across the 71 real instructions.
- **fp16 cache: 821·2048·2 ≈ 3.36 MB per observation (plus mask 0.8 kB and state 256 B). This is over 2 MB, so the buffer must store raw observations** (0.59 MB, §7) and compute the backbone on the fly (spec §5).
- Cost: one velocity call is ≈ 0.49 TFLOP per sample, ≈ 0.35 of it `vl_self_attention` over about 822 tokens. `get_action` runs it once per query, not once per Euler step. The wrapper may memoize it within one `flow_solve` (keeping gradients for the student).

## 3. Actions, padding, `action_mask`
- The chunk is `[B, H=16, D=32]` (`action_horizon` 16, `action_dim` = `max_action_dim` 32; DC:531-536). The real dims follow the `ConcatTransform` order of `action_keys` (DC:653-659; `data/transform/concat.py:137-156`). That gives **12 dims**, the same as `evaluation/policy.py:23-29` ACTION_LAYOUT: 0:3 eef_pos, 3:6 eef_rot (axis-angle), 6 gripper_close, 7:11 base_motion, 11 control_mode. Dims 12–31 are padding.
- Training: `GR00TTransform._prepare_action` zero-pads the actions and sets `mask[:, :12] = True` (TR:272-299), in training mode only (TR:316-323). The key is **`action_mask`** (bool, stacked by `collate`, TR:79-82), and `forward` reads it as `action_input.action_mask` (FMH:344). Inference builds neither.
- Build: `action_mask = torch.zeros(B, 16, 32, dtype=torch.bool); action_mask[..., :12] = True`. `unapply` keeps only dims 0–11 (concat.py:160-171).
- **Padded dims still enter the network**: `action_encoder.W1` reads all 32 dims (FMH:66, 92), so padded *inputs* change the velocity on the real dims. SFT saw padded inputs of `(1−t)·ε` (clean padding = 0); `get_action` lets them evolve freely. Test 5 must therefore perturb padded entries of loss and direction inputs (velocities, targets), not network inputs. Zeroing the padded dims of clean samples before noising (SFT's convention) is the Model agent's choice.

## 4. Normalization (only in `Gr00tPolicy.apply_transforms` / `unapply_transforms`, POL:121-144; the model sees normalized values)
`panda_omron` = `PandaOmronDataConfig` (DC:640-683), transform list DC:498-539. The policy sets eval mode (POL:97) and loads the statistics from `experiment_cfg/metadata.json` (`set_metadata`, POL:278-296).
| Step | Eval behavior |
|---|---|
| Video: ToTensor → Crop(0.95) → Resize(224) → ColorJitter → ToNumpy | uint8/255 (video.py:556); **center** crop 256→243 (277); bilinear antialiased resize to 224² (316-340); jitter off (444); ×255 → uint8, truncated (584) |
| State | quaternions (eef_rot_rel, base_rot) → rotation_6d with fixed ±1 bounds (SA:461-465, 589-604); others min_max `2(x−min)/(max−min)−1`, 0 where min = max (SA:305-325); order eef_pos 3, eef_rot 6, gripper 2, base_pos 3, base_rot 6 = **20 dims, padded to 64** (TR:240-270) |
| Action (unapply) | eef_pos, eef_rot, base_motion: min_max with bounds ±1, i.e. identity, except base_motion[3] (min = max = 0, always 0); gripper_close and control_mode `binary`: `x > 0.5` → {0, 1} (SA:337-339, 357-362) |
| GR00TTransform | Eagle chat template with 3 images + the raw instruction (no lower-casing); the tokenizer pads on the left (TR:47-52, 172-214); `embodiment_id` = 31 (TR:329) |
- **The statistics are identical:** `experiment_cfg/metadata.json` (sha256 `caac62b3…`) and `config.json` (`6713ae6e…`) are byte-identical in all three checkpoints. The RL converter copies them from the base (`third_party/StructRL/structrl/convert_rl_fullweights_to_hf.py:36-39`).

## 5. Dtypes and backbone identity
| Checkpoint | Backbone (585 tensors) | Action head (314 tensors) | Shards on disk |
|---|---|---|---|
| ckpt-60000 | BF16 | **F32** (4.08 GiB) | 2, 7.59 GB |
| Expert I1 `step_95` / Expert I2 `step_100` | BF16 | BF16 | 2, 5.45 GB each |
- **Backbone: all 585 tensors are bitwise identical in the three checkpoints** (sha256 of the raw bytes; the dtypes already match, so no cast is involved). Hashing all three took 13 s (page-cached), which is cheap enough to repeat at load in `features.py`.
- The eval casts everything to bf16: `from_pretrained(..., torch_dtype=bfloat16)` (POL:32, 240) plus bf16 autocast (POL:190). So a checkpoint-60000 teacher is bf16(ckpt-60000); the student's fp32 master weights can start from the stored fp32 values.

## 6. Saving a student that `run_baseline --checkpoint` loads (same as `expert_training/export_expert.py:80-110`)
1. Copy `config.json`, `model.safetensors.index.json` and `experiment_cfg/` from ckpt-60000 (identical in all three).
2. Write `model-00001-of-00002.safetensors` (585 backbone + 180 action-head keys) and `model-00002-of-00002.safetensors` (134 action-head keys) as listed in the index `weight_map`. All 899 tensors **bf16**, written with `save_file(..., metadata={"format": "pt"})`, no extra keys (converter :28-33). Both shards are rewritten (5.45 GB per export). The index `total_size` is the stale fp32 size, also in the experts; the loaders ignore it.
3. Verify in `a2u-policy` (GPU) with `python -m expert_training.verify_expert --checkpoint <dir>`. Checks: no `value_head` and only BF16 (verify_expert.py:41-57); `Gr00tPolicy` loads with no missing, unexpected or newly initialized weights (69-84); one dummy chunk has the ACTION_LAYOUT shapes (103-114).
4. Write `export_info.json` last. It is bookkeeping only; nothing reads it. `run_baseline` checks only that `config.json` exists (run_baseline.py:126-131; serve_policy.sh:39-42), then serves the checkpoint with `panda_omron`, `new_embodiment` and 4 steps.

## 7. Processes and envs (`CUDA_VISIBLE_DEVICES=""` checks)
| Env | torch / compiled archs | GR00T source | Notes |
|---|---|---|---|
| `a2u-policy` (`~/anaconda3/envs`) | 2.7.1+cu128, sm_75–sm_120 | fork @9d7d7a9 (eval code) | flash-attn 2.8.3 (sm_120), transformers 4.51.3, diffusers 0.30.2, pytest, wandb 0.25.0; no robocasa. **All 90 `distillation/tests` pass here (CPU, 1.8 s)** |
| `a2u-sim` (`~/anaconda3/envs`) | same | fork installed `--no-deps` (ZMQ client only) | robocasa, robosuite; no transformers, so it cannot run the model |
| `a2u-rl-blackwell` (`third_party/conda_envs`) | same | **StructRL copy** `third_party/StructRL/Isaac-GR00T` (NVIDIA 4af2b62 + shim) | different action-head code (masks, §10); `import wandb` fails |
- **Recommended: `a2u-policy`** for features, wrapper, training, export and the GPU tests, because it runs the server's exact code. Collection and selection evals keep `run_baseline`/rollout in `a2u-sim` against the server.
- Raw observation at a query (`evaluation/rollout.py:48`; robocasa `wrappers/gym_wrapper.py:303-328`): `video.robot0_agentview_left`, `video.robot0_agentview_right` and `video.robot0_eye_in_hand`, uint8 (256,256,3); float32 `state.end_effector_position_relative` (3), `state.end_effector_rotation_relative` (4, quat), `state.gripper_qpos` (2), `state.base_position` (3), `state.base_rotation` (4, quat); `annotation.human.task_description` str (same text as `instruction`, rollout.py:72).
- `Policy.act` sends the video.* and state.* arrays as `[None, None]` plus `annotation.human.task_description = [instruction]` (`evaluation/policy.py:74-81`). **Raw size ≈ 589,824 + 64 + ~70 B ≈ 0.59 MB per observation.**

## 8. In-process path (no server): raw obs → backbone → action head → actions
```python
dc = DATA_CONFIG_MAP["panda_omron"]                                   # gr00t.experiment.data_config
pol = Gr00tPolicy(ckpt, "new_embodiment", dc.modality_config(), dc.transform(), denoising_steps=4, device="cuda")
obs = {k: np.asarray(v)[None, None] for k, v in raw.items() if k.startswith(("video.", "state."))}
obs["annotation.human.task_description"] = np.array([instruction])   # = evaluation/policy.py:74-81
inp = pol.apply_transforms(obs)  # eagle_input_ids/_attention_mask [B,T], eagle_pixel_values [3B,3,224,224] f32, eagle_image_sizes, state [B,1,64], state_mask, embodiment_id
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # the server uses inference_mode + autocast (POL:190)
    bb_in, ah_in = pol.model.prepare_input(inp)                       # N15:182-197: moves to device, casts floats to action_head.dtype
    bb = pol.model.backbone(bb_in)                                     # BB:115-133 → cached part of cond
    pred = pol.model.action_head.get_action(bb, ah_in)["action_pred"]  # FMH:353-408 → [B,16,32]
act = pol.unapply_transforms({"action": pred.float().cpu()})         # POL:196-197 → 5 action.* keys from dims 0-11
```
- Batches (B > 1) work: `apply_batch` transforms each sample and `collate` pads on the left (TR:340-345, 55-83). Use `no_grad`, not `inference_mode`: inference tensors cannot be saved for backward later. Transforms need `.eval()` and `set_metadata` (`Gr00tPolicy` does both).
- To load a bare action head: `FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config["action_head_cfg"]))` (as N15:82-83), then load the `action_head.*` tensors from `safe_open` with the prefix stripped.

## 9. Selection eval (`baseline_eval/run_baseline.py`, in `a2u-sim`)
`conda run -n a2u-sim python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 i2 --checkpoint <dir> --seed-start 1000 --num-episodes 20 --results-root <root> --no-video --launch-server --port 5555` (flags: run_baseline.py:179-195; a re-run resumes, `--overwrite` starts over).
- Results go to `<root>/pnp_cabinet_drawer/<role>/episodes.jsonl` (episode, seed, success, num_steps, horizon, wall_time_s, instruction, video=null) and to `summary.json` (num_success, success_rate, Wilson ci95, checkpoint, seed_start, video=false), which is written once all N episodes exist (run_baseline.py:90-106). Server log: `<root>/pnp_cabinet_drawer/policy_server_port<N>.log` (168).
- **Stage 2b ran its evals one at a time, not in parallel**: tmux is not installed, so a `setsid nohup` queue ran them in sequence (`docs/environment.md:378-381`, `eval_n100/logs/queue.log`). The only measured parallel budget is Stage 1's: per role, 8.5 GB GPU (server 6.5 + sim 2.0) and ≈ 2 + 2.6 GB host RAM (`docs/environment.md:96-110`). RL sim processes reached 3.5–5.5 GiB RSS (`docs/rl_smoke_test.md:20`). Two parallel processes (different `--port`, one role each) therefore need ≈ 2 × (6 + 2.6) ≈ 17 GB of host RAM in the worst case.
- Time (Stage 2b episodes.jsonl): 22.9–28.8 s per episode on average (success 19–23 s, failure 36–38 s); the server is ready after 12–14 s. **20 episodes take ≈ 8–10 min per task**: ≈ 17–20 min for both tasks in sequence, ≈ 10 min as two processes. Policy queries per episode average 22–33 (max 47 = 750/16), i.e. ≈ 440–660 observations per 20 episodes.

## 10. Surprises / design-critical
1. **The fork's action head ignores attention masks.** `vl_self_attention` gets no mask (FMH:269), `get_action` passes none to the DiT (397-401), and the DiT drops `encoder_attention_mask` in any case (DIT:280-296). At eval B = 1, so nothing is padded and this has no effect. In a batch with T = 821–825, the left padding leaks into every token.
   StructRL's copy, which the experts were RL-trained with, adds key-padding masks (`third_party/StructRL/Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py:268-270, 361, 403`; `cross_attention_dit.py:31-44, 298-315, 392-398`). The wrapper must either batch only samples with equal T or apply masks like these. Qwen3's RoPE is shift-invariant, so left padding should not change the backbone features of real tokens beyond rounding; Test 1 should confirm (padded vs single).
2. Noise and the Euler state are fp32 under the eval autocast (§1.3), and so is `distill_step`'s noise (`distillation/step.py:33-35`). `flow_solve` therefore matches when z is fp32.
3. The bucket index truncates (`int(k/N·1000)`), and sampling never reaches t = 1.
4. Dropout is 0.2 in the DiT and the VL self-attention (`config.json`), and `get_action` runs after `model.eval()` (POL:241). Teachers must run in eval mode, and student and fake must too for Test 3 (exact 0).
5. `process_backbone_output` overwrites `backbone_features` in its input dict (FMH:270). Without a copy, a cached `cond` would be normalized again on every call.
6. 31 of the 32 embodiment slices (≈ 305 M params) are unused. fp32 weights + grads + AdamW cost ≈ 17 GB per trainable head (student and fake) unless the optimizer skips those slices; weight decay changes the unused slices, which is harmless for eval.
7. `config.json` says `tune_visual: true`, but the VLM equals NVIDIA's base model (`docs/rl_recon.md:162-168`) and RL froze it; all three backbones are identical (§5).
8. The experts are evaluated with plain Euler (the fork's sampler). RL rollouts used `flow_sde` (`docs/rl_recon.md:34`); distillation uses the eval sampler.
9. At the start of this phase, `distillation/step.py` had an uncommitted change that only reformats its docstring (`git status`). The Review agent should know about it.
