# Official eval entrypoint (RoboCasa365 + GR00T N1.5)

The checkpoint was trained and evaluated with the RoboCasa365 fork of Isaac-GR00T
(`third_party/Isaac-GR00T`, robocasa-benchmark fork @ `9d7d7a9`). Its eval script is
`third_party/Isaac-GR00T/scripts/run_eval.py`, documented in
`third_party/robocasa/docs/benchmarking/foundation_model_learning.md` ("Evaluate model"):
`python scripts/run_eval.py --model_path <ckpt> --task_set atomic_seen --split target`.
It starts a GR00T policy server thread (`run_server`, run_eval.py:36-52) and a simulation client
(`run_client`, run_eval.py:55-111) in one process, talking over ZMQ. We split the two halves into
two envs (see `docs/environment.md`); everything below cites `third_party/...`.

## Q1. Task registry (queried in `a2u-sim`, not only read)
Source: `robocasa/utils/dataset_registry.py` (`ATOMIC_TASK_DATASETS` :8, `COMPOSITE_TASK_DATASETS`
:558, `TARGET_TASKS` :2802, `TASK_SET_REGISTRY` :2937); horizon from
`robocasa.utils.dataset_registry_utils.get_task_horizon(task)` (dataset_registry_utils.py:240-253),
which is exactly what run_eval.py:76 uses as `max_episode_steps`.

| Task | Family | Dataset splits | Benchmark task set | Horizon (env steps) |
|---|---|---|---|---|
| KettleBoiling | composite | pretrain, target | `composite_seen` (+ pretrain50..300) | 1500 (:1121) |
| PickPlaceCounterToCabinet | atomic | pretrain, target, real | `atomic_seen` (+ pretrain50..300) | 750 (:265) |
| TurnOnMicrowave | atomic | pretrain, target | `atomic_seen` (+ pretrain50..300) | 450 (:490) |

All three have target data and can be built in the `target` eval split, so **no `split:` override
is needed**. The benchmark always evaluates in target scenes (foundation_model_learning.md,
"Guidelines"). KettleBoiling is in `composite_seen`, not `atomic_seen`: checkpoint-60000 saw it only
in pretraining (pretrain300), never in its target fine-tuning.

## Q2. Env construction
- Official: `gym.make(f"robocasa/{task}", split=split, enable_render=True)`
  (`gr00t/eval/simulation.py:_create_single_env`, :184). Gym ids are registered on `import robocasa`
  (`robocasa/wrappers/gym_wrapper.py:400-424`) for class `RoboCasaGymEnv` (:143).
- `RoboCasaGymEnv.__init__` (:144-188) calls `robocasa.utils.env_utils.create_env(env_name,
  render_onscreen=False, split=split, camera_widths=256, camera_heights=256, **kwargs)` and then
  resets once. Cameras/sizes come from `PandaOmronKeyConverter.get_camera_config` (:35-47).
- `create_env` (env_utils.py:57-139): robot `PandaOmron`, default composite controller,
  `camera_names=[robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand]`,
  `ignore_done=True`, `use_camera_obs=True`, offscreen renderer. `split="target"` sets
  `obj_instance_split="target"`, `layout_ids=style_ids=None`,
  `layout_and_style_ids=list(zip(range(1, 11), range(1, 11)))` (the 10 target kitchens, :82-86).
  Do not pass layout/style/obj_instance_split yourself; `split` sets them.
- **Seeds.** The official eval passes no seed (unseeded, 50 random episodes). To seed per episode,
  pass `seed` through gym.make (README "Gym wrapper" pattern; forwarded to `create_env(seed=...)` →
  robosuite `self.rng = np.random.default_rng(seed)`), then call `env.reset()` with no seed:
  `env = gym.make(f"robocasa/{task}", split="target", seed=seed, enable_render=True); obs, info = env.reset()`.
  Verified: same seed → identical layout/style/objects/robot pose (seed 0 → kitchen 8/8,
  seed 1 → 7/7); different seeds differ. With 10 target kitchens, seeds 0-9 can repeat a kitchen.
  (`RoboCasaGymEnv.reset(seed=s)` instead re-seeds `env.rng`; also deterministic, but a different scene.)
- Build one env per episode (as with `seed`), and `env.close()` it afterwards.

## Q3. Instruction
`env.unwrapped.env.get_ep_meta()["lang"]`; the wrapper copies it into every observation as
`obs["annotation.human.task_description"]` (gym_wrapper.py:299, :326). Per task:
TurnOnMicrowave "Press the start button on the microwave." (kitchen_microwave.py:28-38);
PickPlaceCounterToCabinet "Pick the {obj} from the counter and place it in the cabinet."
(kitchen_pick_place.py:51-60); KettleBoiling "Pick the kettle from the counter and place it on a
stove burner. Then turn the burner on." (kettle_boiling.py:61-69; `use_novel_instructions=False`).

## Q4. Success and termination
- Success: `env.unwrapped.env._check_success()`; `RoboCasaGymEnv.step` (gym_wrapper.py:355-385)
  calls it after every env step and returns it as `info["success"]` (reward = 1.0 if success).
  Task checks: kitchen_microwave.py:64-79, kitchen_pick_place.py:127-137, kettle_boiling.py:77-105.
- Termination: robosuite never ends the episode (`ignore_done=True`, robosuite base.py:546;
  `done` is always False). The official `MultiStepWrapper` (gr00t/eval/wrappers/multistep_wrapper.py:
  200-241) executes the 16 actions of a chunk one by one and ends the episode once
  `max_episode_steps = get_task_horizon(task)` env steps have run. The client keeps
  `success |= info["success"]` where the info holds the value of the **last env step of each chunk**
  (`dict_take_last_n(self.info, 1)`, multistep_wrapper.py:232; OR at simulation.py:141). It does
  not stop early on success; since the
  success flag is OR-ed, stopping at the first chunk boundary where `_check_success()` is True gives
  the same per-episode result. Max steps per episode = horizon in env steps (20 Hz control).

## Q5. Policy
- Class: `gr00t.model.policy.Gr00tPolicy(model_path, embodiment_tag="new_embodiment",
  modality_config=cfg.modality_config(), modality_transform=cfg.transform(), denoising_steps=4)`
  with `cfg = gr00t.experiment.data_config.DATA_CONFIG_MAP["panda_omron"]`
  (run_eval.py:36-48; defaults :122-133). `PandaOmronDataConfig`: data_config.py:640-683.
  The checkpoint's `experiment_cfg/metadata.json` has the single embodiment `new_embodiment`;
  `config.json`: `GR00T_N1_5`, action_horizon 16, num_inference_timesteps 4, flash attention.
- Transforms (official, inside the policy; never re-implement): VideoCrop(0.95) + resize to
  224x224, min-max state normalisation with quaternion → rotation_6d, min-max / binary action
  un-normalisation, GR00TTransform(max_state_dim=64, max_action_dim=32) (data_config.py:497-537).
- Observation keys (per query; video/state have a time axis of 1, `observation_indices=[0]`):
  - `video.robot0_agentview_left`, `video.robot0_agentview_right`, `video.robot0_eye_in_hand`:
    uint8 RGB **256x256**, already flipped upright by the wrapper (gym_wrapper.py:270-273).
  - `state.end_effector_position_relative` (3), `state.end_effector_rotation_relative` (4, quat),
    `state.gripper_qpos` (2), `state.base_position` (3), `state.base_rotation` (4, quat), float32
    (mapped from `robot0_base_to_eef_pos/quat`, `robot0_gripper_qpos`, `robot0_base_pos/quat`, :50-59).
  - `annotation.human.task_description`: the instruction string.
  The gym env returns exactly these keys/shapes (no time axis) from `reset()`/`step()`.
- Official request layout (MultiStepWrapper + 1-env vector env → server): video `(1, 1, 256, 256, 3)`,
  state `(1, 1, D)`, language `np.array([instruction])`, i.e. add `[None, None]` to each
  video/state array. `Gr00tPolicy.get_action` (policy.py:146) treats states with ndim 3 as batched
  (:233-237) and also accepts unbatched input.
- Action chunk returned: dict `action.end_effector_position (1,16,3)`, `action.end_effector_rotation
  (1,16,3)`, `action.gripper_close (1,16,1)`, `action.base_motion (1,16,4)`,
  `action.control_mode (1,16,1)` (un-normalised, env units). Concatenated in that order it is the
  12-D layout of `robocasa.utils.env_utils.convert_action` (env_utils.py:142-154), which maps one
  row back to the gym action dict. `RoboCasaGymEnv.step` then binarises gripper/control_mode at 0.5
  and splits base_motion into base (3) + torso (1) (`unmap_action`, gym_wrapper.py:121-140).
- Action horizon executed per query: **16** (`--n_action_steps` default 16, run_eval.py:155-160;
  `MultiStepConfig.n_action_steps=16`, simulation.py:64). Query again after 16 env steps.
- Server/client (two envs): in `a2u-policy`, the fork's `scripts/inference_service.py`
  (defaults already `new_embodiment` / `panda_omron`, :57-90; server branch :136-170) builds the same
  `Gr00tPolicy` and serves it with `gr00t.eval.robot.RobotInferenceServer` (ZMQ REP, endpoints
  `get_action`, `get_modality_config`, `ping`, `kill`):
  `conda run --no-capture-output -n a2u-policy python -u third_party/Isaac-GR00T/scripts/inference_service.py
  --server --model-path "$PWD/<checkpoint>" --embodiment-tag new_embodiment --data-config panda_omron
  --denoising-steps 4 --port 5555` (tyro flags use dashes; `-u` so "Server is ready and listening on
  tcp://*:5555" is printed promptly; binds all interfaces).
  In `a2u-sim` the client is `gr00t.eval.service.ExternalRobotInferenceClient(host="localhost",
  port=5555)` (service.py:193-205; needs only torch + pyzmq, both installed there):
  `client.get_action(obs_dict) -> action_dict`. Payloads are `torch.save` pickles of numpy arrays
  (service.py:24-35); both envs use numpy 2.2.5. The client sets no socket timeout: `ping()` blocks
  until the server has loaded the model and answers. Do not call `get_modality_config` from
  `a2u-sim` (it returns gr00t pydantic objects that need the full gr00t deps to unpickle).
- **Verified end to end** (checkpoint-60000, server in `a2u-policy`, this exact request layout from
  `a2u-sim`, 16 actions per query, success checked per chunk): TurnOnMicrowave target seeds 0-5 →
  4/6 successes (reference 70%); ~75 ms per query after the first.

## Q6. Cameras
Cameras in the PandaOmron kitchen model (queried from `env.unwrapped.env.sim.model.camera_names`):
`robot0_agentview_center, robot0_agentview_left, robot0_agentview_right, robot0_frontview,
robot0_robotview, robot0_eye_in_hand`. The policy uses agentview_left/right + eye_in_hand.
The official video is `env.render()` = the `robot0_agentview_left` observation (first policy
camera, gym_wrapper.py:185, 298; 256x256, every 2nd step at 10 fps, simulation.py:44-56).
**`video.camera: robot0_agentview_left` is confirmed.** For 512x512 frames render separately:
`frame = env.unwrapped.env.sim.render(width=512, height=512, camera_name="robot0_agentview_left")[::-1]`.
Raw `sim.render` output is upside down: the vertical flip `[::-1]` is required (verified;
`docs/sanity_frame.png` is the flipped 512x512 frame of TurnOnMicrowave, target, seed 0 —
upright kitchen, microwave above the range, robot arm visible). Rendering at 512 does not change
the 256x256 policy observations (verified pixel-identical). Frames taken from
`obs["video.robot0_agentview_left"]` are already upright and need no flip.

## Notes for the Integration Agent
- Set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (stored in `a2u-sim`; `conda run` picks them up).
- Import `robocasa` before `gym.make` (registers ids). Expect harmless warnings: robosuite
  "No private macro file", mimicgen not installed, gymnasium "obs not within observation space",
  and EGL errors printed at interpreter exit.
- Always run full 16-step chunks; the last chunk is truncated at the horizon (MultiStepWrapper).
- One checkpoint load takes GPU memory once per server; each Eval Agent starts its own server port.
