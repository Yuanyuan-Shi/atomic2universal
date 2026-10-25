"""Run one seeded closed-loop episode of a RoboCasa365 task and record it as an MP4 (or not).

Everything that defines the benchmark comes from the official code: the env is the RoboCasa365
gym env `robocasa/<task>` built for (split, seed), the instruction is the env's episode metadata,
success is the env's own `info["success"]`, and the episode length is the task's registered
horizon. The loop mirrors the official GR00T eval (`MultiStepWrapper` + `SimulationInferenceClient`):
each policy query returns a chunk of `action_horizon` actions that are all executed, success is
read from the last env step of each chunk, and the episode ends on success or at the horizon
(the last chunk is cut at the horizon). One frame is written per env step, after the reset frame;
without a video path no frame is rendered for video (policy camera observations are unchanged).
An optional `on_query(observation, step)` callback sees the raw observation and the env step index
just before each policy query (distillation data collection); without it nothing changes.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401  (registers the robocasa/<task> gym ids)
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action

from evaluation.config import EvalSpec, TaskSpec
from evaluation.policy import Policy
from evaluation.video import Mp4Writer

OnQuery = Callable[[dict[str, Any], int], None]


def render_frame(env: gym.Env, eval_spec: EvalSpec) -> np.ndarray:
    """Render the video camera upright; raw MuJoCo frames are stored bottom row first."""
    frame = env.unwrapped.env.sim.render(
        width=eval_spec.video_width, height=eval_spec.video_height, camera_name=eval_spec.video_camera
    )
    return np.ascontiguousarray(frame[::-1])


def _check_camera(env: gym.Env, camera: str) -> None:
    camera_names = list(env.unwrapped.env.sim.model.camera_names)
    if camera not in camera_names:
        raise ValueError(f"video camera {camera!r} is not in the scene; pick one of {camera_names} in the eval config")


def _run_chunks(env: gym.Env, observation: dict[str, Any], instruction: str, horizon: int,
                policy: Policy, eval_spec: EvalSpec, writer: Mp4Writer | None,
                on_query: OnQuery | None = None) -> tuple[bool, int]:
    """Execute action chunks until success at a chunk boundary or the horizon; return (success, steps)."""
    num_steps = 0
    success = False
    while num_steps < horizon and not success:
        if on_query is not None:
            on_query(observation, num_steps)
        chunk = policy.act(observation, instruction)
        for action_row in chunk[: horizon - num_steps]:
            observation, _, _, _, info = env.step(convert_action(action_row))
            if writer is not None:
                writer.append(render_frame(env, eval_spec))
            num_steps += 1
        success = bool(info["success"])
    return success, num_steps


def run_episode(task: TaskSpec, seed: int, policy: Policy, eval_spec: EvalSpec, video_path: Path | None,
                on_query: OnQuery | None = None) -> dict:
    """Run one episode and write its video to `video_path` (None: no video); `on_query` is passed to the loop.

    Returns the episode fields of an episodes.jsonl line: seed, success, num_steps, horizon,
    wall_time_s and instruction. The video is written to a temporary file and moved into
    place only when the episode finishes, so an interrupted episode leaves no video behind.
    """
    start_time = time.monotonic()
    horizon = get_task_horizon(task.name)
    partial_path = None if video_path is None else video_path.with_suffix(".partial.mp4")
    env = gym.make(f"robocasa/{task.name}", split=task.split, seed=seed, enable_render=True)
    try:
        observation, _ = env.reset()
        _check_camera(env, eval_spec.video_camera)
        instruction = env.unwrapped.env.get_ep_meta()["lang"]
        policy.reset()
        if partial_path is None:
            success, num_steps = _run_chunks(env, observation, instruction, horizon, policy, eval_spec, None,
                                             on_query)
        else:
            with Mp4Writer(partial_path, fps=eval_spec.video_fps) as writer:
                writer.append(render_frame(env, eval_spec))
                success, num_steps = _run_chunks(env, observation, instruction, horizon, policy, eval_spec, writer,
                                                 on_query)
            partial_path.replace(video_path)
    finally:
        env.close()
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
    return {
        "seed": seed,
        "success": success,
        "num_steps": num_steps,
        "horizon": horizon,
        "wall_time_s": round(time.monotonic() - start_time, 1),
        "instruction": instruction,
    }
