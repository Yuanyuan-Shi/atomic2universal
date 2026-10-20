"""Explicit, recorded scene seeds for every RL training reset (patched in by sitecustomize.py).

Wraps robocasa's `RoboCasaEnv.reset` (behind RLinf's RoboCasa365 gym ids) in the env processes:
  - RLinf's first reset of an env is `reset(seed=env.train.seed + global_env_id)`; the hook reads the
    global env id from it;
  - reset number k of that env (counted from the first scene of this job) gets the seed
        env.train.seed + global_env_id + total_num_envs * (k0 + k),
    where k0 = rollout_epoch * <global step of runner.resume_dir> (0 for a fresh run). The seed seeds the
    robosuite scene RNG (`env.rng`) and numpy's global RNG. With RLinf's one reset per rollout epoch
    (auto_reset off), iteration n uses exactly the seeds of resets k0 + k in [n*E, (n+1)*E), E = rollout_epoch:
    num_envs * E distinct seeds, all >= env.train.seed (100000), the same after a resume, never 0-99 or 1000-1019;
  - each seeded reset appends "<reset index>\t<seed>\t<global env id>" to <run_dir>/seeds/<host>-<pid>.tsv,
    read by a2u_run_metrics.py to log the seeds of every iteration;
  - a second explicit seed raises, since honouring it would replay scenes and ignoring it would be silent.
Settings come from <A2U_RUN_DIR>/resolved_config.yaml (train_expert.sh exports A2U_RUN_DIR). Without
A2U_RUN_DIR the hook leaves reset unchanged (stock behaviour), e.g. in make_step0.py.
"""

import os
import re
import socket
import sys
from pathlib import Path
from types import ModuleType

LOG_PREFIX = "[a2u seed hook]"
STATE = "_a2u_seed_state"  # (global env id, next reset index) on each env instance


def seed_settings(run_dir: Path) -> dict:
    """env.train.seed, total_num_envs, rollout_epoch and the first reset index (resume) of a run."""
    import yaml

    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    epochs = cfg["algorithm"]["rollout_epoch"]
    resume_dir = cfg["runner"].get("resume_dir")
    step = int(re.fullmatch(r".*global_step_(\d+)/?", str(resume_dir)).group(1)) if resume_dir else 0
    return {"offset": cfg["env"]["train"]["seed"], "num_envs": cfg["env"]["train"]["total_num_envs"],
            "epochs": epochs, "first_reset": epochs * step}


def seed_for(settings: dict, env_id: int, reset_index: int) -> int:
    return settings["offset"] + env_id + settings["num_envs"] * reset_index


def patch(module: ModuleType) -> None:
    """Wrap `module.RoboCasaEnv.reset` with explicit per-reset seeds (no-op without A2U_RUN_DIR)."""
    import numpy as np

    env_class = getattr(module, "RoboCasaEnv", None)
    if env_class is None or not callable(getattr(env_class, "reset", None)):
        raise RuntimeError(f"{LOG_PREFIX} {module.__name__} has no RoboCasaEnv.reset; robocasa changed, update the hook")
    if not os.environ.get("A2U_RUN_DIR"):
        return
    run_dir = Path(os.environ["A2U_RUN_DIR"])
    settings = seed_settings(run_dir)
    record = run_dir / "seeds" / f"{socket.gethostname()}-{os.getpid()}.tsv"
    original_reset = env_class.reset

    def reset(self, seed=None, options=None):
        state = getattr(self, STATE, None)
        if seed is not None and state is not None:
            raise RuntimeError(f"{LOG_PREFIX} env {state[0]} already seeded; refusing reset(seed={seed})")
        if seed is None and state is None:  # never seeded by RLinf: stock behaviour
            return original_reset(self, seed=None, options=options)
        if state is None:
            env_id = int(seed) - settings["offset"]
            if not 0 <= env_id < settings["num_envs"]:
                raise RuntimeError(f"{LOG_PREFIX} first seed {seed} is not env.train.seed + a global env id")
            state = (env_id, settings["first_reset"])
        env_id, index = state
        scene_seed = seed_for(settings, env_id, index)
        setattr(self, STATE, (env_id, index + 1))
        self.env.rng = np.random.default_rng(scene_seed)
        result = original_reset(self, seed=scene_seed, options=options)
        record.parent.mkdir(exist_ok=True)
        with record.open("a") as file:
            file.write(f"{index}\t{scene_seed}\t{env_id}\n")
        meta = self.env.get_ep_meta()
        sys.stderr.write(f"{LOG_PREFIX} env {env_id} reset {index} seed {scene_seed}: layout {meta.get('layout_id')} "
                         f"style {meta.get('style_id')} | {meta.get('lang')}\n")
        return result

    env_class.reset = reset
