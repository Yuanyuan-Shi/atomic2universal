"""Startup hook that gives every RL training env its own seeded, reproducible stream of scenes.

CPython imports a top-level `sitecustomize` module at interpreter start if one is on sys.path;
train_expert.sh puts this directory first on PYTHONPATH, so the hook runs in every training
process (driver, Ray workers, env subprocesses) without editing StructRL or RLinf (StructRL's own
ablations use the same mechanism, ablations/beta_sweep/sitecustomize.py). It only registers an
import hook; once `robocasa.utils.gym_utils.gymnasium_basic` is imported, its `RoboCasaEnv.reset`
(the reset behind RLinf's RoboCasa365 gym ids) is wrapped so that:
  - the first reset with a seed (RLinf's `env.reset(seed=env.train.seed + global_env_id)` in
    `RoboCasa365Env.get_env_fns`) seeds the robosuite scene RNG `env.rng` with that seed;
  - every later reset (RLinf passes no seed: each rollout epoch and auto-reset) continues that
    stream, and seeds numpy's global RNG from it instead of from OS entropy;
  - a second explicit seed raises, since honoring it would replay scenes and ignoring it would
    be silent.
Stock robocasa seeds only numpy's global RNG, which scene sampling does not use, so its training
scenes are neither seeded nor reproducible (docs/rl_recon.md Q4). Run this file directly to check
that the hook is active in an interpreter started with the training PYTHONPATH (exit 1 if not).
"""

import argparse
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType

TARGET_MODULE = "robocasa.utils.gym_utils.gymnasium_basic"
SEED_ATTRIBUTE = "_a2u_scene_seed"
LOG_PREFIX = "[a2u seed hook]"


def patch_reset(module: ModuleType) -> None:
    """Wrap `module.RoboCasaEnv.reset` with the seed-once, continue-the-stream behaviour."""
    import numpy as np

    env_class = getattr(module, "RoboCasaEnv", None)
    if env_class is None or not callable(getattr(env_class, "reset", None)):
        raise RuntimeError(f"{LOG_PREFIX} {TARGET_MODULE} has no RoboCasaEnv.reset; robocasa changed, update the hook")
    original_reset = env_class.reset

    def log_scene(self) -> None:
        """One line per seeded reset, so a run's log shows which scenes each env stream produced."""
        meta = self.env.get_ep_meta()
        sys.stderr.write(f"{LOG_PREFIX} pid {os.getpid()} stream {getattr(self, SEED_ATTRIBUTE)}: layout "
                         f"{meta.get('layout_id')} style {meta.get('style_id')} | {meta.get('lang')}\n")

    def reset(self, seed=None, options=None):
        scene_seed = getattr(self, SEED_ATTRIBUTE, None)
        if seed is not None:
            if scene_seed is not None:
                raise RuntimeError(f"{LOG_PREFIX} env already seeded with {scene_seed}; refusing reset(seed={seed})")
            setattr(self, SEED_ATTRIBUTE, int(seed))
            self.env.rng = np.random.default_rng(int(seed))
            sys.stderr.write(f"{LOG_PREFIX} pid {os.getpid()}: scene RNG seeded with {int(seed)}\n")
            result = original_reset(self, seed=seed, options=options)
        elif scene_seed is None:  # never seeded: stock behaviour
            return original_reset(self, seed=None, options=options)
        else:
            result = original_reset(self, seed=int(self.env.rng.integers(2**32)), options=options)
        log_scene(self)
        return result

    env_class.reset = reset


class PatchAfterImport(importlib.abc.MetaPathFinder):
    """Meta-path finder that runs `patch_reset` once, right after TARGET_MODULE is executed."""

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None  # the normal import machinery reports the missing module
        execute = spec.loader.exec_module

        def exec_module(module: ModuleType) -> None:
            execute(module)
            patch_reset(module)
            sys.meta_path.remove(self)  # a reload must not wrap reset twice

        spec.loader.exec_module = exec_module
        return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    hook = sys.modules.get("sitecustomize")
    finder_class = getattr(hook, "PatchAfterImport", None)
    registered = finder_class is not None and any(isinstance(finder, finder_class) for finder in sys.meta_path)
    print(f"{LOG_PREFIX} sitecustomize: {getattr(hook, '__file__', None)}; import hook registered: {registered}")
    if not registered:
        print(f"{LOG_PREFIX} not active: put {os.path.dirname(os.path.abspath(__file__))} first on PYTHONPATH",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
else:
    sys.meta_path.insert(0, PatchAfterImport())
