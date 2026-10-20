"""Startup hooks for RL training processes, without editing StructRL or RLinf.

CPython imports a top-level `sitecustomize` module at interpreter start if one is on sys.path;
train_expert.sh puts this directory first on PYTHONPATH, so this runs in every training process (driver,
Ray workers, env subprocesses), as in StructRL's own ablations (ablations/beta_sweep/sitecustomize.py).
It only registers an import hook that patches a module right after it first executes:
  robocasa.utils.gym_utils.gymnasium_basic -> a2u_scene_seeds.patch: explicit, recorded scene seeds per reset
  rlinf.runners.embodied_runner            -> a2u_run_metrics.patch: W&B run metadata and extra metrics
  rlinf.workers.actor.fsdp_actor_worker    -> a2u_weight_sync_fix.patch: correct actor -> rollout weight sync
Run this file directly to check that the hook is active in an interpreter started with the training
PYTHONPATH (exit 1 if not).
"""

import argparse
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType

LOG_PREFIX = "[a2u startup hooks]"
PATCHES = {"robocasa.utils.gym_utils.gymnasium_basic": "a2u_scene_seeds",
           "rlinf.runners.embodied_runner": "a2u_run_metrics",
           "rlinf.workers.actor.fsdp_actor_worker": "a2u_weight_sync_fix"}  # target module -> module whose patch() runs


class PatchAfterImport(importlib.abc.MetaPathFinder):
    """Meta-path finder that calls `<PATCHES[name]>.patch(module)` once, right after a target module executes."""

    def __init__(self):
        self.pending = set(PATCHES)

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.pending:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None  # the normal import machinery reports the missing module
        execute = spec.loader.exec_module

        def exec_module(module: ModuleType) -> None:
            execute(module)
            self.pending.discard(fullname)  # a reload must not patch twice
            importlib.import_module(PATCHES[fullname]).patch(module)

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
