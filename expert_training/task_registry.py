"""Look a task up in the eval stack's RoboCasa365 registry: task set membership, atomic or not, horizon.

All our evaluation runs in the a2u-sim env (robocasa a9e152e in third_party/robocasa), whose
registry differs from StructRL's robocasa 8f3c96e (e.g. horizon 750 vs 500 for
PickPlaceCounterToCabinet), so expert training takes task facts from a2u-sim: the query runs
there through `conda run`, with PYTHONPATH removed so that env imports only its own robocasa.

Usage: python expert_training/task_registry.py --task <Task> [--task-set atomic_seen]   (prints JSON)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIM_ENV = "a2u-sim"
SIM_ROBOCASA = Path(__file__).resolve().parent.parent / "third_party" / "robocasa"
QUERY = """
import json, sys
import robocasa
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
task, task_set = sys.argv[1:3]
atomic = task in ATOMIC_TASK_DATASETS
print(json.dumps({"robocasa": robocasa.__file__, "task_set": list(TASK_SET_REGISTRY[task_set]), "atomic": atomic,
                  "horizon": get_task_horizon(task) if atomic else None}))
"""


class RegistryError(RuntimeError):
    """The registry could not be queried, or the task is not an atomic task of the task set."""


def lookup(task: str, task_set: str) -> dict:
    """Return {robocasa, task_set, atomic, horizon} for an atomic task of `task_set`, else raise."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        raise RegistryError("conda not found; activate conda or set CONDA_EXE")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([conda, "run", "-n", SIM_ENV, "python", "-c", QUERY, task, task_set],
                            capture_output=True, text=True, env=env)
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        raise RegistryError(f"registry query in conda env {SIM_ENV} failed (exit {result.returncode}):\n"
                            f"{result.stderr[-2000:]}\nCheck `conda run -n {SIM_ENV} python -c 'import robocasa'` "
                            "(built by setup/setup_sim.sh).")
    registry = json.loads(lines[-1])
    where = registry["robocasa"]
    if not Path(where).resolve().is_relative_to(SIM_ROBOCASA):
        raise RegistryError(f"{SIM_ENV} imported robocasa from {where}, not from {SIM_ROBOCASA}")
    if task not in registry["task_set"]:
        raise RegistryError(f"task {task!r} is not in the {task_set} task set of the eval registry ({where}); "
                            f"choose one of {registry['task_set']}")
    if not registry["atomic"]:
        raise RegistryError(f"task {task!r} is not an atomic task in {where}")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="e.g. PickPlaceCounterToCabinet")
    parser.add_argument("--task-set", default="atomic_seen", help="key of TASK_SET_REGISTRY (default: atomic_seen)")
    args = parser.parse_args()
    try:
        print(json.dumps(lookup(args.task, args.task_set), indent=2))
    except RegistryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
