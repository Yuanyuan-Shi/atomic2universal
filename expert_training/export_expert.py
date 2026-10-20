"""Export one raw RL checkpoint of an expert run as an eval-ready GR00T model folder.

Picks <run_dir>/checkpoints/global_step_<N> (--step latest: the newest one whose full_weights.pt
is complete; --step best: checkpoints/best/, saved during training, exported as its step N), runs StructRL's converter structrl/convert_rl_fullweights_to_hf.py in the RL env on
its actor/model_state_dict/full_weights.pt with the run's own starting checkpoint (our
atomic_seen checkpoint-60000, recorded in the run's resolved_config.yaml) as the base, into
checkpoints/experts/<Task>/step_<N>.partial/. Then expert_training/verify_expert.py checks that
folder in the policy env (no value_head tensors, all bf16, loads with the eval server's
Gr00tPolicy setup without weight mismatches, one valid dummy action chunk); only then are
export_info.json written and the folder renamed to step_<N>/. Any failure stops with the reason;
a finished step_<N>/ is never overwritten. Needs only the standard library and pyyaml, so it
runs from the project root in any project env (e.g. a2u-sim).

Usage: python expert_training/export_expert.py --task <Task> --run-dir <run_dir> --step <N|latest|best>
    [--env-prefix DIR]   (RL env for the converter; default a2u-rl-blackwell if present, else a2u-rl)
Environment override: POLICY_ENV (default a2u-policy), as in evaluation/serve_policy.sh.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from raw_checkpoints import checkpoints, full_weights, weights_complete

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERTS_ROOT = PROJECT_ROOT / "checkpoints" / "experts"
CONVERTER = PROJECT_ROOT / "third_party" / "StructRL" / "structrl" / "convert_rl_fullweights_to_hf.py"
RL_ENV_PREFIXES = [PROJECT_ROOT / "third_party" / "conda_envs" / name for name in ("a2u-rl-blackwell", "a2u-rl")]
POLICY_ENV = os.environ.get("POLICY_ENV", "a2u-policy")


class ExportError(RuntimeError):
    """The run, the step or one of the export stages is not usable; the message says why."""


def default_env_prefix() -> Path:
    for prefix in RL_ENV_PREFIXES:
        if (prefix / "bin" / "python").is_file():
            return prefix
    raise ExportError(f"no RL env found at {[str(p) for p in RL_ENV_PREFIXES]}; run setup/setup_rl.sh or pass --env-prefix")


def select_step(run_dir: Path, step: str) -> tuple[int, Path]:
    """The requested raw checkpoint, which must have a complete full_weights.pt."""
    if step == "best":  # checkpoints/best/ from startup_hooks/a2u_best_checkpoint.py; same weights as step N
        best = run_dir / "checkpoints" / "best"
        if not (best / "best.json").is_file() or not weights_complete(best):
            raise ExportError(f"--step best: {best} has no best.json or no complete full_weights.pt")
        return json.loads((best / "best.json").read_text())["step"], best
    complete = {number: path for number, path in checkpoints(run_dir) if weights_complete(path)}
    if not complete:
        raise ExportError(f"no checkpoint with a complete full_weights.pt under {run_dir / 'checkpoints'}")
    if step == "latest":
        number = max(complete)
    elif step.isdigit() and int(step) in complete:
        number = int(step)
    else:
        raise ExportError(f"--step {step}: no complete checkpoint with that step in {run_dir}; exportable: {sorted(complete)}")
    return number, complete[number]


def run_in_env(command: list[str], what: str) -> None:
    """Run a conda-run command with PYTHONPATH removed (each env imports only its own packages)."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        raise ExportError("conda not found; activate conda or set CONDA_EXE")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    print(f"[export] {what}: {' '.join(command)}", flush=True)
    if subprocess.run([conda, "run", "--no-capture-output", *command], cwd=PROJECT_ROOT, env=env).returncode != 0:
        raise ExportError(f"{what} failed; see the output above")


def export(task: str, run_dir: Path, step: str, env_prefix: Path) -> Path:
    config = run_dir / "resolved_config.yaml"
    if not config.is_file():
        raise ExportError(f"{config} not found; --run-dir must be a run made by train_expert.sh or make_step0.py")
    meta = (yaml.safe_load(config.read_text()) or {}).get("a2u_expert")
    if not isinstance(meta, dict):
        raise ExportError(f"{config} has no a2u_expert block; it was not built by expert_training/build_config.py")
    if meta["task"] != task:
        raise ExportError(f"{run_dir} trained {meta['task']}, not --task {task}")
    base = Path(meta["checkpoint"])
    if not (base / "config.json").is_file():
        raise ExportError(f"base checkpoint {base} (a2u_expert.checkpoint) has no config.json")
    number, checkpoint = select_step(run_dir, step)
    out = EXPERTS_ROOT / task / f"step_{number}"
    if out.exists():
        raise ExportError(f"{out} already exists; delete it to export step {number} again")
    partial = out.with_name(f"{out.name}.partial")
    if partial.exists():
        print(f"[export] removing leftover {partial}", flush=True)
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    weights = full_weights(checkpoint)
    run_in_env(["-p", str(env_prefix), "python", str(CONVERTER), str(weights), str(base), str(partial)],
               "convert (StructRL converter, RL env)")
    run_in_env(["-n", POLICY_ENV, "python", "-m", "expert_training.verify_expert", "--checkpoint", str(partial)],
               f"verify (eval loader, {POLICY_ENV})")
    info = {"task": task, "step": number, "run_dir": str(run_dir), "full_weights": str(weights),
            "base_checkpoint": str(base), "converter": str(CONVERTER), "exported": f"{datetime.now():%Y-%m-%d %H:%M}"}
    (partial / "export_info.json").write_text(json.dumps(info, indent=2) + "\n")
    partial.rename(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="task the run trained, e.g. PickPlaceCounterToCabinet")
    parser.add_argument("--run-dir", type=Path, required=True, help="expert_training/runs/<Task>/<run_id>")
    parser.add_argument("--step", required=True, help="global step N of a raw checkpoint, 'latest', or 'best' (checkpoints/best)")
    parser.add_argument("--env-prefix", type=Path, help="RL conda env prefix for the converter")
    args = parser.parse_args()
    try:
        env_prefix = args.env_prefix.resolve() if args.env_prefix else default_env_prefix()
        if not (env_prefix / "bin" / "python").is_file():
            raise ExportError(f"--env-prefix {env_prefix} is not a conda env (no bin/python)")
        print(f"Exported {export(args.task, args.run_dir.resolve(), args.step, env_prefix)}")
    except ExportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
