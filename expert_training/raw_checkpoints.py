"""Find and prune the raw RLinf checkpoints of one expert run (spec rule 5, keep_last_raw).

RLinf saves <run_dir>/checkpoints/global_step_N/actor/ as the resume state (dcp_checkpoint/:
sharded weights, optimizer and scheduler; its .metadata file is written last) followed by
model_state_dict/full_weights.pt, the only file the exporter needs, and never deletes anything.
`prune` keeps the newest keep_last_raw checkpoints (from the run's resolved_config.yaml) whole, so
--resume works and the newest, possibly still-being-written checkpoint is never touched; from
every older one whose full_weights.pt is complete it deletes everything in actor/ except
model_state_dict/. torch.save's zip format writes the archive's directory last, so a
full_weights.pt is complete when it reads as a whole zip file. `resume-dir` prints the newest
checkpoint that has both a complete resume state and complete full weights.

Usage: python expert_training/raw_checkpoints.py prune <run_dir>
       python expert_training/raw_checkpoints.py resume-dir <run_dir>
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """All global_step_N directories of the run, oldest first."""
    found = []
    for path in (run_dir / "checkpoints").glob("global_step_*"):
        step = path.name.removeprefix("global_step_")
        if path.is_dir() and step.isdigit():
            found.append((int(step), path))
    return sorted(found)


def full_weights(checkpoint: Path) -> Path:
    return checkpoint / "actor" / "model_state_dict" / "full_weights.pt"


def weights_complete(checkpoint: Path) -> bool:
    path = full_weights(checkpoint)
    return path.is_file() and zipfile.is_zipfile(path)


def resumable(checkpoint: Path) -> bool:
    return (checkpoint / "actor" / "dcp_checkpoint" / ".metadata").is_file() and weights_complete(checkpoint)


def keep_last_raw(run_dir: Path) -> int:
    config = run_dir / "resolved_config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"{config} not found; is {run_dir} a run directory made by train_expert.sh?")
    keep = yaml.safe_load(config.read_text())["a2u_expert"]["keep_last_raw"]
    if not isinstance(keep, int) or keep < 1:
        raise ValueError(f"a2u_expert.keep_last_raw in {config} must be an int >= 1, got {keep!r}")
    return keep


def _size(path: Path) -> int:
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(file.stat().st_size for file in files if file.is_file())


def prune(run_dir: Path) -> None:
    """Strip the resume state from every checkpoint older than the newest keep_last_raw."""
    keep = keep_last_raw(run_dir)
    all_checkpoints = checkpoints(run_dir)
    for step, checkpoint in all_checkpoints[:-keep]:
        actor = checkpoint / "actor"
        extra = sorted(path for path in actor.iterdir() if path.name != "model_state_dict") if actor.is_dir() else []
        if not extra:
            continue
        if not weights_complete(checkpoint):
            print(f"WARNING: step {step}: {full_weights(checkpoint)} is missing or incomplete; "
                  "keeping its resume state", file=sys.stderr)
            continue
        freed = sum(_size(path) for path in extra)
        for path in extra:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f"[prune] step {step}: removed {[path.name for path in extra]} ({freed / 1e9:.1f} GB); "
              f"kept {full_weights(checkpoint)}", flush=True)
    kept = [step for step, _ in all_checkpoints[-keep:]]
    print(f"[prune] {run_dir.name}: {len(all_checkpoints)} raw checkpoints, newest {kept} kept whole", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prune", "resume-dir"])
    parser.add_argument("run_dir", type=Path, help="expert_training/runs/<Task>/<run_id>")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        if args.command == "prune":
            prune(run_dir)
            return
        candidates = [checkpoint for _, checkpoint in checkpoints(run_dir) if resumable(checkpoint)]
        if not candidates:
            raise FileNotFoundError(f"no checkpoint in {run_dir / 'checkpoints'} has a complete resume state "
                                    "(actor/dcp_checkpoint/.metadata and a complete full_weights.pt)")
        print(candidates[-1])
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
