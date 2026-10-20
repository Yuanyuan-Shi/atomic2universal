"""Attach evaluation results (baseline_eval summary.json files) to an expert's W&B training run.

Resumes the training run (id = run directory name, project from its resolved_config.yaml, entity from
WANDB_ENTITY or .env) and logs, against the training axis `iteration` (= checkpoint step):
  --select  .../step_<N>/.../summary.json ...   eval_select/success, eval_select/ci_low, eval_select/ci_high
  --final-expert / --final-baseline [/ --final-step0] summary.json
            eval_final/{expert,baseline,step0}_success and _ci_low/_ci_high (run summary), and the comparison
            table printed here, uploaded as the W&B table eval_final/comparison
Only numbers and the table are uploaded, never checkpoints or videos. Run in the RL env (has wandb).

Usage: python expert_training/wandb_log_eval.py --run-dir expert_training/runs/<Task>/<run_id>
           [--select <summary.json>...] [--final-expert S --final-baseline S [--final-step0 S]] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["model", "checkpoint", "seeds", "episodes", "successes", "success_rate", "ci_low", "ci_high"]


def load_dotenv(path: Path) -> None:
    """KEY=VALUE or KEY = VALUE lines (the project's .env format); never overrides the environment."""
    for line in path.read_text().splitlines() if path.is_file() else []:
        match = re.fullmatch(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*", line)
        if match and not line.lstrip().startswith("#"):
            os.environ.setdefault(match.group(1), match.group(2).strip("'\""))


def summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    if len(data.get("ci95", [])) != 2 or "success_rate" not in data:
        raise ValueError(f"{path} is not a baseline_eval summary.json (success_rate, ci95)")
    return data


def selection_step(path: Path) -> int:
    match = re.search(r"/step_(\d+)/", str(path.resolve()))
    if not match:
        raise ValueError(f"{path}: no step_<N> directory in the path, so the checkpoint step is unknown")
    return int(match.group(1))


def table_rows(finals: dict[str, dict]) -> list[list]:
    rows = []
    for model, data in finals.items():
        start = data.get("seed_start", 0)  # Stage 1 summaries predate the field; they started at seed 0
        seeds = f"{start}-{start + data['num_episodes'] - 1}"
        rows.append([model, data["checkpoint"], seeds, data["num_episodes"], data["num_success"],
                     data["success_rate"], *data["ci95"]])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="expert_training/runs/<Task>/<run_id>")
    parser.add_argument("--select", type=Path, nargs="*", default=[], help="selection summary.json files")
    for name in ("expert", "baseline", "step0"):
        parser.add_argument(f"--final-{name}", type=Path, help=f"final-test summary.json of the {name}")
    parser.add_argument("--dry-run", action="store_true", help="print what would be logged, contact nothing")
    args = parser.parse_args()
    try:
        config = yaml.safe_load((args.run_dir / "resolved_config.yaml").read_text())
        selection = sorted((selection_step(path), summary(path)) for path in args.select)
        finals = {name: summary(path) for name in ("expert", "baseline", "step0")
                  if (path := getattr(args, f"final_{name}")) is not None}
        if finals and not {"expert", "baseline"} <= finals.keys():
            raise ValueError("--final-expert and --final-baseline go together")
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    for step, data in selection:
        print(f"select step {step}: {data['num_success']}/{data['num_episodes']} = {data['success_rate']:.3f} "
              f"CI [{data['ci95'][0]:.3f}, {data['ci95'][1]:.3f}] (seeds from {data['seed_start']})")
    rows = table_rows(finals)
    if rows:
        print(" | ".join(COLUMNS))
        for row in rows:
            print(" | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row))
    if args.dry_run:
        return
    import wandb

    load_dotenv(PROJECT_ROOT / ".env")
    run = wandb.init(project=config["runner"]["logger"]["project_name"], id=args.run_dir.resolve().name,
                     resume="must", entity=os.environ.get("WANDB_ENTITY") or None)
    run.define_metric("iteration")
    run.define_metric("eval_select/*", step_metric="iteration")
    for step, data in selection:
        run.log({"iteration": step, "eval_select/success": data["success_rate"],
                 "eval_select/ci_low": data["ci95"][0], "eval_select/ci_high": data["ci95"][1]})
    for name, data in finals.items():
        run.summary.update({f"eval_final/{name}_success": data["success_rate"],
                            f"eval_final/{name}_ci_low": data["ci95"][0], f"eval_final/{name}_ci_high": data["ci95"][1]})
    if rows:
        run.log({"eval_final/comparison": wandb.Table(columns=COLUMNS, data=rows)})
    print(f"Logged to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
