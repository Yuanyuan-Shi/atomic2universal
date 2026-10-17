"""Run the Stage 1 baseline: seeded closed-loop episodes with video for each role of a pairing.

For each requested role (default: every role of the config, in config order) it runs episodes
0..N-1 (seed = seed_start + episode) of the role's task with the GR00T N1.5 checkpoint, appends
one JSON line per finished episode to
<results-root>/<pairing>/<role>/episodes.jsonl (flushed and fsynced), and writes summary.json
(success rate with a Wilson 95% CI) once all N episodes exist. Recorded episodes are skipped, so
re-running resumes; --overwrite starts the requested roles fresh. The policy runs behind the
official GR00T server: --launch-server starts one on --port for this run and always stops it;
without it, a server started with evaluation/serve_policy.sh must already listen on --port.

Usage (from the project root, in the a2u-sim env):
    python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer [--roles i1 i2 ...]
        [--num-episodes 10] [--results-root baseline_eval/results] [--overwrite]
        [--port 5555] [--launch-server] [--eval-config configs/eval/baseline.yaml]
        [--checkpoint DIR] [--seed-start N]
"""

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from evaluation.config import DEFAULT_EVAL_CONFIG, PROJECT_ROOT, ConfigError, EvalSpec, TaskSpec  # noqa: E402
from evaluation.config import load_eval_config, load_pairing  # noqa: E402
from evaluation.policy import Policy, PolicyError, load_policy  # noqa: E402
from evaluation.policy_server import PolicyServer, PolicyServerError  # noqa: E402
from evaluation.rollout import run_episode  # noqa: E402
from evaluation.stats import wilson_interval  # noqa: E402

EPISODE_FIELDS = ("episode", "seed", "success", "num_steps", "horizon", "wall_time_s", "instruction", "video")


class ResultsError(RuntimeError):
    """Recorded results are inconsistent with the requested run."""


def read_episodes(role_dir: Path, pairing_dir: Path, eval_spec: EvalSpec) -> dict[int, dict]:
    """Return the recorded episodes of one role by index, after checking each line."""
    path = role_dir / "episodes.jsonl"
    if not path.exists():
        return {}
    episodes: dict[int, dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path} line {line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResultsError(f"{where} is not valid JSON ({error}); delete it (interrupted write?) and re-run") from error
        if not isinstance(record, dict) or sorted(record) != sorted(EPISODE_FIELDS):
            raise ResultsError(f"{where} must have exactly the fields {list(EPISODE_FIELDS)}; re-run with --overwrite")
        episode = record["episode"]
        if episode in episodes:
            raise ResultsError(f"{where} records episode {episode} a second time; delete one line or use --overwrite")
        if record["seed"] != eval_spec.seed_start + episode:
            raise ResultsError(f"{where}: seed {record['seed']} does not match seed_start + episode; use --overwrite")
        if not (pairing_dir / record["video"]).is_file():
            raise ResultsError(f"{where}: video {pairing_dir / record['video']} is missing; use --overwrite")
        episodes[episode] = record
    return episodes


def clear_role(role_dir: Path) -> None:
    """Delete one role's recorded episodes, summary and videos (for --overwrite)."""
    for path in [role_dir / "episodes.jsonl", role_dir / "summary.json", *role_dir.glob("videos/*.mp4")]:
        path.unlink(missing_ok=True)


def append_episode(path: Path, record: dict) -> None:
    """Append one JSON line and force it to disk, so a crash never loses a finished episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_summary(pairing: str, role: str, task: TaskSpec, eval_spec: EvalSpec, episodes: list[dict], role_dir: Path) -> None:
    """Write summary.json for exactly the given (complete) list of episodes."""
    num_success = sum(1 for episode in episodes if episode["success"])
    low, high = wilson_interval(num_success, len(episodes))
    summary = {
        "pairing": pairing, "role": role, "task": task.name, "mode": task.mode, "split": task.split,
        "checkpoint": eval_spec.checkpoint, "seed_start": eval_spec.seed_start,
        "num_episodes": len(episodes), "num_success": num_success,
        "success_rate": num_success / len(episodes), "ci95": [round(low, 3), round(high, 3)],
        "reference_sr": task.reference_sr, "reference_note": task.reference_note,
    }
    temporary_path = role_dir / "summary.json.tmp"
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n")
    temporary_path.replace(role_dir / "summary.json")
    print(f"[{role} {task.name}] summary: {num_success}/{len(episodes)} successes, "
          f"95% CI [{low:.3f}, {high:.3f}] -> {role_dir / 'summary.json'}", flush=True)


def run_role(pairing_dir: Path, role: str, task: TaskSpec, pending: list[int], policy: Policy | None,
             eval_spec: EvalSpec) -> None:
    """Run the pending episodes of one role, recording each as soon as it finishes."""
    for episode in pending:
        seed = eval_spec.seed_start + episode
        video = f"{role}/videos/ep{episode:02d}_seed{seed}.mp4"
        result = run_episode(task, seed, policy, eval_spec, pairing_dir / video)
        append_episode(pairing_dir / role / "episodes.jsonl", {"episode": episode, **result, "video": video})
        prefix = f"[{role} {task.name}] episode {episode:02d} seed {seed}"
        outcome = "SUCCESS" if result["success"] else "FAIL"
        print(f'{prefix}: instruction "{result["instruction"]}"')
        print(f"{prefix}: {outcome} in {result['num_steps']} steps ({result['wall_time_s']:.1f} s)", flush=True)


def apply_overrides(eval_spec: EvalSpec, checkpoint: str | None, seed_start: int | None) -> EvalSpec:
    """Replace the eval config's checkpoint (relative paths are taken from the project root, as in the
    config) and seed_start with the --checkpoint and --seed-start values, if given."""
    if checkpoint is not None:
        path = Path(checkpoint) if Path(checkpoint).is_absolute() else PROJECT_ROOT / checkpoint
        if not (path / "config.json").is_file():
            raise ConfigError(f"--checkpoint {path} is missing or has no config.json; export the expert first "
                              "(expert_training/export_expert.py) or fix the path")
        eval_spec = dataclasses.replace(eval_spec, checkpoint=checkpoint, checkpoint_path=path)
    if seed_start is not None:
        if seed_start < 0:
            raise ConfigError(f"--seed-start must be >= 0, got {seed_start}")
        eval_spec = dataclasses.replace(eval_spec, seed_start=seed_start)
    return eval_spec


def run(args: argparse.Namespace) -> None:
    """Plan the pending episodes per role, run them against one policy server, and write summaries."""
    eval_spec = apply_overrides(load_eval_config(args.eval_config), args.checkpoint, args.seed_start)
    tasks = load_pairing(args.pairing, default_split=eval_spec.split)
    roles = list(dict.fromkeys(args.roles)) if args.roles else list(tasks)
    unknown_roles = [role for role in roles if role not in tasks]
    if unknown_roles:
        raise ConfigError(f"--roles {unknown_roles} are not in pairing {args.pairing}; it has {list(tasks)}")
    num_episodes = eval_spec.num_episodes if args.num_episodes is None else args.num_episodes
    if num_episodes <= 0:
        raise ConfigError(f"--num-episodes must be positive, got {num_episodes}")
    pairing_dir = args.results_root / args.pairing
    pending: dict[str, list[int]] = {}
    for role in roles:
        role_dir = pairing_dir / role
        if args.overwrite:
            clear_role(role_dir)
        recorded = read_episodes(role_dir, pairing_dir, eval_spec)
        pending[role] = [episode for episode in range(num_episodes) if episode not in recorded]
        print(f"[{role} {tasks[role].name}] split {tasks[role].split}: {num_episodes - len(pending[role])} of "
              f"{num_episodes} episodes recorded, {len(pending[role])} to run", flush=True)
        if pending[role]:
            (role_dir / "summary.json").unlink(missing_ok=True)  # rewritten once all episodes exist
    needs_policy = any(pending.values())
    if not needs_policy:
        print("All requested episodes are recorded; no policy server is needed.", flush=True)
    server = contextlib.nullcontext()
    if needs_policy and args.launch_server:
        log_path = pairing_dir / f"policy_server_port{args.port}.log"
        server = PolicyServer(args.port, eval_spec.checkpoint_path, log_path)
    with server:
        policy = load_policy(eval_spec, port=args.port) if needs_policy else None
        for role, pending_episodes in pending.items():
            run_role(pairing_dir, role, tasks[role], pending_episodes, policy, eval_spec)
            episodes = read_episodes(pairing_dir / role, pairing_dir, eval_spec)
            write_summary(args.pairing, role, tasks[role], eval_spec,
                          [episodes[episode] for episode in range(num_episodes)], pairing_dir / role)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairing", required=True, help="pairing config name, e.g. pnp_cabinet_drawer")
    parser.add_argument("--roles", nargs="+", help="roles to run, e.g. i1 i2 (default: every role, in config order)")
    parser.add_argument("--num-episodes", type=int, default=None, help="episodes per role (default: eval config)")
    parser.add_argument("--results-root", type=Path, default=Path("baseline_eval/results"))
    parser.add_argument("--overwrite", action="store_true", help="delete recorded results of the roles and rerun")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--port", type=int, default=5555, help="policy server port (default: 5555)")
    parser.add_argument("--launch-server", action="store_true",
                        help="start evaluation/serve_policy.sh on --port for this run and stop it on exit")
    parser.add_argument("--checkpoint", help="model folder to evaluate instead of the eval config's checkpoint "
                        "(relative to the project root), e.g. checkpoints/experts/<Task>/step_<N>")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="seed of episode 0 (default: seed_start of the eval config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ConfigError, PolicyError, PolicyServerError, ResultsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume from the last recorded episode.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
