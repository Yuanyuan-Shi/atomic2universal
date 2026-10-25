"""Collect distillation data: closed-loop rollouts of one policy on one role that record the raw observation at
every policy query (evaluation/rollout.py `on_query`) into a buffer collection (distillation/gr00t/buffer.py), no video.

A job is (policy label, role, first episode, number of episodes, checkpoint) in one round. Episode e uses seed
200000 + round·1000 + e (`buffer.collection_seed`), nothing else; student jobs use episodes 0..n−1 and round-0 teacher
jobs continue at n..2n−1, so their scenes differ. Committed episodes are skipped (a re-run resumes); a complete
collection starts no server. One job runs in this process (a2u-sim) against a policy server it launches on --port
(`evaluation.policy_server.PolicyServer`). Several jobs run as one a2u-sim subprocess each on ports --port, --port+1,
..., two at a time when /proc/meminfo MemAvailable >= 20 GB, else one. That RAM rule is ours: Stage 2b ran its evals
serially, and two server + sim pairs need about 17 GB of host RAM at worst (docs/distill_gr00t_notes.md §9).

Usage (from the project root; a2u-sim for one job, any env with numpy + yaml for several):
  python -m distillation.gr00t.collect --round 0 --job LABEL ROLE FIRST_EPISODE NUM_EPISODES CHECKPOINT
      [--job ...] [--pairing pnp_cabinet_drawer] [--collections-root distillation/runs/collections] [--port 5600]
"""

import argparse
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from distillation.gr00t.buffer import (
    ARRAY_KEYS, LANGUAGE_KEY, RawObs, collection_dir, collection_seed, init_collection, read_records, task_id,
    write_episode,
)
from evaluation.config import PROJECT_ROOT, load_eval_config, load_pairing
from evaluation.policy import load_policy
from evaluation.policy_server import PolicyServer

SIM_ENV = "a2u-sim"
DEFAULT_PAIRING = "pnp_cabinet_drawer"
COLLECTIONS_ROOT = PROJECT_ROOT / "distillation" / "runs" / "collections"
DEFAULT_PORT = 5600
PARALLEL_RAM_GB = 20.0


@dataclass(frozen=True)
class Job:
    """Policy `label` (served from `checkpoint`: absolute or project-relative) on `role`, episodes first..+n−1."""

    label: str
    role: str
    first_episode: int
    num_episodes: int
    checkpoint: str


def parallel_slots(meminfo: Path = Path("/proc/meminfo")) -> int:
    """Processes that may run at once: 2 when MemAvailable >= PARALLEL_RAM_GB, else 1."""
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return 2 if int(line.split()[1]) / 1024**2 >= PARALLEL_RAM_GB else 1
    raise RuntimeError(f"MemAvailable is missing from {meminfo}")


def sim_command(module: str, *args: str) -> list[str]:
    """`conda run` command that runs `python -u -m module args` in the sim env (a2u-sim)."""
    conda = os.environ.get("CONDA_EXE", "conda")
    return [conda, "run", "--no-capture-output", "-n", SIM_ENV, "python", "-u", "-m", module, *args]


def run_parallel(commands: Sequence[list[str]], log_paths: Sequence[Path], slots: int) -> None:
    """Run each command in its own session, at most `slots` at a time, appending its output to its log;
    stop the running ones if this process is interrupted, and raise once all ended if any failed."""
    queue, running, failed = list(zip(commands, log_paths)), [], []
    try:
        while queue or running:
            while queue and len(running) < slots:
                command, log_path = queue.pop(0)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
                                               start_new_session=True)
                print(f"started (log {log_path}): {shlex.join(command)}", flush=True)
                running.append((process, log_path))
            time.sleep(1.0)
            for process, log_path in [item for item in running if item[0].poll() is not None]:
                running.remove((process, log_path))
                print(f"finished with exit code {process.returncode}: {log_path}", flush=True)
                if process.returncode != 0:
                    failed.append(str(log_path))
    finally:
        for process, _ in running:  # its PolicyServer's SIGTERM handler stops its server
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
    if failed:
        raise RuntimeError(f"{len(failed)} subprocess(es) failed; see {failed}")


def plan_job(job: Job, round_index: int, pairing: str, root: Path) -> Path:
    """Create (or check the plan of) the job's collection directory and return it."""
    task = load_pairing(pairing)[job.role]
    directory = collection_dir(root, job.label, job.role, round_index)
    init_collection(directory, {
        "policy": job.label, "checkpoint": job.checkpoint, "pairing": pairing, "role": job.role, "task": task.name,
        "split": task.split, "task_id": task_id(job.role), "round": round_index,
        "first_episode": job.first_episode, "num_episodes": job.num_episodes,
    })
    return directory


def pending_episodes(job: Job, directory: Path) -> list[int]:
    """The job's episodes that are not committed in its collection yet."""
    done = read_records(directory)
    return [e for e in range(job.first_episode, job.first_episode + job.num_episodes) if e not in done]


def raw_observation(observation: dict[str, Any]) -> RawObs:
    """Copy of the RawObs keys of an env observation (the arrays and instruction that Policy.act sends)."""
    obs: RawObs = {key: np.array(observation[key]) for key in ARRAY_KEYS}
    obs[LANGUAGE_KEY] = str(observation[LANGUAGE_KEY])
    return obs


def run_job(job: Job, round_index: int, pairing: str, root: Path, port: int) -> None:
    """Run the job's pending episodes in this process (a2u-sim) against a policy server launched on `port`."""
    from baseline_eval.run_baseline import append_episode, apply_overrides  # imports robocasa: a2u-sim only
    from evaluation.rollout import run_episode
    directory = plan_job(job, round_index, pairing, root)
    pending = pending_episodes(job, directory)
    prefix = f"[{job.label} {job.role} round {round_index}]"
    print(f"{prefix} {job.num_episodes - len(pending)} of {job.num_episodes} episodes recorded, "
          f"{len(pending)} to run -> {directory}", flush=True)
    if not pending:
        return
    eval_spec = apply_overrides(load_eval_config(), job.checkpoint, None)
    task = load_pairing(pairing, default_split=eval_spec.split)[job.role]
    with PolicyServer(port, eval_spec.checkpoint_path, directory / f"policy_server_port{port}.log"):
        policy = load_policy(eval_spec, port=port)
        for episode in pending:
            queries: list[tuple[RawObs, int]] = []
            result = run_episode(task, collection_seed(round_index, episode), policy, eval_spec, None,
                                 on_query=lambda obs, step: queries.append((raw_observation(obs), step)))
            num_obs = write_episode(directory, episode, [obs for obs, _ in queries], [step for _, step in queries])
            append_episode(directory / "episodes.jsonl", {
                **result, "episode": episode, "num_obs": num_obs, "policy": job.label,
                "task_id": task_id(job.role), "round": round_index})
            print(f"{prefix} episode {episode} seed {result['seed']}: {'SUCCESS' if result['success'] else 'FAIL'} "
                  f"in {result['num_steps']} steps, {num_obs} observations ({result['wall_time_s']:.1f} s)", flush=True)


def job_args(job: Job) -> list[str]:
    """The --job arguments that describe `job` on the command line."""
    return ["--job", job.label, job.role, str(job.first_episode), str(job.num_episodes), job.checkpoint]


def collect(jobs: Sequence[Job], round_index: int, pairing: str = DEFAULT_PAIRING, root: Path = COLLECTIONS_ROOT,
            port: int = DEFAULT_PORT) -> list[Path]:
    """Complete every job's collection, one a2u-sim subprocess per incomplete job (≤ parallel_slots() at a
    time, ports port, port+1, ...); return the collection directories in job order."""
    directories = [plan_job(job, round_index, pairing, root) for job in jobs]
    pending = [(job, d) for job, d in zip(jobs, directories) if pending_episodes(job, d)]
    commands = [sim_command("distillation.gr00t.collect", "--round", str(round_index), "--pairing", pairing,
                            "--collections-root", str(root), "--port", str(port + k), *job_args(job))
                for k, (job, _) in enumerate(pending)]
    run_parallel(commands, [d / "collect.log" for _, d in pending], parallel_slots())
    incomplete = [str(d) for job, d in zip(jobs, directories) if pending_episodes(job, d)]
    if incomplete:
        raise RuntimeError(f"collections still incomplete after collection: {incomplete}")
    return directories


def parse_args() -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, required=True, help="round index r (seeds 200000 + r·1000 + episode)")
    parser.add_argument("--job", nargs=5, action="append", required=True,
                        metavar=("LABEL", "ROLE", "FIRST_EPISODE", "NUM_EPISODES", "CHECKPOINT"))
    parser.add_argument("--pairing", default=DEFAULT_PAIRING)
    parser.add_argument("--collections-root", type=Path, default=COLLECTIONS_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port of the first job's policy server")
    return parser.parse_args()


def main() -> None:
    """Run one job in-process, or several as parallel subprocesses."""
    args = parse_args()
    jobs = [Job(label, role, int(first), int(n), checkpoint) for label, role, first, n, checkpoint in args.job]
    if len(jobs) == 1:
        run_job(jobs[0], args.round, args.pairing, args.collections_root, args.port)
    else:
        collect(jobs, args.round, args.pairing, args.collections_root, args.port)


if __name__ == "__main__":
    main()
