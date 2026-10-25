"""On-disk buffer of raw policy observations for GR00T distillation, and task-balanced sampling from it. numpy and
stdlib only: collect.py (a2u-sim) writes and train.py / rounds.py (a2u-policy) read the same files.

Storage decision (spec §5): a cached fp16 backbone `cond` is ≈ 821 × 2048 × 2 B ≈ 3.36 MB per observation
(docs/distill_gr00t_notes.md §2), over the 2 MB limit, so the buffer stores the RAW observation of every policy query
and train.py runs the frozen backbone on the fly. Measured on the smoke collection (ckpt-60000, i1, seed 200000):
589,888 B of arrays per observation (three uint8 256×256×3 views + five float32 states), ≈ 590 kB on disk with the
.npy headers, plus the instruction once per episode.

One collection = one policy on one role in one round, shared by every run that needs it:
  <collections root>/<policy label>/<role>/round_<r>/
    collection.json     the plan (policy label, checkpoint, pairing, role, task, task id, round, first episode,
                        number of episodes); a re-use must ask for exactly this plan
    episodes.jsonl      one EPISODE_FIELDS line per finished episode, appended by the collector after the arrays are
                        in place: the line is the commit marker
    ep_<episode:04d>/   <key>.npy [num_obs, ...] per ARRAY_KEYS entry and step.npy (env step of each query), written
                        to ep_<episode:04d>.tmp/ and renamed
Reading memory-maps every array; only sampled observations are copied into RAM.
"""

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
STATE_KEYS = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
              "state.gripper_qpos", "state.base_position", "state.base_rotation")
ARRAY_KEYS = VIDEO_KEYS + STATE_KEYS
LANGUAGE_KEY = "annotation.human.task_description"  # = evaluation.policy.LANGUAGE_KEY (that module needs gr00t)
EPISODE_FIELDS = ("episode", "seed", "success", "num_steps", "horizon", "wall_time_s", "instruction",
                  "num_obs", "policy", "task_id", "round")
SEED_BASE, SEEDS_PER_ROUND = 200_000, 1_000

RawObs = dict[str, np.ndarray | str]


def collection_seed(round_index: int, episode: int) -> int:
    """Collection seed 200000 + round·1000 + episode (spec §1), disjoint from test and selection seeds."""
    if round_index < 0 or not 0 <= episode < SEEDS_PER_ROUND:
        raise ValueError(f"need round >= 0 and 0 <= episode < {SEEDS_PER_ROUND}, got {round_index}, {episode}")
    return SEED_BASE + round_index * SEEDS_PER_ROUND + episode


def task_id(role: str) -> int:
    """Router task id of an atomic role: i1 → 1, i2 → 2."""
    if re.fullmatch(r"i[1-9][0-9]*", role) is None:
        raise ValueError(f"role must be atomic (i<number>), got {role!r}")
    return int(role[1:])


def collection_dir(root: Path, label: str, role: str, round_index: int) -> Path:
    """Directory of the collection of policy `label` on `role` in round `round_index`."""
    return root / label / role / f"round_{round_index}"


def init_collection(directory: Path, plan: dict[str, Any]) -> None:
    """Write collection.json for a new collection, or raise unless an existing one holds the same plan."""
    path = directory / "collection.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != plan:
            raise ValueError(f"{path} was planned as {existing}, now asked for {plan}; use another collections root")
        return
    directory.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps(plan, indent=2) + "\n")
    path.with_suffix(".tmp").replace(path)


def read_records(directory: Path) -> dict[int, dict[str, Any]]:
    """Committed episode records of a collection by episode index; malformed or duplicate lines raise."""
    path = directory / "episodes.jsonl"
    records: dict[int, dict[str, Any]] = {}
    lines = path.read_text().splitlines() if path.exists() else []
    for number, line in enumerate(lines, start=1):
        record = json.loads(line)
        if sorted(record) != sorted(EPISODE_FIELDS) or record["episode"] in records:
            raise ValueError(f"{path} line {number}: not a new record with exactly the fields {EPISODE_FIELDS}")
        records[record["episode"]] = record
    return records


def _episode_dir(directory: Path, episode: int) -> Path:
    """Array directory of one episode."""
    return directory / f"ep_{episode:04d}"


def write_episode(directory: Path, episode: int, observations: Sequence[RawObs], steps: Sequence[int]) -> int:
    """Write one uncommitted episode's arrays (replacing leftovers of an interrupted write); return num_obs."""
    if episode in read_records(directory):
        raise ValueError(f"episode {episode} of {directory} is already committed")
    if not observations or len(steps) != len(observations):
        raise ValueError(f"need >= 1 observation and one step per observation, got {len(observations)}, {len(steps)}")
    if len({obs[LANGUAGE_KEY] for obs in observations}) != 1:
        raise ValueError(f"episode {episode}: the instruction changed within the episode")
    final, partial = _episode_dir(directory, episode), _episode_dir(directory, episode).with_suffix(".tmp")
    for stale in (final, partial):
        shutil.rmtree(stale, ignore_errors=True)
    partial.mkdir(parents=True)
    for key in ARRAY_KEYS:
        values = np.stack([obs[key] for obs in observations])
        expected = np.uint8 if key in VIDEO_KEYS else np.float32
        if values.dtype != expected:
            raise ValueError(f"{key} has dtype {values.dtype}, expected {np.dtype(expected)}")
        np.save(partial / f"{key}.npy", values)
    np.save(partial / "step.npy", np.asarray(steps, dtype=np.int32))
    partial.rename(final)
    return len(observations)


def _check_record(directory: Path, record: dict[str, Any], plan: dict[str, Any]) -> None:
    """Raise unless a record matches its collection's plan and the seed formula."""
    expected = {"seed": collection_seed(plan["round"], record["episode"]), "round": plan["round"],
                "task_id": plan["task_id"], "policy": plan["policy"]}
    wrong = {key: record[key] for key, value in expected.items() if record[key] != value}
    if wrong:
        raise ValueError(f"{directory} episode {record['episode']}: {wrong} contradict plan/seed formula {expected}")


class ObservationBuffer:
    """Union of complete collections; task-balanced sampling outside an optional reserved holdout."""

    def __init__(self, directories: Sequence[Path] = ()) -> None:
        """Load every given collection (see `add`)."""
        self._arrays: list[dict[str, np.ndarray]] = []
        self._instructions: list[str] = []
        self._task, self._episode, self._row = (np.zeros(0, dtype=np.int64) for _ in range(3))
        self._held = np.zeros(0, dtype=bool)
        for directory in directories:
            self.add(directory)

    def add(self, directory: Path) -> None:
        """Append every episode of one collection; raise if a planned episode is not committed yet."""
        plan = json.loads((directory / "collection.json").read_text())
        records = read_records(directory)
        planned = range(plan["first_episode"], plan["first_episode"] + plan["num_episodes"])
        missing = sorted(set(planned) - records.keys())
        if missing:
            raise ValueError(f"collection {directory} is incomplete: episodes {missing} are missing")
        for episode in planned:
            _check_record(directory, records[episode], plan)
            self._append_episode(_episode_dir(directory, episode), records[episode])

    def _append_episode(self, episode_dir: Path, record: dict[str, Any]) -> None:
        """Memory-map one episode's arrays and extend the observation index."""
        arrays = {key: np.load(episode_dir / f"{key}.npy", mmap_mode="r") for key in ARRAY_KEYS}
        if any(len(values) != record["num_obs"] for values in arrays.values()):
            raise ValueError(f"{episode_dir}: array lengths differ from num_obs {record['num_obs']}")
        n = record["num_obs"]
        self._task = np.concatenate([self._task, np.full(n, record["task_id"])])
        self._episode = np.concatenate([self._episode, np.full(n, len(self._arrays))])
        self._row = np.concatenate([self._row, np.arange(n)])
        self._held = np.concatenate([self._held, np.zeros(n, dtype=bool)])
        self._arrays.append(arrays)
        self._instructions.append(record["instruction"])

    def task_counts(self) -> dict[int, int]:
        """Observations available for sampling (holdout excluded) per task id."""
        tasks, counts = np.unique(self._task[~self._held], return_counts=True)
        return dict(zip(tasks.tolist(), counts.tolist()))

    def observation(self, index: int) -> RawObs:
        """Observation `index` as a RawObs, copied out of the memory map."""
        arrays = self._arrays[self._episode[index]]
        obs: RawObs = {key: np.array(values[self._row[index]]) for key, values in arrays.items()}
        obs[LANGUAGE_KEY] = self._instructions[self._episode[index]]
        return obs

    def sample(self, batch_size: int, task_balance: bool = True, *,
               rng: np.random.Generator) -> tuple[list[RawObs], np.ndarray]:
        """Draw with replacement outside the holdout; balanced: task uniform among those present, then obs uniform."""
        pool = np.flatnonzero(~self._held)
        if pool.size == 0 or batch_size < 1:
            raise ValueError(f"cannot sample {batch_size} from {pool.size} available observations")
        if task_balance:
            pools = {task: pool[self._task[pool] == task] for task in np.unique(self._task[pool]).tolist()}
            chosen = rng.choice(list(pools), size=batch_size)
            indices = np.array([pools[task][rng.integers(len(pools[task]))] for task in chosen.tolist()])
        else:
            indices = rng.choice(pool, size=batch_size)
        return [self.observation(i) for i in indices.tolist()], self._task[indices]

    def holdout(self, n: int, seed: int) -> tuple[list[RawObs], np.ndarray]:
        """Reserve n observations (split evenly over tasks, deterministic in seed), never sampled; return them."""
        if self._held.any():
            raise RuntimeError("a holdout is already reserved in this buffer")
        rng, tasks = np.random.default_rng(seed), np.unique(self._task).tolist()
        for k, task in enumerate(tasks):
            pool = np.flatnonzero(self._task == task)
            size = n // len(tasks) + int(k < n % len(tasks))
            if size >= len(pool):
                raise ValueError(f"holdout of {size} would leave none of the {len(pool)} observations of task {task}")
            self._held[rng.choice(pool, size=size, replace=False)] = True
        indices = np.flatnonzero(self._held)
        return [self.observation(i) for i in indices.tolist()], self._task[indices]
