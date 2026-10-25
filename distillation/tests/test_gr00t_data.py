"""CPU tests of the Stage 3b data side: the observation buffer (write/read round trip through memory maps, plan and
record checks, task-balanced sampling statistics, holdout exclusion), the collection seed formula and job plans,
collect.py's RAM rule and subprocess runner, the config helpers and the round / early-stop / resume logic of rounds.py
with fake models, collections and selection evals, and that `evaluation/rollout.py` behaves exactly as before when no
`on_query` callback is given (the env, policy and robocasa helpers are fakes, so no simulator is needed)."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from distillation.gr00t import rounds
from distillation.gr00t.buffer import (
    LANGUAGE_KEY, STATE_KEYS, VIDEO_KEYS, ObservationBuffer, RawObs, collection_dir, collection_seed, init_collection,
    task_id, write_episode,
)
from distillation.gr00t.collect import Job, parallel_slots, plan_job, pending_episodes, run_parallel
from distillation.gr00t.train import load_config, stalled, unset_keys

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "distillation"
STATE_DIMS = (3, 4, 2, 3, 4)


def fake_obs(index: int, image: tuple[int, ...] = (4, 4, 3), instruction: str = "pick the bread") -> RawObs:
    """A RawObs whose state vectors all hold `index` (to identify it after sampling)."""
    rng = np.random.default_rng(index)
    obs: RawObs = {key: rng.integers(0, 256, image, dtype=np.uint8) for key in VIDEO_KEYS}
    obs.update({key: np.full(dim, index, dtype=np.float32) for key, dim in zip(STATE_KEYS, STATE_DIMS)})
    obs[LANGUAGE_KEY] = instruction
    return obs


def obs_index(obs: RawObs) -> int:
    """The index that `fake_obs` stored in the observation."""
    return int(obs["state.base_position"][0])


def commit(directory: Path, episode: int, observations: list[RawObs], plan: dict) -> None:
    """Write an episode's arrays and append its record, as collect.run_job does."""
    num_obs = write_episode(directory, episode, observations, [16 * k for k in range(len(observations))])
    record = {"episode": episode, "seed": collection_seed(plan["round"], episode), "success": episode % 2 == 0,
              "num_steps": 16 * num_obs, "horizon": 750, "wall_time_s": 1.0, "instruction": observations[0][LANGUAGE_KEY],
              "num_obs": num_obs, "policy": plan["policy"], "task_id": plan["task_id"], "round": plan["round"]}
    with open(directory / "episodes.jsonl", "a") as file:
        file.write(json.dumps(record) + "\n")


def make_collection(root: Path, role: str, num_episodes: int, obs_per_episode: int, start: int,
                    image: tuple[int, ...] = (4, 4, 3), label: str = "ckpt") -> Path:
    """A complete round-0 collection whose observations are fake_obs(start), fake_obs(start + 1), ..."""
    directory = collection_dir(root, label, role, 0)
    plan = {"policy": label, "checkpoint": "ckpt_dir", "pairing": "p", "role": role, "task": "T", "split": "target",
            "task_id": task_id(role), "round": 0, "first_episode": 0, "num_episodes": num_episodes}
    init_collection(directory, plan)
    for episode in range(num_episodes):
        first = start + episode * obs_per_episode
        commit(directory, episode, [fake_obs(first + i, image) for i in range(obs_per_episode)], plan)
    return directory


def test_buffer_round_trip_through_memory_maps(tmp_path: Path) -> None:
    """Real-sized observations come back equal, copied out of np.memmap arrays, with the episode's instruction."""
    directory = make_collection(tmp_path, "i1", num_episodes=1, obs_per_episode=3, start=0, image=(256, 256, 3))
    written = [fake_obs(i, image=(256, 256, 3)) for i in range(3)]
    buffer = ObservationBuffer([directory])
    assert buffer.task_counts() == {1: 3}
    assert all(isinstance(values, np.memmap) for values in buffer._arrays[0].values())
    for i, expected in enumerate(written):
        got = buffer.observation(i)
        assert got.keys() == expected.keys() and got[LANGUAGE_KEY] == expected[LANGUAGE_KEY]
        for key in (*VIDEO_KEYS, *STATE_KEYS):
            assert got[key].dtype == expected[key].dtype and np.array_equal(got[key], expected[key])
            assert not isinstance(got[key], np.memmap)
    assert np.load(directory / "ep_0000" / "step.npy").tolist() == [0, 16, 32]


def test_buffer_rejects_inconsistent_collections(tmp_path: Path) -> None:
    """Incomplete collections, a changed plan, re-writing a committed episode, wrong dtypes and seeds all raise."""
    directory = make_collection(tmp_path, "i1", num_episodes=2, obs_per_episode=2, start=0)
    plan = json.loads((directory / "collection.json").read_text())
    with pytest.raises(ValueError, match="planned as"):
        init_collection(directory, plan | {"num_episodes": 3})
    with pytest.raises(ValueError, match="already committed"):
        write_episode(directory, 0, [fake_obs(0)], [0])
    bad = fake_obs(0) | {"state.gripper_qpos": np.zeros(2, dtype=np.float64)}
    with pytest.raises(ValueError, match="dtype"):
        write_episode(directory, 5, [bad], [0])
    (directory / "collection.json").write_text(json.dumps(plan | {"num_episodes": 3}))
    with pytest.raises(ValueError, match="incomplete"):
        ObservationBuffer([directory])
    (directory / "collection.json").write_text(json.dumps(plan | {"round": 1}))
    with pytest.raises(ValueError, match="contradict"):
        ObservationBuffer([directory])


def test_uncommitted_episode_is_rewritten(tmp_path: Path) -> None:
    """Arrays left by an interrupted episode (no record line) are replaced on the next write."""
    directory = tmp_path / "c"
    directory.mkdir()
    write_episode(directory, 0, [fake_obs(1), fake_obs(2)], [0, 16])
    assert write_episode(directory, 0, [fake_obs(3)], [0]) == 1
    assert np.load(directory / "ep_0000" / "state.base_position.npy")[:, 0].tolist() == [3.0]
    assert not (directory / "ep_0000.tmp").exists()


def test_task_balanced_sampling_statistics(tmp_path: Path) -> None:
    """With 90 vs 10 observations, balanced batches are ≈ 50/50 by task and uniform within a task; unbalanced ≈ 90/10."""
    buffer = ObservationBuffer([make_collection(tmp_path, "i1", 9, 10, start=0),
                                make_collection(tmp_path, "i2", 2, 5, start=1000)])
    observations, tasks = buffer.sample(4000, rng=np.random.default_rng(0))
    assert abs(np.mean(tasks == 2) - 0.5) < 0.03
    assert all(task_id == (2 if obs_index(obs) >= 1000 else 1) for obs, task_id in zip(observations, tasks))
    counts = np.bincount([obs_index(obs) - 1000 for obs in observations if obs_index(obs) >= 1000], minlength=10)
    assert counts.min() > 150 and counts.max() < 250  # 200 expected per minority observation
    _, tasks = buffer.sample(4000, task_balance=False, rng=np.random.default_rng(1))
    assert abs(np.mean(tasks == 2) - 0.1) < 0.02


def test_holdout_is_balanced_deterministic_and_never_sampled(tmp_path: Path) -> None:
    """holdout(n, seed) reserves n/2 per task, repeatably, and sample() never returns those observations."""
    dirs = [make_collection(tmp_path, "i1", 4, 10, start=0), make_collection(tmp_path, "i2", 4, 10, start=1000)]
    buffer = ObservationBuffer(dirs)
    held, held_tasks = buffer.holdout(10, seed=3)
    assert sorted(held_tasks.tolist()) == [1] * 5 + [2] * 5
    assert buffer.task_counts() == {1: 35, 2: 35}
    again, _ = ObservationBuffer(dirs).holdout(10, seed=3)
    assert [obs_index(o) for o in again] == [obs_index(o) for o in held]
    sampled, _ = buffer.sample(5000, rng=np.random.default_rng(0))
    assert not {obs_index(o) for o in sampled} & {obs_index(o) for o in held}
    with pytest.raises(RuntimeError, match="already reserved"):
        buffer.holdout(2, seed=0)
    with pytest.raises(ValueError, match="would leave none"):
        ObservationBuffer(dirs).holdout(80, seed=0)


def test_collection_seeds_and_round_jobs() -> None:
    """Seeds are 200000 + round·1000 + episode only; teacher episodes continue after the student's (other scenes)."""
    assert [collection_seed(0, 0), collection_seed(0, 39), collection_seed(2, 19)] == [200000, 200039, 202019]
    for bad in [(0, 1000), (0, -1), (-1, 0)]:
        with pytest.raises(ValueError):
            collection_seed(*bad)
    cfg = load_config(CONFIGS / "level3_mtd_pilot.yaml", None)
    jobs = {(job.label, job.role): job for job in rounds.round_jobs(cfg, 0)}
    assert jobs.keys() == {("ckpt60k", "i1"), ("ckpt60k", "i2"), ("expert_i1", "i1"), ("expert_i2", "i2")}
    student, teacher = jobs["ckpt60k", "i1"], jobs["expert_i1", "i1"]
    seeds = [collection_seed(0, e) for job in (student, teacher)
             for e in range(job.first_episode, job.first_episode + job.num_episodes)]
    assert seeds == list(range(200000, 200040))
    later = rounds.round_jobs(cfg, 1)
    assert [(job.label, job.role, job.first_episode) for job in later] == \
        [("level3_mtd_pilot_round_0", role, 0) for role in ("i1", "i2")]
    assert later[0].checkpoint.endswith("checkpoints/students/level3_mtd_pilot/round_0")
    assert rounds.round_jobs(load_config(CONFIGS / "level1_recovery.yaml", None), 0) == \
        [Job("ckpt60k", role, 0, 20, cfg["checkpoints"]["ckpt60k"]) for role in ("i1", "i2")]


def test_collect_plans_resume_and_ram_rule(tmp_path: Path) -> None:
    """plan_job writes the plan once, pending_episodes skips committed ones; 2 slots only with >= 20 GB available."""
    job = Job("ckpt", "i1", 20, 3, "checkpoints/x")
    directory = plan_job(job, 0, "pnp_cabinet_drawer", tmp_path)
    plan = json.loads((directory / "collection.json").read_text())
    assert plan["task"] == "PickPlaceCounterToCabinet" and plan["task_id"] == 1 and plan["first_episode"] == 20
    commit(directory, 21, [fake_obs(0)], plan)
    assert pending_episodes(job, directory) == [20, 22]
    with pytest.raises(ValueError, match="planned as"):
        plan_job(Job("ckpt", "i1", 20, 4, "checkpoints/x"), 0, "pnp_cabinet_drawer", tmp_path)
    for kib, slots in [(25 * 1024**2, 2), (19 * 1024**2, 1)]:
        (tmp_path / "meminfo").write_text(f"MemTotal: 1 kB\nMemAvailable: {kib} kB\n")
        assert parallel_slots(tmp_path / "meminfo") == slots


def test_run_parallel_logs_and_fails_loudly(tmp_path: Path) -> None:
    """Every command runs with its output in its log; a failing one raises after all have finished."""
    ok = [sys.executable, "-c", "print('hello')"]
    bad = [sys.executable, "-c", "import sys; sys.exit(3)"]
    logs = [tmp_path / "a.log", tmp_path / "b.log", tmp_path / "c.log"]
    with pytest.raises(RuntimeError, match="b.log"):
        run_parallel([ok, bad, ok], logs, slots=2)
    assert logs[0].read_text().strip() == "hello" and logs[2].read_text().strip() == "hello"
    run_parallel([], [], slots=2)


def test_config_helpers() -> None:
    """Variants merge into the base and rename the run; null placeholders are reported; stalled() per its rule."""
    cfg = load_config(CONFIGS / "level2_single.yaml", "fm_only")
    assert cfg["run_id"] == "level2_single_fm_only" and cfg["distill"]["alpha_dmd"] == 0.0
    assert cfg["distill"]["k_fake"] == 5 and cfg["roles"] == ["i1"]
    assert unset_keys(cfg) == []
    pending = cfg | {"train": cfg["train"] | {"batch_size": None, "steps_per_round": None}}
    assert unset_keys(pending) == ["train.batch_size", "train.steps_per_round"]
    with pytest.raises(ValueError, match="variant"):
        load_config(CONFIGS / "level2_single.yaml", None)
    with pytest.raises(ValueError, match="unset"):
        rounds.check_config(pending)
    ready = cfg | {"train": cfg["train"] | {"batch_size": 8, "steps_per_round": 10}}
    rounds.check_config(ready)
    with pytest.raises(ValueError, match="seeds"):
        rounds.check_config(ready | {"selection": ready["selection"] | {"seed_start": 1001}})
    history = [{"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4}, {"a": 0.6, "b": 0.5}]
    assert not stalled(history, 2) and stalled(history + [{"a": 0.55, "b": 0.5}], 2)
    assert not stalled(history + [{"a": 0.55, "b": 0.55}], 2) and not stalled(history * 3, 0)


class FakeDistiller:
    """Stands in for train.Distiller: counts training, writes and reads a tiny state."""

    instances: list["FakeDistiller"] = []

    def __init__(self, cfg: dict) -> None:
        self.step, self.student, self.trained, self.loaded = 0, object(), [], []
        FakeDistiller.instances.append(self)

    def train(self, buffer, num_steps, log, metric, round_index) -> dict:
        self.step += num_steps
        self.trained.append(round_index)
        log({"round": round_index, "step": self.step})
        return {"steps": num_steps}

    def save(self, directory: Path, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").write_text(json.dumps({**meta, "step": self.step}))

    def load(self, directory: Path) -> dict:
        self.loaded.append(directory.name)
        state = json.loads((directory / "state.json").read_text())
        self.step = state["step"]
        return state


@pytest.fixture
def fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """rounds.py with fake collection, buffer, models, export, logger and selection (SR scripted per checkpoint)."""
    calls = SimpleNamespace(collect=[], export=[], selection=[], logged=[])
    sr = {"ckpt60k": 0.6, "round_0": 0.7, "round_1": 0.65, "round_2": 0.7, "round_3": 0.9}

    def selection(cfg, checkpoint, results_root, roles):
        calls.selection.append((Path(checkpoint).name, results_root.name, tuple(roles)))
        return {role: sr[Path(checkpoint).name.replace("checkpoint-60000", "ckpt60k")] for role in roles}
    monkeypatch.setattr(rounds, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(rounds, "STUDENTS_ROOT", tmp_path / "students")
    monkeypatch.setattr(rounds, "collect", lambda jobs, r, pairing: calls.collect.append((r, jobs)) or [])
    monkeypatch.setattr(rounds, "ObservationBuffer", lambda: SimpleNamespace(add=None, holdout=lambda n, s: ([], [])))
    monkeypatch.setattr(rounds, "Distiller", FakeDistiller)
    monkeypatch.setattr(rounds, "OfflineMetric", lambda *args: "metric")
    monkeypatch.setattr(rounds, "export_student", lambda s, out, info, base: calls.export.append(out.name) or out.mkdir(
        parents=True))
    monkeypatch.setattr(rounds, "RunLogger", lambda run_dir, cfg: SimpleNamespace(log=calls.logged.append))
    monkeypatch.setattr(rounds, "selection", selection)
    FakeDistiller.instances = []
    cfg = load_config(CONFIGS / "level2_single.yaml", "dmd_fm")
    cfg = cfg | {"max_rounds": 5, "train": cfg["train"] | {"batch_size": 8, "steps_per_round": 10}}
    return SimpleNamespace(cfg=cfg, calls=calls, tmp_path=tmp_path)


def test_rounds_stop_after_two_rounds_without_improvement(fake_run: SimpleNamespace) -> None:
    """SR 0.6 (init) → 0.7, 0.65, 0.7: rounds 1 and 2 do not beat 0.7, so it stops after round 2 of 5; best round 0."""
    rounds.Run(fake_run.cfg).execute()
    calls, run_dir = fake_run.calls, fake_run.tmp_path / "runs" / "level2_single_dmd_fm"
    progress = json.loads((run_dir / "rounds.json").read_text())
    assert progress["rounds_run"] == 3 and progress["best_round"] == 0 and progress["init"] == {"i1": 0.6, "i2": 0.6}
    assert [r for r, _ in calls.collect] == [0, 1, 2] and calls.export == ["round_0", "round_1", "round_2"]
    assert FakeDistiller.instances[0].trained == [0, 1, 2]
    assert calls.selection[0] == ("checkpoint-60000", "ckpt60k", ("i1", "i2"))  # init: roles + extra roles, shared
    assert calls.selection[-1] == ("round_0", "round_0", ("i2",))  # forgetting check of the best student
    assert [p.name for p in (run_dir / "state").iterdir()] == ["round_2"]
    assert {"round": 1, "step": 20, "selection/i1": 0.65} in calls.logged


def test_rounds_rerun_skips_finished_stages(fake_run: SimpleNamespace) -> None:
    """A re-run trains and exports nothing; with a missing export it reloads that round's state and re-exports."""
    rounds.Run(fake_run.cfg).execute()
    calls_before = len(fake_run.calls.export)
    rounds.Run(fake_run.cfg).execute()
    assert len(fake_run.calls.export) == calls_before and FakeDistiller.instances[1:] == []
    (fake_run.tmp_path / "students" / "level2_single_dmd_fm" / "round_2").rmdir()
    rounds.Run(fake_run.cfg).execute()
    assert fake_run.calls.export[-1] == "round_2" and FakeDistiller.instances[-1].loaded == ["round_2"]
    assert FakeDistiller.instances[-1].trained == []
    with pytest.raises(ValueError, match="differs"):
        rounds.Run(fake_run.cfg | {"max_rounds": 4})


@pytest.fixture
def rollout(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> ModuleType:
    """evaluation.rollout imported against stub robocasa modules (horizon 40, identity convert_action)."""
    stubs = {"robocasa": {}, "robocasa.utils": {},
             "robocasa.utils.dataset_registry_utils": {"get_task_horizon": lambda name: 40},
             "robocasa.utils.env_utils": {"convert_action": lambda row: row}}
    for name, attributes in stubs.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))
    sys.modules.pop("evaluation.rollout", None)
    request.addfinalizer(lambda: sys.modules.pop("evaluation.rollout", None))
    return importlib.import_module("evaluation.rollout")


class FakeEnv:
    """Gym-like env whose observation is the step count; success from step `success_at` on."""

    def __init__(self, success_at: int) -> None:
        self.t, self.success_at = 0, success_at
        meta = SimpleNamespace(get_ep_meta=lambda: {"lang": "pick"}, sim=SimpleNamespace(
            model=SimpleNamespace(camera_names=["robot0_agentview_left"])))
        self.unwrapped = SimpleNamespace(env=meta)

    def reset(self) -> tuple[dict, dict]:
        return {"t": 0}, {}

    def step(self, action: np.ndarray) -> tuple:
        self.t += 1
        return {"t": self.t}, 0.0, False, False, {"success": self.t >= self.success_at}

    def close(self) -> None:
        pass


@pytest.mark.parametrize("success_at", [20, 1000])
def test_rollout_on_query_default_changes_nothing(rollout: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                  success_at: int) -> None:
    """Without on_query the episode is identical to one with a recording callback, which sees each queried
    observation (the same object the policy gets) and its env step just before the query."""
    from evaluation.config import EvalSpec, TaskSpec
    monkeypatch.setattr(rollout, "gym", SimpleNamespace(make=lambda *args, **kwargs: FakeEnv(success_at)))
    spec = EvalSpec("c", Path("c"), "target", 1, 0, 16, "robot0_agentview_left", 8, 8, 20)
    task = TaskSpec("PickPlaceCounterToCabinet", "atomic", "target", None, None)
    results, queried = [], []
    for on_query in (None, lambda obs, step: queried.append((obs, step))):
        acted = []
        policy = SimpleNamespace(reset=lambda: None, act=lambda obs, instruction: acted.append(obs) or np.zeros((16, 12)))
        kwargs = {} if on_query is None else {"on_query": on_query}
        results.append((rollout.run_episode(task, 7, policy, spec, None, **kwargs), acted))
    (plain, plain_acted), (recorded, recorded_acted) = results
    assert {k: v for k, v in plain.items() if k != "wall_time_s"} == \
        {k: v for k, v in recorded.items() if k != "wall_time_s"}
    assert plain_acted == recorded_acted and [obs for obs, _ in queried] == recorded_acted
    assert all(a is b for a, (b, _) in zip(recorded_acted, queried))
    assert [step for _, step in queried] == ([0, 16] if success_at == 20 else [0, 16, 32])
    assert plain["num_steps"] == (32 if success_at == 20 else 40) and plain["success"] == (success_at == 20)
