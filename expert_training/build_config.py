"""Build the RLinf config of one single-task expert run and check it before anything starts.

Composes StructRL's GR00T-N1.5 PPO base config with its Hydra defaults and applies, in order: our
SFT checkpoint (rollout and actor; configs/eval/baseline.yaml), split=target, task_filter=[task],
the task's horizon, configs/rl/ppo_sparse.yaml, configs/rl/hw_<hw>.yaml, output paths under the run
directory. RLinf has no split/task_filter/task_soup/task_mode fields: split and task_filter become
env.{train,eval}.rldx_robocasa_split and .task_names; task_soup (atomic_seen) and task_mode (atomic)
are checked against the eval stack's registry (task_registry.py, which also gives the horizon) and
recorded in an `a2u_expert:` block. RLinf's batch divisibility rules (batch_rules.py) are checked
with the hardware file's GPU count. Writes <run_dir>/resolved_config.yaml fully resolved: the exact file
train_expert.sh launches. --dry-run prints the key fields and the checks and writes nothing.

Usage (RL env): python expert_training/build_config.py --task <Task> --hw <workstation|aws_node>
    [--run-dir DIR] [--resume-dir <run_dir>/checkpoints/global_step_N] [--dry-run]
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import batch_rules
from task_registry import RegistryError, lookup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # run as a script: make the project's `evaluation` package importable
from evaluation.config import ConfigError, load_eval_config  # noqa: E402

STRUCTRL_ROOT = PROJECT_ROOT / "third_party" / "StructRL"
RL_CONFIG_DIR = PROJECT_ROOT / "configs" / "rl"
RUNS_ROOT = PROJECT_ROOT / "expert_training" / "runs"
TASK_SOUP, TASK_MODE, SPLIT = "atomic_seen", "atomic", "target"
TUNING_MODES = ("as_structrl",)  # StructRL's mode: action head + value head trained, VLM frozen
EVAL_SEEDS = (range(0, 100), range(1000, 1020))  # final test and checkpoint selection (spec rule 4)
NEW_KEYS = ("env.train.rldx_robocasa_split", "env.eval.rldx_robocasa_split")  # absent from the GR00T base config
SETTING_TYPES = {"base_config": str, "reward": dict, "iterations": int, "train_seed_offset": int, "save_every": int,
                 "keep_last_raw": int, "gpus": int, "num_envs": int, "rollout_epochs": int, "global_batch": int,
                 "micro_batch": int, "tuning": str}


class BuildError(RuntimeError):
    """The settings, the task or the batch sizes are invalid; the message says what to fix."""


ERRORS = (BuildError, ConfigError, RegistryError)  # what build() raises with a message for the user


def hw_choices() -> list[str]:
    return sorted(path.stem.removeprefix("hw_") for path in RL_CONFIG_DIR.glob("hw_*.yaml"))


def load_settings(hw: str) -> dict[str, Any]:
    """Merge configs/rl/ppo_sparse.yaml and configs/rl/hw_<hw>.yaml (hardware wins) and check every value."""
    settings: dict[str, Any] = {}
    for path in (RL_CONFIG_DIR / "ppo_sparse.yaml", RL_CONFIG_DIR / f"hw_{hw}.yaml"):
        if not path.is_file():
            raise BuildError(f"{path} not found; --hw must be one of {hw_choices()}")
        settings.update(yaml.safe_load(path.read_text()) or {})
    unknown = sorted(set(settings) - set(SETTING_TYPES))
    if unknown:
        raise BuildError(f"unknown keys {unknown} in configs/rl/ppo_sparse.yaml or hw_{hw}.yaml; allowed: {list(SETTING_TYPES)}")
    for key, expected in SETTING_TYPES.items():
        value = settings.get(key)
        if not isinstance(value, expected) or isinstance(value, bool) or (expected is int and value < 1):
            raise BuildError(f"'{key}' must be a {'positive int' if expected is int else expected.__name__}, got {value!r}")
    if settings["tuning"] not in TUNING_MODES:
        raise BuildError(f"tuning {settings['tuning']!r} is not implemented; use one of {list(TUNING_MODES)}")
    seeds = range(settings["train_seed_offset"], settings["train_seed_offset"] + settings["num_envs"])
    clashes = [f"{r.start}-{r.stop - 1}" for r in EVAL_SEEDS if set(seeds) & set(r)]
    if clashes:
        raise BuildError(f"training env seeds {seeds.start}-{seeds.stop - 1} overlap evaluation seeds {clashes}; "
                         "raise train_seed_offset in configs/rl/ppo_sparse.yaml")
    return settings


def compose_base(base_config: str) -> DictConfig:
    """Compose the base config with its Hydra defaults, as train_embodied_agent.py would."""
    path = STRUCTRL_ROOT / base_config
    if not path.is_file():
        raise BuildError(f"base config {path} not found; run StructRL's scripts/setup_upstream.sh (docs/environment.md)")
    os.environ["PROJECT_ROOT"] = str(STRUCTRL_ROOT)  # StructRL launcher variables the base config interpolates
    os.environ["EMBODIED_PATH"] = str(path.parent.parent)
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.1"):
        return compose(config_name=path.stem)


def overrides(task: str, settings: dict, horizon: int, chunk: int, checkpoint: Path, run_dir: Path,
              resume_dir: Path | None) -> dict[str, Any]:
    """RLinf keys to set, in the order they are applied (a later key wins)."""
    steps = math.ceil(horizon / chunk) * chunk  # whole action chunks covering the horizon
    values = {"rollout.model.model_path": str(checkpoint), "actor.model.model_path": str(checkpoint)}
    for split in ("train", "eval"):
        values |= {f"env.{split}.rldx_robocasa_split": SPLIT, f"env.{split}.task_names": [task],
                   f"env.{split}.max_episode_steps": horizon, f"env.{split}.max_steps_per_rollout_epoch": steps}
    return values | settings["reward"] | {
        "runner.max_epochs": settings["iterations"], "runner.save_interval": settings["save_every"],
        "env.train.seed": settings["train_seed_offset"], "cluster.num_nodes": 1,
        "env.train.total_num_envs": settings["num_envs"], "algorithm.rollout_epoch": settings["rollout_epochs"],
        "actor.global_batch_size": settings["global_batch"], "actor.micro_batch_size": settings["micro_batch"],
        "runner.logger.log_path": str(run_dir),
        "runner.logger.experiment_name": ".",  # RLinf saves to <log_path>/<experiment_name>/checkpoints/global_step_N
        "runner.resume_dir": None if resume_dir is None else str(resume_dir)}


def resolve(task: str, hw: str, run_dir: Path, resume_dir: Path | None = None) -> DictConfig:
    """Compose, override and check the run config; print its key fields."""
    settings = load_settings(hw)
    registry = lookup(task, TASK_SOUP)
    checkpoint = load_eval_config().checkpoint_path
    cfg = compose_base(settings["base_config"])
    values = overrides(task, settings, registry["horizon"], cfg.actor.model.num_action_chunks, checkpoint,
                       run_dir, resume_dir)
    for key, value in values.items():
        if key not in NEW_KEYS and OmegaConf.select(cfg, key, default=KeyError) is KeyError:
            raise BuildError(f"{key} is not in the base config {settings['base_config']}; fix the override")
        OmegaConf.update(cfg, key, value, merge=False, force_add=key in NEW_KEYS)
    meta = {"task": task, "task_soup": TASK_SOUP, "task_filter": [task], "task_mode": TASK_MODE, "split": SPLIT,
            "tuning": settings["tuning"], "train_seed_offset": settings["train_seed_offset"],
            "horizon": registry["horizon"], "base_config": settings["base_config"], "checkpoint": str(checkpoint),
            "hw": hw, "gpus": settings["gpus"], "keep_last_raw": settings["keep_last_raw"],
            "eval_registry": registry["robocasa"]}
    OmegaConf.update(cfg, "a2u_expert", meta, force_add=True)
    shown = {"a2u_expert": meta, "overrides": {key: OmegaConf.select(cfg, key) for key in values}}
    print(yaml.safe_dump(OmegaConf.to_container(OmegaConf.create(shown)), sort_keys=False), end="")
    failures = batch_rules.check(cfg.env.train.total_num_envs, cfg.algorithm.rollout_epoch, cfg.actor.global_batch_size,
                                 cfg.actor.micro_batch_size, settings["gpus"], cfg.env.train.max_steps_per_rollout_epoch,
                                 cfg.actor.model.num_action_chunks, cfg.rollout.pipeline_stage_num, cfg.env.train.group_size)
    if failures:
        raise BuildError(f"RLinf batch divisibility rules violated: {failures}; adjust configs/rl/hw_{hw}.yaml")
    return cfg


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}; lists stay values."""
    if not isinstance(tree, dict):
        return {prefix: tree}
    return {key: value for name, child in tree.items()
            for key, value in flatten(child, f"{prefix}.{name}" if prefix else str(name)).items()}


def build(task: str, hw: str, run_dir: Path, resume_dir: Path | None = None) -> Path:
    """Write <run_dir>/resolved_config.yaml; with resume_dir, only runner.resume_dir may change."""
    run_dir = run_dir.resolve()
    cfg = resolve(task, hw, run_dir, resume_dir)
    content = {"defaults": [{"override hydra/job_logging": "stdout"}, "_self_"],  # as in the base config
               "hydra": {"run": {"dir": "."}, "output_subdir": None}, **OmegaConf.to_container(cfg, resolve=True)}
    path = run_dir / "resolved_config.yaml"
    if path.exists() and resume_dir is None:
        raise BuildError(f"{path} exists: run {run_dir.name} was already started; resume it or pick a new run id")
    if path.exists():
        before, after = flatten(yaml.safe_load(path.read_text())), flatten(content)
        changed = sorted(key for key in before.keys() | after.keys()
                         if key != "runner.resume_dir" and before.get(key) != after.get(key))
        if changed:
            raise BuildError(f"resuming {run_dir} would change {changed}; resume with the settings it started with")
    run_dir.mkdir(parents=True, exist_ok=True)
    header = (f"# Built by expert_training/build_config.py --task {task} --hw {hw} on {datetime.now():%Y-%m-%d %H:%M}.\n"
              "# train_expert.sh launches RLinf on exactly this file; do not edit it.\n")
    path.write_text(header + yaml.safe_dump(content, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="atomic_seen task, e.g. PickPlaceCounterToCabinet")
    parser.add_argument("--hw", required=True, help=f"hardware file configs/rl/hw_<hw>.yaml: {hw_choices()}")
    parser.add_argument("--run-dir", type=Path, help="default: expert_training/runs/<task>/<YYYYmmdd-HHMM>-<hw>")
    parser.add_argument("--resume-dir", type=Path, help="raw checkpoint to resume from (runner.resume_dir)")
    parser.add_argument("--dry-run", action="store_true", help="print the key fields and checks, write nothing")
    args = parser.parse_args()
    run_dir = (args.run_dir or RUNS_ROOT / args.task / f"{datetime.now():%Y%m%d-%H%M}-{args.hw}").resolve()
    resume_dir = None if args.resume_dir is None else args.resume_dir.resolve()
    try:
        if args.dry_run:
            resolve(args.task, args.hw, run_dir, resume_dir)
            print(f"Dry run: nothing written (run dir would be {run_dir}).")
        else:
            print(f"Wrote {build(args.task, args.hw, run_dir, resume_dir)}")
    except ERRORS as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
