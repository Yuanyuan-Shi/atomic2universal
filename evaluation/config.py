"""Load and validate the pairing and eval configs into frozen dataclasses.

A pairing config (configs/pairings/<name>.yaml) says WHAT is evaluated: one or more roles (atomic i1,
i2, ... and an optional composite), each a RoboCasa365 task with a mode, optional split override and
reference success rate; roles keep their config order.
The eval config (configs/eval/baseline.yaml) says HOW; bad values fail before anything starts.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAIRINGS_DIR = PROJECT_ROOT / "configs" / "pairings"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "eval" / "baseline.yaml"
ROLE_PATTERN = re.compile(r"i[1-9][0-9]*|composite")
MODES = ("atomic", "composite")


class ConfigError(ValueError):
    """A config file is missing or holds an invalid value."""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    mode: str
    split: str
    reference_sr: float | None
    reference_note: str | None


@dataclass(frozen=True)
class EvalSpec:
    checkpoint: str  # as written in the config (recorded in summary.json)
    checkpoint_path: Path  # absolute; a relative `checkpoint` is taken from the project root
    split: str
    num_episodes: int
    seed_start: int
    action_horizon: int
    video_camera: str
    video_width: int
    video_height: int
    video_fps: int


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _read_yaml(path: Path) -> dict[str, Any]:
    _check(path.is_file(), f"config file not found: {path}")
    content = yaml.safe_load(path.read_text())
    _check(isinstance(content, dict), f"{path} must contain a YAML mapping at the top level")
    return content


def _require(mapping: dict[str, Any], key: str, expected_type: type, where: str, minimum: int | None = None) -> Any:
    value = mapping.get(key)
    valid = isinstance(value, expected_type) and not isinstance(value, bool) and value != ""
    _check(valid, f"{where}: '{key}' must be set to a {expected_type.__name__}, got {value!r}")
    _check(minimum is None or value >= minimum, f"{where}: '{key}' must be >= {minimum}, got {value}")
    return value


def load_eval_config(path: Path | str = DEFAULT_EVAL_CONFIG) -> EvalSpec:
    """Load the eval config and check that the checkpoint directory exists."""
    where = str(path)
    content = _read_yaml(Path(path))
    video = content.get("video")
    _check(isinstance(video, dict), f"{where}: 'video' must be a mapping with camera, width, height, fps")
    checkpoint = _require(content, "checkpoint", str, where)
    checkpoint_path = Path(checkpoint) if Path(checkpoint).is_absolute() else PROJECT_ROOT / checkpoint
    _check((checkpoint_path / "config.json").is_file(),
           f"checkpoint directory {checkpoint_path} is missing or has no config.json; "
           "run `bash setup/fetch_checkpoint.sh` or fix 'checkpoint' in the eval config")
    return EvalSpec(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        split=_require(content, "split", str, where),
        num_episodes=_require(content, "num_episodes", int, where, minimum=1),
        seed_start=_require(content, "seed_start", int, where, minimum=0),
        action_horizon=_require(content, "action_horizon", int, where, minimum=1),
        video_camera=_require(video, "camera", str, f"{where} video"),
        video_width=_require(video, "width", int, f"{where} video", minimum=1),
        video_height=_require(video, "height", int, f"{where} video", minimum=1),
        video_fps=_require(video, "fps", int, f"{where} video", minimum=1),
    )


def load_pairing(name: str, default_split: str | None = None) -> dict[str, TaskSpec]:
    """Load configs/pairings/<name>.yaml as {role: TaskSpec}; a task's `split` overrides
    `default_split`, which defaults to the split in configs/eval/baseline.yaml."""
    path = PAIRINGS_DIR / f"{name}.yaml"
    content = _read_yaml(path)
    _check(content.get("name") == name, f"{path}: 'name' is {content.get('name')!r}, expected {name!r}")
    tasks = content.get("tasks")
    _check(isinstance(tasks, dict) and len(tasks) > 0,
           f"{path}: 'tasks' must map one or more roles (i1, i2, ... and an optional composite), got {tasks!r}")
    if default_split is None:
        default_split = load_eval_config(DEFAULT_EVAL_CONFIG).split
    specs = {}
    for role, entry in tasks.items():
        where = f"{path} tasks.{role}"
        _check(ROLE_PATTERN.fullmatch(str(role)) is not None, f"{where}: a role must be i<number> or 'composite'")
        _check(isinstance(entry, dict), f"{where} must be a mapping")
        mode = _require(entry, "mode", str, where)
        _check(mode in MODES, f"{where}: 'mode' must be one of {list(MODES)}, got {mode!r}")
        reference_sr = entry.get("reference_sr")
        _check(reference_sr is None or (isinstance(reference_sr, (int, float)) and 0 <= reference_sr <= 1),
               f"{where}: 'reference_sr' must be a number in [0, 1], got {reference_sr!r}")
        specs[role] = TaskSpec(
            name=_require(entry, "name", str, where),
            mode=mode,
            split=_require(entry, "split", str, where) if "split" in entry else default_split,
            reference_sr=None if reference_sr is None else float(reference_sr),
            reference_note=entry.get("reference_note"),
        )
    return specs
