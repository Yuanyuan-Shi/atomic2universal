"""Check that an exported expert folder is a clean policy-only checkpoint that the eval server loads and runs.

1. Reads the safetensors index and every shard header (no weights loaded): fails if any tensor
   name contains `value_head` (the PPO critic must not ship) or if any tensor is not bf16 (RLinf
   trains and saves in bf16, so anything else did not come through the RL pipeline); reports the
   dtype counts.
2. Loads the folder with the RoboCasa365 fork's Gr00tPolicy exactly as scripts/inference_service.py
   --server does for evaluation/serve_policy.sh (data config panda_omron, embodiment
   new_embodiment, 4 denoising steps) and fails if transformers reports missing, unexpected or
   newly initialized weights.
3. Runs one get_action on a dummy observation laid out like evaluation/policy.py's requests
   (shapes from the folder's experiment_cfg/metadata.json) and checks the chunk against that
   module's ACTION_LAYOUT: keys, shape (1, >=16, width), finite values.
Run in the a2u-policy env from the project root (GPU):
    python -m expert_training.verify_expert --checkpoint checkpoints/experts/<Task>/step_<N>
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from evaluation.policy import ACTION_LAYOUT, LANGUAGE_KEY

DATA_CONFIG, EMBODIMENT, DENOISING_STEPS = "panda_omron", "new_embodiment", 4  # as evaluation/serve_policy.sh
EXPECTED_DTYPE = "BF16"
CHUNK_LENGTH = 16
LOAD_WARNINGS = ("were not used when initializing", "newly initialized", "not initialized from the model checkpoint")


class VerifyError(RuntimeError):
    """The folder is not a clean, loadable expert checkpoint."""


def check_tensors(folder: Path) -> Counter:
    """Fail on value_head tensors or non-bf16 dtypes in the index or shard headers; return dtype counts."""
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    dtypes: Counter = Counter()
    names = set(index)
    for shard in sorted(set(index.values())):
        with open(folder / shard, "rb") as file:
            header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
        header.pop("__metadata__", None)
        names |= set(header)
        dtypes.update(entry["dtype"] for entry in header.values())
    critic = sorted(name for name in names if "value_head" in name)
    if critic:
        raise VerifyError(f"{len(critic)} value_head tensors in {folder}, e.g. {critic[:3]}; the converter must drop them")
    if set(dtypes) != {EXPECTED_DTYPE}:
        raise VerifyError(f"tensor dtypes {dict(dtypes)} in {folder}, expected only {EXPECTED_DTYPE}")
    return dtypes


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def load_policy(folder: Path, device: str = "cuda") -> Gr00tPolicy:
    """Gr00tPolicy as the eval server builds it; fail on transformers' missing/unexpected-weight warnings."""
    data_config = DATA_CONFIG_MAP[DATA_CONFIG]
    collector = _Collect()
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.addHandler(collector)
    try:
        policy = Gr00tPolicy(model_path=str(folder), modality_config=data_config.modality_config(),
                             modality_transform=data_config.transform(), embodiment_tag=EMBODIMENT,
                             denoising_steps=DENOISING_STEPS, device=device)
    finally:
        transformers_logger.removeHandler(collector)
    bad = [message for message in collector.messages if any(text in message for text in LOAD_WARNINGS)]
    if bad:
        raise VerifyError("loading reported weight mismatches:\n" + "\n".join(message[:1500] for message in bad))
    return policy


def dummy_request(folder: Path) -> dict:
    """One observation in evaluation/policy.py's request layout: (batch=1, time=1, ...) arrays."""
    modalities = json.loads((folder / "experiment_cfg" / "metadata.json").read_text())[EMBODIMENT]["modalities"]
    request = {}
    for key, spec in modalities["video"].items():
        width, height = spec["resolution"]
        request[f"video.{key}"] = np.full((1, 1, height, width, spec["channels"]), 128, dtype=np.uint8)
    for key, spec in modalities["state"].items():
        state = np.zeros((1, 1, *spec["shape"]), dtype=np.float32)
        if spec["rotation_type"] == "quaternion":
            state[..., -1] = 1.0  # a unit quaternion
        request[f"state.{key}"] = state
    request[LANGUAGE_KEY] = np.array(["pick the object from the counter and place it in the cabinet"])
    return request


def check_action(policy: Gr00tPolicy, folder: Path) -> None:
    response = policy.get_action(dummy_request(folder))
    expected = sorted(key for key, _ in ACTION_LAYOUT)
    if sorted(response) != expected:
        raise VerifyError(f"get_action returned keys {sorted(response)}, expected {expected}")
    for key, width in ACTION_LAYOUT:
        value = np.asarray(response[key])
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < CHUNK_LENGTH or value.shape[2] != width:
            raise VerifyError(f"{key} has shape {value.shape}, expected (1, >={CHUNK_LENGTH}, {width})")
        if not np.all(np.isfinite(value)):
            raise VerifyError(f"{key} has non-finite values")
    print("get_action OK: " + ", ".join(f"{key} {np.asarray(response[key]).shape}" for key, _ in ACTION_LAYOUT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="exported model folder")
    args = parser.parse_args()
    folder = args.checkpoint.resolve()
    try:
        if not (folder / "config.json").is_file():
            raise VerifyError(f"{folder} has no config.json; is it an exported model folder?")
        print(f"tensors OK: no value_head, dtypes {dict(check_tensors(folder))}", flush=True)
        policy = load_policy(folder)
        print("Gr00tPolicy load OK: no missing, unexpected or newly initialized weights", flush=True)
        check_action(policy, folder)
    except VerifyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"VERIFIED {folder}")


if __name__ == "__main__":
    main()
