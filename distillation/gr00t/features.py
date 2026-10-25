"""The frozen GR00T backbone shared by all teachers, students and fake networks, and the `cond` it produces for
`GR00TVelocity`. `BackboneFeatures` first checks that every checkpoint in use has bitwise identical `backbone.*`
tensors (sha256 of the raw bytes, one tensor at a time through `safe_open`), the same `backbone_cfg` and a
byte-identical `experiment_cfg/metadata.json` (recon §4-5), then builds the Eagle backbone once (bf16, eval, frozen;
only `backbone.*` tensors are read) and the `panda_omron` transforms in eval mode with that metadata, as
`Gr00tPolicy` does (POL:97, 278-296). On a batch of raw observations it runs the transforms per observation on the
CPU (`transform`, in `Policy.act`'s request layout) and the backbone on the GPU per group of equal token length T
(`encode`), so each sample's features equal the B = 1 eval computation; groups are then left padded to the batch's
longest T, the fork tokenizer's layout (TR:47-52). Paths under third_party/Isaac-GR00T/gr00t/model/:
POL = policy.py, N15 = gr00t_n1.py, TR = transforms.py, BB = backbone/eagle_backbone.py."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.backbone import EagleBackbone
from torch import Tensor

from distillation.gr00t.wrapper import checkpoint_tensors
from evaluation.policy import ACTION_LAYOUT, LANGUAGE_KEY
from expert_training.verify_expert import DATA_CONFIG, EMBODIMENT

ACTION_HORIZON, ACTION_DIM = 16, 32  # config.json action_head_cfg: action_horizon, action_dim (padded, recon §3)
REAL_ACTION_DIM = sum(width for _, width in ACTION_LAYOUT)  # 12: the concatenated panda_omron action keys
METADATA_FILE = "experiment_cfg/metadata.json"
RawObs = dict[str, np.ndarray | str]


def action_mask(batch_size: int, device: str | torch.device) -> Tensor:
    """Bool [B, 16, 32] mask of the real action dims (the first 12), as GR00TTransform builds it (TR:292-297)."""
    mask = torch.zeros(batch_size, ACTION_HORIZON, ACTION_DIM, dtype=torch.bool, device=device)
    mask[..., :REAL_ACTION_DIM] = True
    return mask


def _fingerprint(checkpoint: Path) -> dict[str, object]:
    """What checkpoints sharing one backbone must agree on: metadata bytes, `backbone_cfg`, and per backbone tensor
    its dtype, shape and the sha256 of its raw bytes."""
    tensors = {name: (str(t.dtype), tuple(t.shape), hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).digest())
               for name, t in checkpoint_tensors(checkpoint, "backbone.")}
    return {"metadata": (checkpoint / METADATA_FILE).read_bytes(), "tensors": tensors,
            "backbone_cfg": json.loads((checkpoint / "config.json").read_text())["backbone_cfg"]}


def verify_shared_backbone(checkpoints: Sequence[Path]) -> None:
    """Raise ValueError unless all checkpoints share the first one's fingerprint (bitwise identical backbone)."""
    reference, *others = [Path(c) for c in checkpoints]
    expected = _fingerprint(reference)
    for checkpoint in others:
        found = _fingerprint(checkpoint)
        differ = [key for key in expected if found[key] != expected[key]]
        if differ:
            names = expected["tensors"].keys() | found["tensors"].keys()
            tensors = sorted(n for n in names if expected["tensors"].get(n) != found["tensors"].get(n))
            raise ValueError(f"{checkpoint} does not share the backbone of {reference}: {differ} differ "
                             f"({len(tensors)} backbone tensors, e.g. {tensors[:3]})")


def _load_backbone(checkpoint: Path, device: torch.device) -> EagleBackbone:
    """EagleBackbone from config.json's `backbone_cfg` (as N15:81) with empty weights, then only the checkpoint's
    `backbone.*` tensors (strict), bf16, eval, frozen."""
    config = json.loads((checkpoint / "config.json").read_text())["backbone_cfg"]
    with init_empty_weights():
        backbone = EagleBackbone(**config)
    state = {name: t.to(torch.bfloat16) for name, t in checkpoint_tensors(checkpoint, "backbone.", device)}
    backbone.load_state_dict(state, strict=True, assign=True)
    return backbone.to(device).eval().requires_grad_(False)


def _eval_transform(checkpoint: Path) -> ComposedModalityTransform:
    """The panda_omron transforms in eval mode with the checkpoint's statistics (POL:97, 281-295)."""
    transform = DATA_CONFIG_MAP[DATA_CONFIG].transform()
    transform.eval()
    metadata = json.loads((checkpoint / METADATA_FILE).read_text())[EMBODIMENT]
    transform.set_metadata(DatasetMetadata.model_validate(metadata))
    return transform


def _request(observation: RawObs) -> dict[str, np.ndarray]:
    """One raw observation in `Policy.act`'s request layout (evaluation/policy.py:74-81): video.* and state.* as
    [1, 1, ...] arrays, the instruction as a one-element array."""
    request = {key: np.asarray(value)[None, None] for key, value in observation.items()
               if key.startswith(("video.", "state."))}
    request[LANGUAGE_KEY] = np.array([observation[LANGUAGE_KEY]])
    return request


class BackboneFeatures:
    """Shared frozen backbone + eval transforms: raw observations → `cond` for `GR00TVelocity`."""

    def __init__(self, checkpoints: Sequence[Path], device: str | torch.device) -> None:
        """Verify that the checkpoints share one backbone and metadata, then load it once with the transforms."""
        unique = list(dict.fromkeys(Path(c).resolve() for c in checkpoints))
        verify_shared_backbone(unique)
        self.device = torch.device(device)
        self.backbone = _load_backbone(unique[0], self.device)
        self.eval_transform = _eval_transform(unique[0])

    def __call__(self, observations: Sequence[RawObs]) -> dict[str, Tensor]:
        """`cond` of a batch of raw observations: `encode(transform(observations))`."""
        return self.encode(self.transform(observations))

    def transform(self, observations: Sequence[RawObs]) -> list[dict[str, Tensor]]:
        """CPU part: the eval transforms on each observation alone (B = 1, so nothing is padded)."""
        return [self.eval_transform(_request(observation)) for observation in observations]

    @torch.no_grad()
    def encode(self, samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        """GPU part: the backbone per group of equal token length T, scattered into tensors left padded to the
        longest T: `backbone_features` [B, T, 2048] bf16, `backbone_attention_mask` [B, T] bool, `state`
        [B, 1, 64] fp32, `embodiment_id` [B] int64 and `action_mask` [B, 16, 32] bool, all on the device."""
        lengths = [sample["eagle_input_ids"].shape[1] for sample in samples]
        groups = {length: [b for b, n in enumerate(lengths) if n == length] for length in sorted(set(lengths))}
        outputs = {length: self._backbone([samples[b] for b in idx]) for length, idx in groups.items()}
        first = next(iter(outputs.values()))["backbone_features"]
        features = first.new_zeros(len(samples), max(lengths), first.shape[-1])
        mask = torch.zeros(len(samples), max(lengths), dtype=torch.bool, device=self.device)
        for length, idx in groups.items():
            features[idx, -length:] = outputs[length]["backbone_features"]
            mask[idx, -length:] = outputs[length]["backbone_attention_mask"].bool()
        return {"backbone_features": features, "backbone_attention_mask": mask,
                "state": torch.cat([s["state"] for s in samples]).to(self.device, torch.float32),
                "embodiment_id": torch.cat([s["embodiment_id"] for s in samples]).to(self.device),
                "action_mask": action_mask(len(samples), self.device)}

    def _backbone(self, group: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Backbone output (BB:115-133) of samples with equal token length: eagle inputs concatenated, moved and
        cast to bf16 as `prepare_input` does for the eval's bf16 model (N15:187-196), bf16 autocast (POL:190)."""
        batch = {key: torch.cat([sample[key] for sample in group]) for key in group[0] if key.startswith("eagle_")}
        inputs = {key: value.to(self.device, torch.bfloat16) if value.is_floating_point() else value.to(self.device)
                  for key, value in batch.items()}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            return self.backbone(inputs)
