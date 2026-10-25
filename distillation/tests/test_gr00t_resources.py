"""GPU integration tests of what the GR00T distillation shares and spends (spec §6, tests 2 and 7). Test 2: the three
checkpoints share one backbone and one set of normalization statistics (it only hashes tensors, ≈ 30 s on the CPU, so
it stays out of the fast CPU suite). Test 7: memory and speed of one full `distill_step` at the two-teacher MTD
setting, built by train.py's `Distiller` from configs/distillation/mtd_pnp_cabinet_drawer.yaml: student and fake
network fp32 with AdamW, Expert I1 and Expert I2 bf16 teachers routed on both tasks, the shared bf16 backbone (α_DMD
= α_FM = 0.5, K_ψ = 5, λ_IDA = 0.99), on task-balanced batches of the four round-0 collections, for B = 8, 16, 32.
Per B: WARMUP updates, then over MEASURED updates the median time of backbone encode + distill_step
(`Distiller._update`), of the encode alone, and of the CPU transform (prefetched on a thread in training), the peak
`torch.cuda.max_memory_allocated` and the peak whole-GPU memory used as nvidia-smi reports it (polled every POLL_S);
an OOM is recorded as a result. Then what the trainer keeps on the GPU after training at that B (`Distiller.train`
ends with empty_cache, but fragmented segments stay reserved) while rounds.py runs two policy servers and sims for
collection and selection. Test 7 passes if some B fits with ≥ HEADROOM of the card free in both phases; readings, the
allocator setting (PYTORCH_CUDA_ALLOC_CONF) and the largest such B are printed (run with -s). Skips without CUDA or
without the checkpoints and collections."""

import gc
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from distillation.gr00t import features as features_module
from distillation.gr00t.buffer import ObservationBuffer, collection_dir
from distillation.gr00t.export import CKPT60K
from distillation.gr00t.features import METADATA_FILE, verify_shared_backbone
from distillation.gr00t.rounds import round_jobs
from distillation.gr00t.train import Distiller, load_config, project_path
from distillation.gr00t.wrapper import INDEX_FILE
from distillation.interfaces import DistillConfig
from distillation.tests.test_gr00t_integration import COLLECTIONS, EXPERT_I1, EXPERT_I2, report
from evaluation.config import PROJECT_ROOT

CONFIG = load_config(PROJECT_ROOT / "configs/distillation/mtd_pnp_cabinet_drawer.yaml")
DATA_DIRS = [collection_dir(COLLECTIONS, job.label, job.role, 0) for job in round_jobs(CONFIG, 0)]
TEACHERS = [project_path(teacher["checkpoint"]) for teacher in CONFIG["teachers"]]
FLIPPED_TENSOR = "eagle_model.mlp1.0.bias"  # backbone tensor tampered with in test 2's negative control
BATCH_SIZES, WARMUP, MEASURED, POLL_S = (8, 16, 32), 2, 5, 0.1
HEADROOM = 0.15
SERVERS, SERVER_GIB, SIM_GIB = 2, 6.5, 2.0  # rounds.py runs ≤ 2 server + sim pairs (docs/distill_gr00t_notes.md §9)
GIB = 2 ** 30
MISSING = [str(p) for p in [*DATA_DIRS, project_path(CONFIG["student"]["checkpoint"]), *TEACHERS] if not p.exists()]
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="GR00T integration tests need CUDA"),
              pytest.mark.skipif(bool(MISSING), reason=f"needs the checkpoints and round-0 collections: {MISSING}")]


def test_2_shared_backbone_and_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All three checkpoints share bitwise identical backbone tensors, backbone_cfg and metadata.json (normalization
    statistics): verify_shared_backbone passes. Negative controls, without copying checkpoints: it raises naming
    exactly one tensor when one bit of one expert backbone tensor is flipped (checkpoint_tensors patched), and it
    raises on metadata when one statistic differs (a folder of symlinked shards with an edited metadata.json)."""
    verify_shared_backbone([CKPT60K, EXPERT_I1, EXPERT_I2])
    untouched = features_module.checkpoint_tensors

    def one_bit_flipped(checkpoint: Path, prefix: str, device: str = "cpu"):
        for name, t in untouched(checkpoint, prefix, device):
            if Path(checkpoint) == EXPERT_I1 and name == FLIPPED_TENSOR:
                t = t.clone()
                t.view(torch.int16).view(-1)[0] ^= 1  # lowest mantissa bit of one bf16 element
            yield name, t
    monkeypatch.setattr(features_module, "checkpoint_tensors", one_bit_flipped)
    with pytest.raises(ValueError, match=r"\['tensors'\] differ \(1 backbone tensors, e.g. \['" + FLIPPED_TENSOR):
        verify_shared_backbone([CKPT60K, EXPERT_I1])
    monkeypatch.undo()
    fake = tmp_path / "ckpt"
    (fake / "experiment_cfg").mkdir(parents=True)
    shards = set(json.loads((CKPT60K / INDEX_FILE).read_text())["weight_map"].values())
    for name in ("config.json", INDEX_FILE, *shards):
        (fake / name).symlink_to(CKPT60K / name)
    original = (CKPT60K / METADATA_FILE).read_bytes()
    number = re.search(rb"-?\d+\.\d+", original)  # the first statistic
    (fake / METADATA_FILE).write_bytes(original[:number.start()] + repr(float(number.group()) + 0.1).encode()
                                       + original[number.end():])
    with pytest.raises(ValueError, match=r"\['metadata'\] differ \(0 backbone tensors"):
        verify_shared_backbone([CKPT60K, fake])


def smi_gib(field: str) -> float:
    """nvidia-smi's memory `field` (memory.used, memory.total) of the GPU in GiB (the workstation has one GPU)."""
    out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) / 1024


class SmiPeak:
    """Context manager that polls nvidia-smi's used memory on a thread and keeps the peak (`peak_gib`)."""

    def __init__(self) -> None:
        """Start with no reading."""
        self.peak_gib, self._stop = 0.0, threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self) -> "SmiPeak":
        """Start polling."""
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop polling."""
        self._stop.set()
        self._thread.join()

    def _poll(self) -> None:
        """Update the peak every POLL_S until stopped."""
        while not self._stop.is_set():
            self.peak_gib = max(self.peak_gib, smi_gib("memory.used"))
            self._stop.wait(POLL_S)


def release(distiller: Distiller) -> None:
    """After an OOM: drop gradients and memoized (graph-carrying) backbone outputs, then the cached blocks."""
    for model in (distiller.student, distiller.fake, *distiller.router.teachers.values()):
        model.zero_grad(set_to_none=True)
        model._memo = None
    gc.collect()
    torch.cuda.empty_cache()


def measure(distiller: Distiller, buffer: ObservationBuffer, batch_size: int) -> dict[str, Any]:
    """WARMUP + MEASURED updates on balanced batches of `batch_size`: medians (s) and peaks (GiB) of the measured."""
    rng, times = np.random.default_rng(batch_size), {"transform_s": [], "features_s": [], "step_s": []}
    encode = distiller.features.encode

    def timed_encode(samples: list) -> dict:
        start = time.perf_counter()
        cond = encode(samples)
        torch.cuda.synchronize()
        times["features_s"].append(time.perf_counter() - start)
        return cond
    try:
        with SmiPeak() as smi, mock.patch.object(distiller.features, "encode", timed_encode):
            for k in range(WARMUP + MEASURED):
                if k == WARMUP:
                    torch.cuda.reset_peak_memory_stats()
                    smi.peak_gib, times["features_s"] = 0.0, []
                observations, task_ids = buffer.sample(batch_size, True, rng=rng)
                start = time.perf_counter()
                samples = distiller.features.transform(observations)
                step_start = time.perf_counter()
                distiller._update(samples, task_ids)
                torch.cuda.synchronize()
                if k >= WARMUP:
                    times["transform_s"].append(step_start - start)
                    times["step_s"].append(time.perf_counter() - step_start)
    except torch.OutOfMemoryError:
        release(distiller)
        return {"batch_size": batch_size, "oom": True}
    return {"batch_size": batch_size, "oom": False, **{key: statistics.median(v) for key, v in times.items()},
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB, "smi_peak_gib": smi.peak_gib}


@pytest.fixture(scope="module")
def distiller() -> Distiller:
    """Student, fake, both expert teachers, router, backbone and optimizers of the MTD config."""
    return Distiller(CONFIG)


def resident(row: dict[str, Any]) -> dict[str, Any]:
    """`row` plus the trainer's footprint after training at its B, as rounds.py leaves it for collection and selection
    (Distiller.train ends with empty_cache): allocated, reserved (fragmented segments stay), nvidia-smi used, and that
    plus two policy servers and sims."""
    gc.collect()
    torch.cuda.empty_cache()
    used = smi_gib("memory.used")
    return {**row, "resident_allocated_gib": torch.cuda.memory_allocated() / GIB,
            "resident_reserved_gib": torch.cuda.memory_reserved() / GIB, "resident_smi_gib": used,
            "with_servers_gib": used + SERVERS * (SERVER_GIB + SIM_GIB)}


def test_7_memory_and_speed(distiller: Distiller) -> None:
    """Measure B = 8, 16, 32 in increasing order (see the module docstring); some B must fit with ≥ 15% headroom both in
    training (nvidia-smi peak) and afterwards, when the trainer's resident memory shares the GPU with two policy servers
    and sims."""
    assert distiller.dcfg == DistillConfig(alpha_dmd=0.5, alpha_fm=0.5, k_fake=5, lambda_ida=0.99, num_flow_steps=4)
    buffer, limit = ObservationBuffer(DATA_DIRS), (1 - HEADROOM) * smi_gib("memory.total")
    print(f"\n[phase C] test 7 PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(default)')}")
    fits = []
    for batch_size in BATCH_SIZES:
        row = resident(measure(distiller, buffer, batch_size))
        report(f"test 7 B={batch_size}", **{key: v for key, v in row.items() if key != "batch_size"})
        if not row["oom"] and max(row["smi_peak_gib"], row["with_servers_gib"]) <= limit:
            fits.append(batch_size)
    report("test 7", limit_gib=limit, largest_fitting_batch=max(fits, default=0))
    assert fits
