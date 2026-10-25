"""Export a student action head with the shared backbone as an eval-ready GR00T folder, in the format of the experts
(`expert_training/export_expert.py`, recon §6): config.json, the safetensors index and experiment_cfg/ are copied from
the base checkpoint; both shards are rewritten as the index's weight_map lists them, every tensor bf16 (the backbone
copied from the base, the action head cast from the student); `expert_training/verify_expert.py` then checks the
folder (bf16 only, no value head, the eval's `Gr00tPolicy` loads it without weight mismatches and returns a valid
chunk). Only after that is `export_info.json` written and `<out>.partial` renamed to `<out>`, so `<out>` exists only
as a complete, verified export, and it is never overwritten.
CLI (a2u-policy, GPU, from the project root), e.g. Level 1's perturbed copy of checkpoint-60000:
    python -m distillation.gr00t.export --source <ckpt> --out <dir> [--perturb-scale s --perturb-seed n]"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from distillation.gr00t.wrapper import INDEX_FILE, GR00TVelocity, checkpoint_tensors, perturb_

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT60K = (PROJECT_ROOT / "checkpoints/sft_joint/gr00t_n1-5/foundation_model_learning/target_posttraining/"
           "atomic_seen/checkpoint-60000")


def _export_tensors(student: GR00TVelocity, base: Path) -> dict[str, Tensor]:
    """Every tensor of the export by full name, bf16 on the CPU: the base's backbone and the student's head."""
    tensors = {f"backbone.{name}": t.to(torch.bfloat16) for name, t in checkpoint_tensors(base, "backbone.")}
    for name, t in student.action_head.state_dict().items():
        tensors[f"action_head.{name}"] = t.detach().to(torch.bfloat16).cpu().contiguous()
    return tensors


def _write_shards(tensors: dict[str, Tensor], base: Path, folder: Path) -> None:
    """Write each shard of the base index's weight_map, as StructRL's convert_rl_fullweights_to_hf.py:28-31 does;
    raise unless the tensor names are exactly the index's."""
    weight_map = json.loads((base / INDEX_FILE).read_text())["weight_map"]
    missing, extra = sorted(weight_map.keys() - tensors.keys()), sorted(tensors.keys() - weight_map.keys())
    if missing or extra:
        raise ValueError(f"export tensors differ from {base / INDEX_FILE}: missing {missing[:3]}, extra {extra[:3]}")
    for shard in sorted(set(weight_map.values())):
        save_file({name: tensors[name] for name, s in weight_map.items() if s == shard}, str(folder / shard),
                  metadata={"format": "pt"})


def _verify(folder: Path) -> None:
    """Run `python -m expert_training.verify_expert --checkpoint <folder>` in this interpreter's env; its own
    process loads the policy, so its GPU and host memory are freed when it exits."""
    command = [sys.executable, "-m", "expert_training.verify_expert", "--checkpoint", str(folder)]
    if subprocess.run(command, cwd=PROJECT_ROOT).returncode != 0:
        raise RuntimeError(f"verify_expert rejected {folder}; see its output above")


def export_student(student: GR00TVelocity, out_dir: Path, info: dict, base_checkpoint: Path = CKPT60K) -> Path:
    """Export `student` (bf16) with `base_checkpoint`'s backbone and configs to `out_dir`, verify it, record `info`
    in `export_info.json` and return `out_dir`; raises FileExistsError if `out_dir` exists (a complete export)."""
    out_dir, base = Path(out_dir), Path(base_checkpoint)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists (a complete export); delete it to export again")
    partial = out_dir.with_name(f"{out_dir.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for name in ("config.json", INDEX_FILE):
        shutil.copy2(base / name, partial / name)
    shutil.copytree(base / "experiment_cfg", partial / "experiment_cfg")
    _write_shards(_export_tensors(student, base), base, partial)
    _verify(partial)
    record = {**info, "base_checkpoint": str(base.resolve()), "exported": f"{datetime.now():%Y-%m-%d %H:%M}"}
    (partial / "export_info.json").write_text(json.dumps(record, indent=2) + "\n")
    partial.rename(out_dir)
    return out_dir


def main() -> None:
    """CLI: export a copy of `--source` (its own backbone and configs), optionally perturbed with `perturb_`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="checkpoint folder to copy")
    parser.add_argument("--out", type=Path, required=True, help="export folder (must not exist)")
    parser.add_argument("--perturb-scale", type=float, default=0.0, help="p += scale·std(p)·N(0, 1); 0 = exact copy")
    parser.add_argument("--perturb-seed", type=int, default=0, help="seed of the perturbation noise")
    parser.add_argument("--device", default="cuda", help="device that holds the fp32 action head")
    args = parser.parse_args()
    if args.perturb_scale < 0:
        parser.error(f"--perturb-scale must be >= 0, got {args.perturb_scale}")
    student = GR00TVelocity.from_checkpoint(args.source, torch.float32, args.device, trainable=True)
    perturbation = None
    if args.perturb_scale > 0:
        perturb_(student, args.perturb_scale, args.perturb_seed)
        perturbation = {"scale": args.perturb_scale, "seed": args.perturb_seed}
    info = {"source": str(args.source.resolve()), "perturbation": perturbation}
    print(f"Exported {export_student(student, args.out, info, base_checkpoint=args.source)}")


if __name__ == "__main__":
    main()
