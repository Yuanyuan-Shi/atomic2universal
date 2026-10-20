"""Make the step-0 control expert: checkpoint-60000 pushed through the RL pipeline with no training.

Builds the run config with build_config.build (so the model section is identical to a real run),
loads the actor model exactly as RLinf's actor worker does (rlinf.models.get_model on
cfg.actor.model, which casts to bf16 and initializes the value head; RLinf's load_to_device
switch keeps it on the CPU, which leaves the weights unchanged), and saves its state dict where
and as RLinf's FSDP full-state-dict save does (hybrid_engines/fsdp/strategy/base.py:184-266):
<run_dir>/checkpoints/global_step_0/actor/model_state_dict/full_weights.pt, a plain dict of the
module's parameter and buffer names to CPU tensors, written with torch.save's zip format.
`export_expert.py --step 0` then exports it like any trained step (StructRL's converter asserts
the key set). StructRL's RLinf and Isaac-GR00T are put first on sys.path, as in train_expert.sh.
CUDA must be visible (the Eagle backbone queries it at import); no GPU memory is used.

Usage (RL env, project root): python expert_training/make_step0.py --task <Task> --hw <workstation|node8>
    [--run-dir DIR]   (default: expert_training/runs/<Task>/step0-control)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from build_config import ERRORS, RUNS_ROOT, STRUCTRL_ROOT, BuildError, build, hw_choices
from raw_checkpoints import full_weights

STRUCTRL_PATHS = [STRUCTRL_ROOT / name for name in ("", "RLinf", "robocasa", "robosuite", "Isaac-GR00T")]
sys.path[:0] = [str(path) for path in STRUCTRL_PATHS]  # train_expert.sh's PYTHONPATH, minus the env-only seed hook

import torch  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402


def check_origin(module_name: str) -> None:
    origin = Path(sys.modules[module_name].__file__).resolve()
    if not origin.is_relative_to(STRUCTRL_ROOT):
        raise BuildError(f"{module_name} was imported from {origin}, not from StructRL's checkout {STRUCTRL_ROOT}")


def make_step0(task: str, hw: str, run_dir: Path) -> Path:
    """Write the untrained actor's full_weights.pt as RLinf's global_step_0 checkpoint of run_dir."""
    run_dir = run_dir.resolve()
    target = full_weights(run_dir / "checkpoints" / "global_step_0")
    if target.parent.parent.parent.exists():
        raise BuildError(f"{target.parent.parent.parent} exists; delete {run_dir} to make the step-0 control again")
    cfg = OmegaConf.load(build(task, hw, run_dir))
    model_cfg = cfg.actor.model
    with open_dict(model_cfg):
        model_cfg.load_to_device = False  # rlinf.models.get_model: skip model.to(device)
    from rlinf.models import get_model

    model = get_model(model_cfg)
    for module_name in ("rlinf", "gr00t"):
        check_origin(module_name)
    state_dict = dict(model.state_dict())  # the FQNs FSDP's get_model_state_dict(full_state_dict=True) returns
    dtypes = Counter(str(tensor.dtype) for tensor in state_dict.values())
    value_head = sum(1 for name in state_dict if ".value_head." in name)
    print(f"actor model {type(model).__name__}: {len(state_dict)} tensors ({value_head} value_head), "
          f"{sum(t.numel() for t in state_dict.values()) / 1e9:.3f} B values, dtypes {dict(dtypes)}", flush=True)
    target.parent.mkdir(parents=True)
    partial = target.with_name(f"{target.name}.partial")
    torch.save(state_dict, partial, _use_new_zipfile_serialization=True)
    partial.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="atomic_seen task, e.g. PickPlaceCounterToCabinet")
    parser.add_argument("--hw", required=True, help=f"hardware file configs/rl/hw_<hw>.yaml: {hw_choices()}")
    parser.add_argument("--run-dir", type=Path, help="default: expert_training/runs/<task>/step0-control")
    args = parser.parse_args()
    run_dir = args.run_dir or RUNS_ROOT / args.task / "step0-control"
    try:
        path = make_step0(args.task, args.hw, run_dir)
    except ERRORS as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {path}\nExport it with: python expert_training/export_expert.py --task {args.task} "
          f"--run-dir {run_dir.resolve()} --step 0")


if __name__ == "__main__":
    main()
