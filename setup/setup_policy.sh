#!/usr/bin/env bash
# Create the policy env (default: a2u-policy): the RoboCasa365 fork of Isaac-GR00T (GR00T N1.5)
# with a Blackwell-capable PyTorch (cu128) and a matching flash-attn wheel. This env runs the
# official GR00T inference server (scripts/inference_service.py); the simulator lives in a2u-sim.
# Idempotent: re-running skips finished steps.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash setup/setup_policy.sh [-h]

Builds the conda env "${POLICY_ENV:-a2u-policy}" (Python ${POLICY_PYTHON:-3.11}) with the
robocasa-benchmark Isaac-GR00T fork (pinned), torch 2.7.1+cu128 and flash-attn 2.8.3, verifies
CUDA + flash-attn on the GPU, then writes setup/requirements.policy.lock.txt.
Takes no positional arguments. Environment overrides: POLICY_ENV, POLICY_PYTHON.
EOF
}

if [ "$#" -gt 0 ]; then
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then usage; exit 0; fi
  echo "ERROR: unexpected argument(s): $*" >&2
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${POLICY_ENV:-a2u-policy}"
PYTHON_VERSION="${POLICY_PYTHON:-3.11}"
THIRD_PARTY="$ROOT/third_party"

# RoboCasa365's official GR00T N1.5 fork (docs/benchmarking/policy_learning_algorithms.md).
GROOT_URL="https://github.com/robocasa-benchmark/Isaac-GR00T.git"
GROOT_COMMIT="9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"

# Deviation (Blackwell sm_120): the fork's [base] extra pins torch 2.5.1 + flash-attn 2.7.1.post4,
# which have no sm_120 kernels. Use the cu128 build of torch 2.7.1 and flash-attn 2.8.3 compiled
# from source for sm_120: every official flash-attn wheel for torch 2.7 needs GLIBC >= 2.32, but
# Ubuntu 20.04 ships 2.31. nvcc 12.8 comes from NVIDIA's conda channel into third_party/cuda-12.8.
# numpy matches RoboCasa's pin so that arrays pickled by the sim env (torch.save over ZMQ) load
# identically here.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PINS=("torch==2.7.1" "torchvision==0.22.1")
EXTRA_PINS=("numpy==2.2.5" "diffusers==0.30.2" "pyzmq")
FLASH_ATTN_VERSION="2.8.3"
CUDA_TOOLKIT_PREFIX="$THIRD_PARTY/cuda-12.8"
CUDA_TOOLKIT_CHANNEL="nvidia/label/cuda-12.8.1"
CUDA_TOOLKIT_PACKAGES=("cuda-version=12.8" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-driver-dev cuda-nvrtc-dev
  cuda-nvtx-dev cuda-profiler-api libcublas-dev libcurand-dev libcusolver-dev libcusparse-dev)
FLASH_ATTN_BUILD_JOBS="${FLASH_ATTN_BUILD_JOBS:-6}"  # each nvcc job peaks at ~8 GB of RAM

log() { echo "[setup_policy] $*"; }

clone_at() {  # clone_at <url> <dir> <commit>: clone if missing, then check out the pinned commit
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then
    log "cloning $url -> $dir"
    git clone -q "$url" "$dir"
  fi
  if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$dir" fetch -q origin
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    log "checking out $commit in $dir"
    git -C "$dir" checkout -q --detach "$commit"
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Anaconda/Miniconda first." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "creating conda env $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"
set -u

mkdir -p "$THIRD_PARTY"
clone_at "$GROOT_URL" "$THIRD_PARTY/Isaac-GR00T" "$GROOT_COMMIT"

if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.7.1') and torch.version.cuda == '12.8' else 1)" 2>/dev/null; then
  log "installing ${TORCH_PINS[*]} (cu128)"
  pip install "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
fi

# The fork's README installs the core package only (`pip install -e .`); torch is pinned here so
# that resolving gr00t's dependencies never replaces the cu128 build.
if ! python -c "import gr00t, diffusers, zmq" >/dev/null 2>&1; then
  log "installing Isaac-GR00T fork (editable) + ${EXTRA_PINS[*]}"
  pip install -e "$THIRD_PARTY/Isaac-GR00T" "${EXTRA_PINS[@]}" "${TORCH_PINS[@]}" \
    --extra-index-url "$TORCH_INDEX"
fi

if ! python -c "import flash_attn" >/dev/null 2>&1; then
  if [ -x "$CUDA_TOOLKIT_PREFIX/bin/nvcc" ] && ! "$CUDA_TOOLKIT_PREFIX/bin/nvcc" --version | grep -q "release 12.8"; then
    log "removing $CUDA_TOOLKIT_PREFIX (not CUDA 12.8)"
    conda env remove -y -p "$CUDA_TOOLKIT_PREFIX"
  fi
  if [ ! -x "$CUDA_TOOLKIT_PREFIX/bin/nvcc" ]; then
    log "installing CUDA 12.8 compiler + headers into $CUDA_TOOLKIT_PREFIX (build-time only)"
    conda create -y -p "$CUDA_TOOLKIT_PREFIX" -c "$CUDA_TOOLKIT_CHANNEL" -c conda-forge \
      "${CUDA_TOOLKIT_PACKAGES[@]}"
  fi
  log "building flash-attn $FLASH_ATTN_VERSION from source for sm_120 (~30 min, $FLASH_ATTN_BUILD_JOBS jobs)"
  pip install ninja packaging psutil wheel
  pip uninstall -y flash-attn  # drop any wheel that cannot load on this machine
  # FORCE_BUILD skips flash-attn's own attempt to fetch the (GLIBC 2.32) release wheel.
  CUDA_HOME="$CUDA_TOOLKIT_PREFIX" PATH="$CUDA_TOOLKIT_PREFIX/bin:$PATH" \
    CPATH="$CUDA_TOOLKIT_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}" \
    LIBRARY_PATH="$CUDA_TOOLKIT_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    FLASH_ATTN_CUDA_ARCHS="120" FLASH_ATTENTION_FORCE_BUILD="TRUE" \
    MAX_JOBS="$FLASH_ATTN_BUILD_JOBS" NVCC_THREADS="4" \
    pip install --no-build-isolation --no-deps --no-binary flash-attn "flash-attn==$FLASH_ATTN_VERSION"
fi

log "verifying gr00t import, CUDA matmul and a flash-attn kernel on the GPU"
python - <<'EOF'
import torch
from flash_attn import flash_attn_func
from gr00t.model.policy import Gr00tPolicy  # noqa: F401  (imports the N1.5 model + Eagle backbone)
from gr00t.experiment.data_config import DATA_CONFIG_MAP

if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False; check the NVIDIA driver / CUDA_VISIBLE_DEVICES")
device = torch.device("cuda")
name, capability = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
a = torch.randn(1024, 1024, device=device)
matmul_error = (a @ a.T - (a.double() @ a.double().T).float()).abs().max().item()
q, k, v = (torch.randn(2, 128, 8, 64, device=device, dtype=torch.bfloat16) for _ in range(3))
out = flash_attn_func(q, k, v, causal=False)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
flash_error = (out.float() - ref.float()).abs().max().item()
torch.cuda.synchronize()
assert "panda_omron" in DATA_CONFIG_MAP
assert flash_error < 2e-2, f"flash-attn output mismatch vs SDPA: {flash_error}"
print(f"GPU {name} sm_{capability[0]}{capability[1]} torch {torch.__version__} cuda {torch.version.cuda} "
      f"matmul max err {matmul_error:.2e} flash-attn max err {flash_error:.2e}")
EOF

pip freeze --exclude-editable > "$ROOT/setup/requirements.policy.lock.txt"
{
  echo "# editable installs (third_party, pinned commits):"
  echo "# gr00t (Isaac-GR00T robocasa-benchmark fork) @ $GROOT_COMMIT"
  echo "# torch index: $TORCH_INDEX; conda-forge python=$(python -c 'import platform; print(platform.python_version())')"
  echo "# flash_attn built from the PyPI sdist: FLASH_ATTN_CUDA_ARCHS=120, nvcc 12.8 ($CUDA_TOOLKIT_CHANNEL)"
} >> "$ROOT/setup/requirements.policy.lock.txt"
log "OK: policy env '$ENV_NAME' ready (lock: setup/requirements.policy.lock.txt)"
