#!/usr/bin/env bash
# Create the simulation env (default: a2u-sim): robosuite (master, pinned) + RoboCasa365 (pinned)
# + kitchen assets + ffmpeg/ffprobe + the light GR00T ZMQ client (gr00t.eval.service, no deps).
# The GR00T policy itself runs in a separate env built by setup/setup_policy.sh (see
# docs/environment.md for why two envs are used). Idempotent: re-running skips finished steps.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash setup/setup_sim.sh [-h]

Builds the conda env "${SIM_ENV:-a2u-sim}" (Python ${SIM_PYTHON:-3.11}) with robosuite, RoboCasa365,
the RoboCasa kitchen assets (~10 GB, stored under third_party/robocasa), ffmpeg and the GR00T
ZMQ client, then writes setup/requirements.sim.lock.txt. Takes no positional arguments.
Environment overrides: SIM_ENV, SIM_PYTHON.
EOF
}

if [ "$#" -gt 0 ]; then
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then usage; exit 0; fi
  echo "ERROR: unexpected argument(s): $*" >&2
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${SIM_ENV:-a2u-sim}"
PYTHON_VERSION="${SIM_PYTHON:-3.11}"
THIRD_PARTY="$ROOT/third_party"

# Pinned refs (see docs/environment.md). RoboCasa365 requires robosuite master, not a release tag.
ROBOSUITE_URL="https://github.com/ARISE-Initiative/robosuite.git"
ROBOSUITE_COMMIT="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
ROBOCASA_URL="https://github.com/robocasa/robocasa.git"
ROBOCASA_COMMIT="a9e152e2eea16d342dac40e4577fb4c68b932854"
GROOT_URL="https://github.com/robocasa-benchmark/Isaac-GR00T.git"
GROOT_COMMIT="9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"

# Blackwell (sm_120) needs CUDA >= 12.8 wheels. torch 2.7.1 also satisfies lerobot==0.3.3
# (a RoboCasa dependency: torch<2.8, torchvision<0.23) and matches the policy env build.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PINS=("torch==2.7.1" "torchvision==0.22.1")

ASSET_TYPES=(tex tex_generative fixtures_lw objs_objaverse objs_aigen objs_lw)

log() { echo "[setup_sim] $*"; }

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
# conda's activate scripts reference unset variables, so relax -u around them.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "creating conda env $ENV_NAME (python=$PYTHON_VERSION, ffmpeg from conda-forge)"
  conda create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" ffmpeg
fi
conda activate "$ENV_NAME"
set -u

if ! command -v ffprobe >/dev/null 2>&1; then
  log "installing ffmpeg (ffprobe) from conda-forge"
  conda install -y -n "$ENV_NAME" -c conda-forge ffmpeg
fi

# Rendering through EGL on a headless NVIDIA GPU; stored in the env so `conda run` picks it up.
if ! conda env config vars list -n "$ENV_NAME" | grep -q "MUJOCO_GL"; then
  conda env config vars set -n "$ENV_NAME" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl >/dev/null
  set +u; conda deactivate; conda activate "$ENV_NAME"; set -u
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

mkdir -p "$THIRD_PARTY"
clone_at "$ROBOSUITE_URL" "$THIRD_PARTY/robosuite" "$ROBOSUITE_COMMIT"
clone_at "$ROBOCASA_URL" "$THIRD_PARTY/robocasa" "$ROBOCASA_COMMIT"
clone_at "$GROOT_URL" "$THIRD_PARTY/Isaac-GR00T" "$GROOT_COMMIT"

if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.7.1') and torch.version.cuda == '12.8' else 1)" 2>/dev/null; then
  log "installing ${TORCH_PINS[*]} (cu128)"
  pip install "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
fi

if ! python -c "import robosuite, robocasa" >/dev/null 2>&1; then
  log "installing robosuite and robocasa (editable)"
  pip install -e "$THIRD_PARTY/robosuite"
  pip install -e "$THIRD_PARTY/robocasa" "${TORCH_PINS[@]}" --extra-index-url "$TORCH_INDEX"
fi

ROBOCASA_PKG="$THIRD_PARTY/robocasa/robocasa"
if [ ! -f "$ROBOCASA_PKG/macros_private.py" ]; then
  python -m robocasa.scripts.setup_macros
fi

# Kitchen assets (~10 GB). The official script asks for confirmation, so answer "y" once.
# A marker per asset type makes re-runs skip completed downloads.
missing_types=()
for asset_type in "${ASSET_TYPES[@]}"; do
  [ -f "$THIRD_PARTY/.robocasa_assets_${asset_type}.done" ] || missing_types+=("$asset_type")
done
if [ "${#missing_types[@]}" -gt 0 ]; then
  for asset_type in "${missing_types[@]}"; do
    log "downloading RoboCasa asset type: $asset_type"
    download_log="$THIRD_PARTY/.robocasa_assets_${asset_type}.log"
    echo y | python -m robocasa.scripts.download_kitchen_assets --type "$asset_type" 2>&1 \
      | tee "$download_log"
    # The official script prints "Failed to download" and exits 0 on failure, so check its log.
    if grep -q "Failed to download" "$download_log" || ! grep -q "Done." "$download_log"; then
      echo "ERROR: asset download '$asset_type' failed; see $download_log and retry." >&2
      exit 1
    fi
    touch "$THIRD_PARTY/.robocasa_assets_${asset_type}.done"
  done
fi

# Video writing + configs + GR00T ZMQ client transport (pyzmq). gr00t is installed without its
# dependencies: the sim env only imports gr00t.eval.service (needs torch + zmq), never the model.
pip install "imageio[ffmpeg]" pyyaml pyzmq
if ! pip show gr00t >/dev/null 2>&1; then
  pip install --no-deps -e "$THIRD_PARTY/Isaac-GR00T"
fi

log "verifying imports"
python - <<'EOF'
import mujoco, numpy, robosuite, robocasa, torch, zmq
from gr00t.eval.service import ExternalRobotInferenceClient  # noqa: F401
print(f"mujoco {mujoco.__version__} numpy {numpy.__version__} robosuite {robosuite.__version__} "
      f"robocasa {robocasa.__version__} torch {torch.__version__}")
EOF
ffprobe -version | head -n 1

pip freeze --exclude-editable > "$ROOT/setup/requirements.sim.lock.txt"
{
  echo "# editable installs (third_party, pinned commits):"
  echo "# robosuite @ $ROBOSUITE_COMMIT"
  echo "# robocasa @ $ROBOCASA_COMMIT"
  echo "# gr00t (Isaac-GR00T robocasa-benchmark fork, --no-deps) @ $GROOT_COMMIT"
  echo "# conda-forge: python=$(python -c 'import platform; print(platform.python_version())'), $(ffprobe -version | head -n 1 | cut -d' ' -f1-3)"
} >> "$ROOT/setup/requirements.sim.lock.txt"
log "OK: sim env '$ENV_NAME' ready (lock: setup/requirements.sim.lock.txt)"
