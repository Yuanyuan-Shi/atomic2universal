#!/usr/bin/env bash
# Stage 2 RL env: StructRL (RLinf + GR00T-N1.5 + RoboCasa365) per third_party/StructRL/docs/INSTALL.md.
# Order: upstream checkouts -> torch -> StructRL's frozen package list -> flash-attn (source build)
# -> `pip install --no-deps -e` robosuite, robocasa, Isaac-GR00T -> robocasa macros + kitchen assets.
# RLinf is used from source through PYTHONPATH. Idempotent: finished steps are skipped.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash setup/setup_rl.sh --variant blackwell|frozen [--prefix DIR] [--cuda-archs LIST]
                              [--flash-attn-wheel WHL] [--resolve-only] [-h]

  --variant blackwell   StructRL's pins, but torch 2.7.1+cu128 / torchvision 0.22.1+cu128 and the
                        CUDA 12.8 nvidia-*-cu12 libs they pull (sm_100/sm_120: RTX Pro 6000, B200, B300)
  --variant frozen      StructRL's pins unchanged (torch 2.7.1 CUDA 12.6 build; H100/H200 = sm_90)
  --prefix DIR          conda env prefix (default third_party/conda_envs/a2u-rl-blackwell or .../a2u-rl)
  --cuda-archs LIST     flash-attn build archs, e.g. "120" or "90;100" (default: local GPUs, nvidia-smi)
  --flash-attn-wheel W  install this prebuilt flash_attn 2.8.3 wheel instead of building (it must match
                        the variant's torch build and the GPU arch)
  --resolve-only        create the bare Python env and run pip's resolver on the pins (pip --dry-run,
                        flash-attn excluded: source-only); install nothing else
Env overrides: PIP_CACHE_DIR (default third_party/.pip-cache), FLASH_ATTN_BUILD_JOBS.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[setup_rl] $*"; }

VARIANT="" PREFIX="" CUDA_ARCHS="" FA_WHEEL="" RESOLVE_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant|--prefix|--cuda-archs|--flash-attn-wheel)
      [ "$#" -ge 2 ] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; }
      case "$1" in
        --variant) VARIANT="$2" ;; --prefix) PREFIX="$2" ;;
        --cuda-archs) CUDA_ARCHS="$2" ;; --flash-attn-wheel) FA_WHEEL="$2" ;;
      esac
      shift 2 ;;
    --resolve-only) RESOLVE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$VARIANT" in
  blackwell) DEFAULT_ENV=a2u-rl-blackwell; TORCH_CUDA=12.8 ;;
  frozen) DEFAULT_ENV=a2u-rl; TORCH_CUDA=12.6 ;;
  *) echo "ERROR: --variant must be blackwell or frozen" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
S="$TP/StructRL"
PREFIX="$(realpath -m "${PREFIX:-$TP/conda_envs/$DEFAULT_ENV}")"
PY="$PREFIX/bin/python"
PIN_LIST="$S/envs/rlinf-gr00t.txt"
STRUCTRL_COMMIT=6caba1f9538167e0752a7338663e45aac8df965d
declare -A UPSTREAM=([RLinf]=499363965432843cdbae607a561d59b68f785d34
  [robocasa]=8f3c96ec8d1bfcd8126cad2bca887da98d30e997 [robosuite]=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
  [Isaac-GR00T]=4af2b622892f7dcb5aae5a3fb70bcb02dc217b96)
TORCH_PINS=(torch==2.7.1 torchvision==0.22.1)
CU128_INDEX=https://download.pytorch.org/whl/cu128
FLASH_ATTN_VERSION=2.8.3
CUDA_TOOLKIT_PREFIX="$TP/cuda-12.8"   # nvcc + headers, build-time only (shared with setup_policy.sh)
CUDA_TOOLKIT_PACKAGES=("cuda-version=12.8" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-driver-dev cuda-nvrtc-dev
  cuda-nvtx-dev cuda-profiler-api libcublas-dev libcurand-dev libcusolver-dev libcusparse-dev)
STAGE1_ASSETS="$TP/robocasa/robocasa/models/assets"
ASSET_TYPES=(tex tex_generative fixtures_lw objs_objaverse objs_aigen objs_lw)
unset PYTHONPATH  # e.g. ROS Noetic (~/.bashrc) adds /opt/ros/noetic/...; it would leak into pip and imports
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TP/.pip-cache}" TMPDIR="$TP/.pip-tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
pip_() { "$PY" -m pip "$@"; }

command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
[ -f "$PIN_LIST" ] || die "$PIN_LIST missing; clone StructRL at $STRUCTRL_COMMIT into $S first"
[ "$(git -C "$S" rev-parse HEAD)" = "$STRUCTRL_COMMIT" ] || die "$S is not at $STRUCTRL_COMMIT"

# ---- Python env (conda-forge Python 3.11, like StructRL's frozen env) ----
if [ ! -x "$PY" ]; then
  log "creating conda env at $PREFIX (python=3.11)"
  conda create -y -p "$PREFIX" -c conda-forge python=3.11
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || die "$PY is not Python 3.11"

# Variant pin list: flash_attn is installed separately (source-only, built per GPU arch); the
# blackwell variant takes torch, torchvision and their nvidia-*-cu12 libs from the cu128 index.
REQS="$TMPDIR/rlinf-gr00t.$VARIANT.txt"
if [ "$VARIANT" = blackwell ]; then
  grep -v -E '^(flash_attn|torch|torchvision|nvidia-[a-z-]+-cu12)==' "$PIN_LIST" > "$REQS"
else
  grep -v -E '^flash_attn==' "$PIN_LIST" > "$REQS"
fi

# The freeze is not self-consistent (pettingzoo 1.26.1 needs gymnasium>=1.0, lerobot 0.3.3 needs
# gymnasium<1.0, gymnasium==0.29.1 is pinned), so pip's resolver rejects it. It is a complete
# freeze, so it is installed as one: exactly the listed versions, with --no-deps.
if [ "$RESOLVE_ONLY" = 1 ]; then
  extra=(); [ "$VARIANT" = blackwell ] && extra=(--extra-index-url "$CU128_INDEX" torch==2.7.1+cu128 torchvision==0.22.1+cu128)
  log "resolving $(grep -c '==' "$REQS") pins (pip --dry-run --no-deps, nothing installed)"
  pip_ install --dry-run --ignore-installed --no-deps -r "$REQS" "${extra[@]}" | tail -n 3
  log "OK: every pin has a distribution for this Python/platform (--variant $VARIANT)"
  exit 0
fi

# ---- 1. Upstream checkouts (StructRL's own script; adds only git-ignored dirs to StructRL) ----
log "running StructRL scripts/setup_upstream.sh (GR00T-N1.5 + RoboCasa365 only)"
(cd "$S" && PROJECT_ROOT="$S" bash scripts/setup_upstream.sh)
for repo in "${!UPSTREAM[@]}"; do
  [ "$(git -C "$S/$repo" rev-parse HEAD)" = "${UPSTREAM[$repo]}" ] || die "$S/$repo is not at ${UPSTREAM[$repo]}"
done
[ -z "$(git -C "$S" status --porcelain)" ] || die "StructRL working tree is not clean: git -C $S status"

# ---- 2. torch ----
if ! "$PY" -c "import torch, sys; sys.exit(torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '$TORCH_CUDA')" 2>/dev/null; then
  log "installing ${TORCH_PINS[*]} (CUDA $TORCH_CUDA build)"
  if [ "$VARIANT" = blackwell ]; then pip_ install "${TORCH_PINS[@]}" --index-url "$CU128_INDEX"
  else pip_ install "${TORCH_PINS[@]}"; fi
fi

# ---- 3. StructRL's frozen package list ----
log "installing the frozen package list ($PIN_LIST, variant $VARIANT)"
pip_ install --no-deps -r "$REQS"
"$PY" -c "import torch, sys; sys.exit(torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '$TORCH_CUDA')" \
  || die "the package list replaced torch 2.7.1 (CUDA $TORCH_CUDA)"
# The list also holds orphaned CUDA 13 wheels; nvidia-cudnn-cu13 and nvidia-nccl-cu13 write the same
# files as torch's nvidia-cudnn-cu12 / nvidia-nccl-cu12, so installing the list replaces torch's
# cuDNN/NCCL with CUDA 13 builds (they also need driver >= 580: CUDNN_STATUS_NOT_INITIALIZED here).
# Re-install, at the same versions, every cu12 wheel whose libraries no longer match its RECORD.
clobbered="$("$PY" - <<'EOF'
import base64, hashlib, importlib.metadata as md
d = {x.metadata["Name"].lower(): x for x in md.distributions()}
libs = lambda n: [f for f in d[n].files or [] if ".so" in f.name]
others = {str(f) for n in d if n.startswith("nvidia-") and not n.endswith("-cu12") for f in libs(n)}
def intact(f):
    digest = hashlib.sha256(f.locate().read_bytes()).digest()
    return f.hash is None or base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == f.hash.value
print(" ".join(f"{n}=={d[n].version}" for n in d if n.startswith("nvidia-") and n.endswith("-cu12")
               and not all(intact(f) for f in libs(n) if str(f) in others)))
EOF
)"
if [ -n "$clobbered" ]; then
  log "re-installing torch's CUDA 12 libraries overwritten by CUDA 13 wheels: $clobbered"
  idx=(); [ "$VARIANT" = blackwell ] && idx=(--extra-index-url "$CU128_INDEX")
  # shellcheck disable=SC2086
  pip_ install --no-deps --force-reinstall "${idx[@]}" $clobbered
fi

# ---- 4. flash-attn 2.8.3 (PyPI has only the sdist; release wheels need GLIBC >= 2.32) ----
if ! "$PY" -c "import flash_attn" >/dev/null 2>&1; then
  if [ -n "$FA_WHEEL" ]; then
    [ -f "$FA_WHEEL" ] || die "--flash-attn-wheel $FA_WHEEL not found"
    log "installing prebuilt $FA_WHEEL"
    pip_ install --no-deps "$FA_WHEEL"
  else
    if [ -z "$CUDA_ARCHS" ]; then
      command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi; pass --cuda-archs"
      CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
    fi
    [[ "$CUDA_ARCHS" =~ ^[0-9]+(\;[0-9]+)*$ ]] || die "bad --cuda-archs '$CUDA_ARCHS'"
    WHEEL_DIR="$TP/wheels/flash-attn-$FLASH_ATTN_VERSION-$VARIANT-sm${CUDA_ARCHS//;/_}"
    if ! ls "$WHEEL_DIR"/flash_attn-*.whl >/dev/null 2>&1; then
      if [ ! -x "$CUDA_TOOLKIT_PREFIX/bin/nvcc" ]; then
        log "installing CUDA 12.8 compiler + headers into $CUDA_TOOLKIT_PREFIX (build-time only)"
        conda create -y -p "$CUDA_TOOLKIT_PREFIX" -c nvidia/label/cuda-12.8.1 -c conda-forge "${CUDA_TOOLKIT_PACKAGES[@]}"
      fi
      "$CUDA_TOOLKIT_PREFIX/bin/nvcc" --version | grep -q "release 12.8" || die "$CUDA_TOOLKIT_PREFIX is not CUDA 12.8"
      mem_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
      jobs="${FLASH_ATTN_BUILD_JOBS:-$(( mem_gb / 9 < $(nproc) ? mem_gb / 9 : $(nproc) ))}"  # nvcc peaks ~8 GB/job
      [ "$jobs" -ge 1 ] || jobs=1
      log "building flash-attn $FLASH_ATTN_VERSION for sm_$CUDA_ARCHS ($jobs jobs, ~25 min per arch)"
      pip_ install ninja packaging psutil wheel
      # --no-cache-dir: pip's wheel cache ignores FLASH_ATTN_CUDA_ARCHS and the torch build.
      # FORCE_BUILD stops flash-attn's setup.py from fetching the (GLIBC 2.32) release wheel.
      PATH="$PREFIX/bin:$CUDA_TOOLKIT_PREFIX/bin:$PATH" CUDA_HOME="$CUDA_TOOLKIT_PREFIX" \
        CPATH="$CUDA_TOOLKIT_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}" \
        LIBRARY_PATH="$CUDA_TOOLKIT_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCHS" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$jobs" NVCC_THREADS=4 \
        "$PY" -m pip wheel --no-cache-dir --no-build-isolation --no-deps --no-binary flash-attn \
        -w "$WHEEL_DIR" "flash-attn==$FLASH_ATTN_VERSION"
    fi
    pip_ install --no-deps "$WHEEL_DIR"/flash_attn-*.whl
  fi
fi

# ---- 5. Patched upstream checkouts (editable, no deps) ----
for pkg in robosuite:robosuite robocasa:robocasa gr00t:Isaac-GR00T; do
  mod="${pkg%%:*}" dir="$S/${pkg#*:}"
  if ! (cd / && "$PY" -c "import $mod, sys; sys.exit(not $mod.__file__.startswith('$dir/'))") >/dev/null 2>&1; then
    log "installing $dir (editable, --no-deps)"
    pip_ install --no-deps -e "$dir"
  fi
done

# ---- 6. robocasa macros + kitchen assets (robocasa reads only <package>/models/assets) ----
RC_PKG="$S/robocasa/robocasa"
[ -f "$RC_PKG/macros_private.py" ] || (cd / && "$PY" -m robocasa.scripts.setup_macros)
NEW_ASSETS="$RC_PKG/models/assets"
stage1_ok=1
for t in "${ASSET_TYPES[@]}"; do [ -f "$TP/.robocasa_assets_$t.done" ] || stage1_ok=0; done
if [ "$stage1_ok" = 1 ]; then
  log "reusing Stage 1 assets from $STAGE1_ASSETS (symlinks + hard links, no download)"
  for d in textures generative_textures objects; do  # wholly git-ignored in both checkouts
    if [ ! -L "$NEW_ASSETS/$d" ]; then
      [ ! -e "$NEW_ASSETS/$d" ] || die "$NEW_ASSETS/$d exists and is not a symlink"
      ln -s "$STAGE1_ASSETS/$d" "$NEW_ASSETS/$d"
    fi
  done
  # Lightwheel fixtures: hard-link every file that is git-ignored in the Stage 1 checkout, after
  # checking that it is also ignored (never tracked) in the StructRL checkout; never overwrite.
  (cd "$TP/robocasa" && git status --porcelain --ignored robocasa/models/assets/fixtures \
    | awk '$1 == "!!" {print $2}' | xargs -r find | while read -r f; do [ -d "$f" ] || echo "$f"; done) \
    > "$TMPDIR/fixtures_lw.txt"
  [ -s "$TMPDIR/fixtures_lw.txt" ] || die "no git-ignored fixture files found under $STAGE1_ASSETS/fixtures"
  n_files=$(wc -l < "$TMPDIR/fixtures_lw.txt")
  n_ign=$(git -C "$S/robocasa" check-ignore --no-index --stdin < "$TMPDIR/fixtures_lw.txt" | wc -l)
  [ "$n_ign" = "$n_files" ] || die "$((n_files - n_ign)) Stage 1 fixture files are not git-ignored in $S/robocasa"
  n_new=0
  while read -r f; do
    if [ ! -e "$S/robocasa/$f" ]; then
      mkdir -p "$(dirname "$S/robocasa/$f")"
      ln "$TP/robocasa/$f" "$S/robocasa/$f"
      n_new=$((n_new + 1))
    fi
  done < "$TMPDIR/fixtures_lw.txt"
  log "fixtures: $n_files ignored files, $n_new newly hard-linked"
else
  for t in "${ASSET_TYPES[@]}"; do
    mark="$S/robocasa/.a2u_assets_$t.done" dl_log="$S/robocasa/.a2u_assets_$t.log"
    [ -f "$mark" ] && continue
    log "downloading RoboCasa asset type $t (no Stage 1 assets found)"
    (cd / && echo y | "$PY" -m robocasa.scripts.download_kitchen_assets --type "$t") 2>&1 | tee "$dl_log"
    # The official script prints "Failed to download" and still exits 0, so check its log.
    if grep -q "Failed to download" "$dl_log" || ! grep -q "Done." "$dl_log"; then die "asset download $t failed: $dl_log"; fi
    touch "$mark"
  done
fi
[ -z "$(git -C "$S/robocasa" status --porcelain robocasa/models/assets)" ] \
  || die "tracked or unignored asset files changed in $S/robocasa"
conda env config vars set -p "$PREFIX" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl >/dev/null

# ---- 7. Verify on the GPU (StructRL launcher PYTHONPATH, docs/rl_recon.md Q6) ----
log "verifying torch, flash-attn and the RL stack imports"
(cd / && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH="$S:$S/RLinf:$S/robocasa:$S/robosuite:$S/Isaac-GR00T" "$PY" - <<'EOF'
import torch
from flash_attn import flash_attn_func
import rlinf, gr00t, robocasa, robosuite, ray, hydra  # noqa: F401
from rlinf.models.embodiment.gr00t import get_model  # noqa: F401

assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
mean = (x @ x).float().abs().mean().item()
q, k, v = (torch.randn(2, 256, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
err = (flash_attn_func(q, k, v).float() - ref.float()).abs().max().item()
conv = torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 32, 32, device="cuda")).abs().mean().item()  # cuDNN
assert mean == mean and conv == conv and err < 2e-2, (mean, conv, err)
print(f"{torch.__version__} cuda {torch.version.cuda} {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_capability(0)} matmul {mean:.3f} flash-attn max err {err:.1e} "
      f"cudnn {torch.backends.cudnn.version()} nccl {torch.cuda.nccl.version()}")
EOF
)

LOCK="$ROOT/setup/requirements.rl-$VARIANT.lock.txt"
pip_ freeze --exclude-editable > "$LOCK"
{
  echo "# setup/setup_rl.sh --variant $VARIANT; python $("$PY" -c 'import platform; print(platform.python_version())') (conda-forge)"
  echo "# StructRL @ $STRUCTRL_COMMIT; editable --no-deps from its checkouts (patched by setup_upstream.sh):"
  for repo in robosuite robocasa Isaac-GR00T; do echo "#   $repo @ ${UPSTREAM[$repo]}"; done
  echo "#   RLinf @ ${UPSTREAM[RLinf]} via PYTHONPATH (not installed)"
  if [ -n "$FA_WHEEL" ]; then echo "# flash_attn: prebuilt wheel $(basename "$FA_WHEEL") (--flash-attn-wheel)"
  else echo "# flash_attn: built from the PyPI sdist with nvcc 12.8, FLASH_ATTN_CUDA_ARCHS=${CUDA_ARCHS:-<already installed>}"; fi
} >> "$LOCK"
log "OK: $PREFIX ready (lock: ${LOCK#$ROOT/})"
