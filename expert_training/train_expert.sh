#!/usr/bin/env bash
# Train one single-task PPO expert (StructRL's GR00T-N1.5 Sparse-RL recipe on RLinf, one node, no Slurm):
# build <run_dir>/resolved_config.yaml (build_config.py), launch RLinf's train_embodied_agent.py on exactly
# that file in its own process group, tee everything to <run_dir>/train.log, prune old raw checkpoints every
# 10 min and at exit (raw_checkpoints.py); any exit stops the group, incl. the local Ray cluster RLinf starts.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash expert_training/train_expert.sh --task <Task> --hw <workstation|aws_node>
                                            [--run-id ID] [--resume] [--env-prefix DIR]
  --task        atomic_seen task, e.g. PickPlaceCounterToCabinet
  --hw          hardware file configs/rl/hw_<hw>.yaml
  --run-id      run directory expert_training/runs/<Task>/<ID> (default: <YYYYmmdd-HHMM>-<hw>)
  --resume      continue from the newest complete raw checkpoint of the run named by --run-id,
                else of the run of <Task> whose train.log changed last
  --env-prefix  RL conda env (default: third_party/conda_envs/a2u-rl-blackwell if present, else a2u-rl)
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARGS="$*"
TASK="" HW="" RUN_ID="" RESUME=0 ENV_PREFIX="" TRAIN_PID="" PRUNE_PID=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --task|--hw|--run-id|--env-prefix)
      if [ "$#" -lt 2 ]; then echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; fi
      case "$1" in --task) TASK="$2" ;; --hw) HW="$2" ;; --run-id) RUN_ID="$2" ;; *) ENV_PREFIX="$2" ;; esac
      shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
HW_FILE="$ROOT/configs/rl/hw_$HW.yaml"
if [ -z "$TASK" ] || [ ! -f "$HW_FILE" ]; then
  echo "ERROR: --task and --hw are required, and configs/rl/hw_<hw>.yaml must exist (got --hw '$HW')" >&2; usage >&2; exit 2
fi
for prefix in "$ROOT/third_party/conda_envs/a2u-rl-blackwell" "$ROOT/third_party/conda_envs/a2u-rl"; do
  if [ -z "$ENV_PREFIX" ] && [ -x "$prefix/bin/python" ]; then ENV_PREFIX="$prefix"; fi
done
if [ -z "$ENV_PREFIX" ] || [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "ERROR: no RL env '${ENV_PREFIX}' (default: a2u-rl-blackwell, else a2u-rl in third_party/conda_envs)." \
       "Run setup/setup_rl.sh or pass --env-prefix." >&2; exit 1
fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[ -n "$CONDA" ] || { echo "ERROR: conda not found. Activate conda or set CONDA_EXE." >&2; exit 1; }

# StructRL's launcher environment (docs/rl_recon.md Q6). PYTHONPATH is replaced, not appended
# (ROS Noetic in ~/.bashrc sets it), with the scene-seed hook (seed_hook/sitecustomize.py) first.
ST="$ROOT/third_party/StructRL"
export PROJECT_ROOT="$ST" REPO_PATH="$ST/RLinf" EMBODIED_PATH="$ST/RLinf/examples/embodiment" STRUCTRL_PROJECT_ROOT="$ST"
export STRUCTRL_TIMING_JSON="$ST/structrl/demo_timing_composite_seen.json" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/expert_training/seed_hook:$ST:$ST/RLinf:$ST/robocasa:$ST/robosuite:$ST/Isaac-GR00T"
unset RAY_ADDRESS
PY=("$CONDA" run --no-capture-output -p "$ENV_PREFIX" python -u)

# Preflight: visible GPUs match the hardware file; no Ray cluster that RLinf's ray.init(address="auto") would join.
GPUS="$(sed -n 's/^gpus:[[:space:]]*\([0-9]*\).*/\1/p' "$HW_FILE")"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then VISIBLE="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)"
else VISIBLE="$(nvidia-smi -L | grep -c '^GPU ')"; fi
if [ "$VISIBLE" != "$GPUS" ]; then
  echo "ERROR: $HW_FILE has gpus: $GPUS but $VISIBLE GPU(s) are visible; set CUDA_VISIBLE_DEVICES or pick another --hw" >&2; exit 1
fi
if pgrep -x gcs_server >/dev/null; then
  echo "ERROR: a Ray cluster is already running here (gcs_server pid $(pgrep -x gcs_server | tr '\n' ' '));" \
       "RLinf would join it instead of starting its own. Stop it if it is yours (ray stop) or wait." >&2; exit 1
fi
HOOK_CHECK="$("${PY[@]}" "$ROOT/expert_training/seed_hook/sitecustomize.py" 2>&1)" || { echo "$HOOK_CHECK" >&2; exit 1; }

RUNS="$ROOT/expert_training/runs/$TASK"
BUILD_ARGS=(--task "$TASK" --hw "$HW")
if [ "$RESUME" = 1 ]; then
  RUN_DIR="$RUNS/$RUN_ID"
  if [ -z "$RUN_ID" ]; then LATEST="$(ls -t "$RUNS"/*/train.log 2>/dev/null | head -n 1 || true)"; RUN_DIR="${LATEST%/train.log}"; fi
  if [ ! -f "$RUN_DIR/train.log" ]; then echo "ERROR: no run of $TASK to resume (no train.log in '$RUN_DIR')" >&2; exit 1; fi
  RESUME_DIR="$("${PY[@]}" "$ROOT/expert_training/raw_checkpoints.py" resume-dir "$RUN_DIR")"
  BUILD_ARGS+=(--resume-dir "$RESUME_DIR")
else
  RUN_DIR="$RUNS/${RUN_ID:-$(date +%Y%m%d-%H%M)-$HW}"
  if [ -e "$RUN_DIR" ]; then echo "ERROR: $RUN_DIR exists; pick another --run-id or pass --resume" >&2; exit 1; fi
fi
BUILD_OUTPUT="$("${PY[@]}" "$ROOT/expert_training/build_config.py" "${BUILD_ARGS[@]}" --run-dir "$RUN_DIR" 2>&1)" \
  || { echo "$BUILD_OUTPUT" >&2; exit 1; }
LOG="$RUN_DIR/train.log"
{ echo "===== $(date '+%F %T') train_expert.sh $ARGS | env $ENV_PREFIX | host $(hostname)"
  echo "$HOOK_CHECK"; echo "$BUILD_OUTPUT"; } | tee -a "$LOG"

prune() {
  "${PY[@]}" "$ROOT/expert_training/raw_checkpoints.py" prune "$RUN_DIR" 2>&1 | tee -a "$LOG" \
    || echo "WARNING: pruning raw checkpoints failed (see above); disk use keeps growing" | tee -a "$LOG" >&2
}
cleanup() {
  local status=$?
  trap - EXIT; trap '' INT TERM HUP  # a second Ctrl-C must not abort the cleanup (bounded: <= 60 s + prune)
  if [ -n "$PRUNE_PID" ]; then kill "$PRUNE_PID" 2>/dev/null || true; wait "$PRUNE_PID" 2>/dev/null || true; fi
  if [ -n "$TRAIN_PID" ] && kill -0 -- "-$TRAIN_PID" 2>/dev/null; then
    echo "Stopping process group $TRAIN_PID (RLinf driver, local Ray, env workers); wait <= 60 s" | tee -a "$LOG" >&2
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
    ( sleep 60; kill -KILL -- "-$TRAIN_PID" 2>/dev/null ) &  # escalate if anything ignores SIGTERM
    local killer=$!
    wait "$TRAIN_PID" 2>/dev/null || true  # reap the leader, so the group check below sees only live processes
    while kill -0 -- "-$TRAIN_PID" 2>/dev/null && kill -0 "$killer" 2>/dev/null; do sleep 1; done
    kill -KILL "$killer" 2>/dev/null || true  # it inherited the ignored SIGTERM
  fi
  prune
  echo "Run directory: $RUN_DIR (exit $status)" | tee -a "$LOG"
  exit "$status"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP

cd "$RUN_DIR"  # MuJoCo (MUJOCO_LOG.TXT) and other stray outputs land in the run directory
setsid "${PY[@]}" "$EMBODIED_PATH/train_embodied_agent.py" --config-path "$RUN_DIR" --config-name resolved_config \
  > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1 < /dev/null &
TRAIN_PID=$!
( trap 'kill $! 2>/dev/null; exit 0' TERM; while true; do sleep 600 & wait $!; prune; done ) &
PRUNE_PID=$!
STATUS=0; wait "$TRAIN_PID" || STATUS=$?
echo "RLinf exited with status $STATUS" | tee -a "$LOG"
exit "$STATUS"
