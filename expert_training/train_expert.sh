#!/usr/bin/env bash
# Train one single-task PPO expert (StructRL's GR00T-N1.5 Sparse-RL recipe on RLinf, one node, no Slurm):
# build <run_dir>/resolved_config.yaml (build_config.py), launch RLinf's train_embodied_agent.py on exactly
# that file in its own process group, tee everything to <run_dir>/train.log, prune old raw checkpoints every
# 10 min and at exit (raw_checkpoints.py); any exit stops the group, incl. the local Ray cluster RLinf starts.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash expert_training/train_expert.sh --task <Task> --hw <workstation|node8>
                                            [--run-id ID] [--resume] [--override KEY=N]... [--env-prefix DIR]
  --task        atomic_seen task, e.g. PickPlaceCounterToCabinet
  --hw          hardware file configs/rl/hw_<hw>.yaml
  --run-id      run directory expert_training/runs/<Task>/<ID> (default: <YYYYmmdd-HHMM>-<hw>); also the
                W&B run id and name
  --resume      continue from the newest complete raw checkpoint of the run named by --run-id (or start
                that run if it has none yet, e.g. the first job of a Slurm chain), else of the run of
                <Task> whose train.log changed last
  --override    passed to build_config.py, e.g. iterations=2 (pass the same values when resuming)
  --env-prefix  RL conda env (default: third_party/conda_envs/a2u-rl-blackwell if present, else a2u-rl)
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARGS="$*"
TASK="" HW="" RUN_ID="" RESUME=0 ENV_PREFIX="" TRAIN_PID="" PRUNE_PID="" WATCH_PID="" OVERRIDES=()
STALL_MIN="${A2U_STALL_MIN:-20}" FIRST_ITER_MIN="${A2U_FIRST_ITER_MIN:-60}"  # watchdog limits (minutes)
while [ "$#" -gt 0 ]; do
  case "$1" in
    --task|--hw|--run-id|--env-prefix|--override)
      if [ "$#" -lt 2 ]; then echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; fi
      case "$1" in
        --task) TASK="$2" ;; --hw) HW="$2" ;; --run-id) RUN_ID="$2" ;; --override) OVERRIDES+=(--override "$2") ;;
        *) ENV_PREFIX="$2" ;;
      esac
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
# Threads count against `ulimit -u`. Slurm copies the login shell's soft limit (4096 on the cluster) into jobs, and
# Ray's ~1 worker per core with dozens of threads each crosses it: the raylet dies in pthread_create (EAGAIN) and the
# driver hangs in RegisterClient (docs/cluster.md). Raise the soft limit to the hard one and refuse a low one.
ulimit -S -u "$(ulimit -H -u)"
if [ "$(ulimit -u)" != unlimited ] && [ "$(ulimit -u)" -lt 65536 ]; then
  echo "ERROR: max user processes (threads) is $(ulimit -u) (hard limit); Ray needs far more on a large node" >&2; exit 1
fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[ -n "$CONDA" ] || { echo "ERROR: conda not found. Activate conda or set CONDA_EXE." >&2; exit 1; }

# StructRL's launcher environment (docs/rl_recon.md Q6). PYTHONPATH is replaced, not appended
# (ROS Noetic in ~/.bashrc sets it), with our startup hooks (startup_hooks/sitecustomize.py) first.
ST="$ROOT/third_party/StructRL"
export PROJECT_ROOT="$ST" REPO_PATH="$ST/RLinf" EMBODIED_PATH="$ST/RLinf/examples/embodiment" STRUCTRL_PROJECT_ROOT="$ST"
export STRUCTRL_TIMING_JSON="$ST/structrl/demo_timing_composite_seen.json" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/expert_training/startup_hooks:$ST:$ST/RLinf:$ST/robocasa:$ST/robosuite:$ST/Isaac-GR00T"
# RLinf's ray.init(address="auto") joins any Ray cluster registered under the Ray temp dir; a private one per
# job keeps it from finding another user's cluster on a shared node (short: Ray's socket paths are length-limited).
unset RAY_ADDRESS; export RAY_TMPDIR="/tmp/a2u-ray-${SLURM_JOB_ID:-$$}"
PY=("$CONDA" run --no-capture-output -p "$ENV_PREFIX" python -u)

# Preflight: visible GPUs match the hardware file; no Ray cluster that RLinf's ray.init(address="auto") would join.
GPUS="$(sed -n 's/^gpus:[[:space:]]*\([0-9]*\).*/\1/p' "$HW_FILE")"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then VISIBLE="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)"
else VISIBLE="$(nvidia-smi -L | grep -c '^GPU ')"; fi
if [ "$VISIBLE" != "$GPUS" ]; then
  echo "ERROR: $HW_FILE has gpus: $GPUS but $VISIBLE GPU(s) are visible; set CUDA_VISIBLE_DEVICES or pick another --hw" >&2; exit 1
fi
if pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  echo "ERROR: a Ray cluster of yours is already running here (gcs_server pid $(pgrep -u "$(id -u)" -x gcs_server | tr '\n' ' '));" \
       "RLinf would join it instead of starting its own. Stop it if it is yours (ray stop) or wait." >&2; exit 1
fi
HOOK_CHECK="$("${PY[@]}" "$ROOT/expert_training/startup_hooks/sitecustomize.py" 2>&1)" || { echo "$HOOK_CHECK" >&2; exit 1; }

RUNS="$ROOT/expert_training/runs/$TASK"
BUILD_ARGS=(--task "$TASK" --hw "$HW" "${OVERRIDES[@]}")
if [ "$RESUME" = 1 ]; then
  RUN_DIR="$RUNS/$RUN_ID"
  if [ -z "$RUN_ID" ]; then LATEST="$(ls -t "$RUNS"/*/train.log 2>/dev/null | head -n 1 || true)"; RUN_DIR="${LATEST%/train.log}"; fi
  if [ -z "$RUN_ID" ] && [ ! -f "$RUN_DIR/train.log" ]; then echo "ERROR: no run of $TASK to resume (no train.log in '$RUN_DIR')" >&2; exit 1; fi
  if RESUME_DIR="$("${PY[@]}" "$ROOT/expert_training/raw_checkpoints.py" resume-dir "$RUN_DIR" 2>/dev/null)"; then
    BUILD_ARGS+=(--resume-dir "$RESUME_DIR")
  else
    echo "No complete raw checkpoint in $RUN_DIR yet: starting run $(basename "$RUN_DIR") from iteration 0"
  fi
else
  RUN_DIR="$RUNS/${RUN_ID:-$(date +%Y%m%d-%H%M)-$HW}"
  if [ -e "$RUN_DIR" ]; then echo "ERROR: $RUN_DIR exists; pick another --run-id or pass --resume" >&2; exit 1; fi
fi
BUILD_OUTPUT="$("${PY[@]}" "$ROOT/expert_training/build_config.py" "${BUILD_ARGS[@]}" --run-dir "$RUN_DIR" 2>&1)" \
  || { echo "$BUILD_OUTPUT" >&2; exit 1; }
LOG="$RUN_DIR/train.log"
# Startup hooks: seeds and run metadata (a2u_scene_seeds.py, a2u_run_metrics.py). W&B settings come from the
# resolved config's a2u_expert.wandb (build_config.py); WANDB_API_KEY and WANDB_ENTITY from .env.
export A2U_RUN_DIR="$RUN_DIR" A2U_RUN_ID="$(basename "$RUN_DIR")"
[ -n "${WANDB_API_KEY:-}" ] || { source "$ROOT/setup/dotenv.sh" && load_dotenv "$ROOT/.env"; }
WANDB_ENV="$("${PY[@]}" -c '
import sys, yaml
w = yaml.safe_load(open(sys.argv[1]))["a2u_expert"]["wandb"]
print(int(w["enabled"]), w["group"], w["job_type"], ",".join(w["tags"]))' "$RUN_DIR/resolved_config.yaml")"
read -r WANDB_ON WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS <<<"$WANDB_ENV"
if [ "$WANDB_ON" = 1 ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != offline ]; then
    echo "ERROR: W&B logging is on but WANDB_API_KEY is not set (.env); set it or WANDB_MODE=offline" >&2; exit 1
  fi
  GPU_TAG="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | sed 's/^NVIDIA //; s/ /-/g')"
  export WANDB_RUN_ID="$A2U_RUN_ID" WANDB_RESUME=allow WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS="$WANDB_TAGS,$GPU_TAG"
fi
{ echo "===== $(date '+%F %T') train_expert.sh $ARGS | env $ENV_PREFIX | host $(hostname) | job ${SLURM_JOB_ID:-none}"
  echo "$HOOK_CHECK"; echo "$BUILD_OUTPUT"; } | tee -a "$LOG"

prune() {  # its output goes to prune.log, so train.log grows only with RLinf's output (watchdog)
  "${PY[@]}" "$ROOT/expert_training/raw_checkpoints.py" prune "$RUN_DIR" >> "$RUN_DIR/prune.log" 2>&1 \
    || echo "WARNING: pruning raw checkpoints failed (see prune.log); disk use keeps growing" | tee -a "$LOG" >&2
}
watchdog() {  # stop the run when RLinf hangs: train.log silent for STALL_MIN, or no iteration of this job in FIRST_ITER_MIN
  set +o pipefail  # grep -q may close the pipe early
  local start offset last_size=-1 last_change size now reason first_done=0
  start="$(date +%s)" last_change="$(date +%s)" offset="$(stat -c %s "$LOG")"
  while sleep 60; do
    now="$(date +%s)" size="$(stat -c %s "$LOG")" reason=""
    if [ "$size" != "$last_size" ]; then last_size="$size" last_change="$now"; fi
    if [ "$first_done" = 0 ] && tail -c "+$((offset + 1))" "$LOG" | grep -q "\[a2u metrics\] iteration"; then first_done=1; fi
    if [ $((now - last_change)) -gt $((STALL_MIN * 60)) ]; then reason="train.log has had no new output for $STALL_MIN min"
    elif [ "$first_done" = 0 ] && [ $((now - start)) -gt $((FIRST_ITER_MIN * 60)) ]; then
      reason="no iteration finished within $FIRST_ITER_MIN min of the launch"; fi
    if [ -n "$reason" ]; then echo "WATCHDOG: $reason; stopping the run" | tee -a "$LOG" >&2; kill -TERM $$; return; fi
  done
}
save_ray_logs() {  # Ray's session logs live in the node-local RAY_TMPDIR; keep them with the run, then remove it
  local dest="$RUN_DIR/ray_logs/job_${SLURM_JOB_ID:-$$}" logs
  [ -d "$RAY_TMPDIR" ] || return 0
  while IFS= read -r logs; do
    mkdir -p "$dest" && cp -r "$logs" "$dest/$(basename "$(dirname "$logs")")" 2>/dev/null || true
  done < <(find "$RAY_TMPDIR" -maxdepth 4 -type d -name logs -path '*session_*' 2>/dev/null)
  [ -d "$dest" ] && echo "Ray logs: $dest" | tee -a "$LOG"
  rm -rf "$RAY_TMPDIR"
}
cleanup() {
  local status=$?
  trap - EXIT; trap '' INT TERM HUP  # a second Ctrl-C must not abort the cleanup (bounded: <= 60 s + prune)
  for pid in $PRUNE_PID $WATCH_PID; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
  if [ -n "$TRAIN_PID" ] && kill -0 -- "-$TRAIN_PID" 2>/dev/null; then
    echo "Stopping process group $TRAIN_PID (RLinf driver, local Ray, env workers); wait <= 60 s" | tee -a "$LOG" >&2
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
    ( sleep 60; kill -KILL -- "-$TRAIN_PID" 2>/dev/null ) &  # escalate if anything ignores SIGTERM
    local killer=$!
    wait "$TRAIN_PID" 2>/dev/null || true  # reap the leader, so the group check below sees only live processes
    while kill -0 -- "-$TRAIN_PID" 2>/dev/null && kill -0 "$killer" 2>/dev/null; do sleep 1; done
    kill -KILL "$killer" 2>/dev/null || true  # it inherited the ignored SIGTERM
  fi
  save_ray_logs
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
watchdog &
WATCH_PID=$!
STATUS=0; wait "$TRAIN_PID" || STATUS=$?
echo "RLinf exited with status $STATUS" | tee -a "$LOG"
exit "$STATUS"
