#!/bin/bash
# One training job of an expert run on one 8-GPU node, B200 or B300, whichever starts first (docs/cluster.md).
# Submitted by submit_training.sh (from the project root, with TASK, RUN_ID, HW, OVERRIDES exported and
# -o <run_dir>/slurm-%j.out); every job runs train_expert.sh --resume, so a chained job continues the run.
#SBATCH --job-name=a2u-train
#SBATCH --partition=b200,b300
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=124
#SBATCH --mem=1488000M
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
: "${TASK:?TASK not set (use submit_training.sh)}" "${RUN_ID:?RUN_ID not set}" "${HW:=node8}" "${OVERRIDES:=}"
cd "$ROOT"
[ -f expert_training/train_expert.sh ] || { echo "ERROR: submit from the project root ($ROOT is not)" >&2; exit 1; }
RUN_DIR="$ROOT/expert_training/runs/$TASK/$RUN_ID"
JOBS="$RUN_DIR/jobs.tsv"  # one line per job: job id, node, partition, resume step at start, at end, exit status

source "$ROOT/setup/dotenv.sh"
load_dotenv "$ROOT/.env"  # WANDB_API_KEY, WANDB_ENTITY; never echoed
RL_ENV="$ROOT/third_party/conda_envs/a2u-rl-blackwell"
set +u  # conda's activate scripts reference unset variables
source "$ROOT/third_party/miniforge3/etc/profile.d/conda.sh"
conda activate "$RL_ENV"
set -u
export WANDB_MODE="${WANDB_MODE:-online}"  # compute nodes reach api.wandb.ai (docs/cluster.md); else offline + wandb sync

resume_step() {  # global step of the newest complete raw checkpoint, 0 if none
  local dir; dir="$(python "$ROOT/expert_training/raw_checkpoints.py" resume-dir "$RUN_DIR" 2>/dev/null)" || { echo 0; return; }
  echo "${dir##*global_step_}"
}
START="$(resume_step)"
if [ -f "$RUN_DIR/resolved_config.yaml" ]; then
  TOTAL="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["runner"]["max_epochs"])' "$RUN_DIR/resolved_config.yaml")"
  if [ "$START" -ge "$TOTAL" ]; then echo "Run $RUN_ID is complete (global_step_$START); nothing to do."; exit 0; fi
fi
# Stop a chain that makes no progress: the last two jobs both ended without a new checkpoint.
if [ -f "$JOBS" ] && [ "$(tail -n 2 "$JOBS" | awk -F'\t' '$4 == $5' | wc -l)" -ge 2 ]; then
  echo "ERROR: the last two jobs of $RUN_ID made no progress (see $JOBS); not starting another" >&2; exit 1
fi

echo "== $(date '+%F %T') job $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION), run $RUN_ID from global_step_$START"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
FLAGS=(--task "$TASK" --hw "$HW" --run-id "$RUN_ID" --resume)
for item in $OVERRIDES; do FLAGS+=(--override "$item"); done
STATUS=0
bash "$ROOT/expert_training/train_expert.sh" --env-prefix "$RL_ENV" "${FLAGS[@]}" & CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM  # wall time (--signal): let train_expert.sh stop RLinf and prune
wait "$CHILD" || STATUS=$?
if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD" || STATUS=$?; fi  # the first wait returns early on TERM
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "$(hostname)" "$SLURM_JOB_PARTITION" "$START" "$(resume_step)" "$STATUS" >> "$JOBS"
if [ "${WANDB_MODE}" = offline ]; then echo "W&B offline: sync from the login node (expert_training/README.md)"; fi
exit "$STATUS"
