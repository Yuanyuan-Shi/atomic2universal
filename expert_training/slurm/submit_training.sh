#!/usr/bin/env bash
# Submit an expert training run as a chain of Slurm jobs (train_expert.sbatch): the first job, then N follow-ups,
# each starting after the previous one ends (afterany) and resuming from the newest raw checkpoint, so the run
# survives wall-time limits and node failures. Prints the run directory and every job id.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash expert_training/slurm/submit_training.sh --task <Task> [--chain N] [--hw node8]
                                                     [--override KEY=N]... [--run-id ID] [--time T]
  --task      atomic_seen task, e.g. PickPlaceCounterToCabinet
  --chain     follow-up jobs after the first (default 0)
  --hw        hardware file configs/rl/hw_<hw>.yaml (default node8)
  --override  passed to build_config.py, e.g. iterations=2 (a continued run must get the same values)
  --run-id    continue this existing run instead of creating <YYYYmmdd-HHMM>-<hw>
  --time      wall time per job, e.g. 02:00:00 for the pilot (default: train_expert.sbatch's 3 days); size it
              to the run, since a hung job bills the whole node until it ends
EOF
}

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="" CHAIN=0 HW=node8 RUN_ID="" OVERRIDES="" TIME=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --task|--chain|--hw|--override|--run-id|--time)
      [ "$#" -ge 2 ] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; }
      case "$1" in
        --task) TASK="$2" ;; --chain) CHAIN="$2" ;; --hw) HW="$2" ;; --run-id) RUN_ID="$2" ;; --time) TIME="$2" ;;
        *) OVERRIDES="${OVERRIDES:+$OVERRIDES }$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$TASK" ] || { echo "ERROR: --task is required" >&2; usage >&2; exit 2; }
[[ "$CHAIN" =~ ^[0-9]+$ ]] || { echo "ERROR: --chain must be a number >= 0" >&2; exit 2; }
[ -f "$ROOT/configs/rl/hw_$HW.yaml" ] || { echo "ERROR: configs/rl/hw_$HW.yaml not found" >&2; exit 2; }

# W&B preflight on the login node: the key works and WANDB_ENTITY (if set) is the user or one of their teams. A bad
# entity otherwise fails only after the job has waited hours for a node.
if [ "${WANDB_MODE:-online}" != offline ]; then
  (source "$ROOT/setup/dotenv.sh" && load_dotenv "$ROOT/.env" && "$ROOT/third_party/conda_envs/a2u-rl-blackwell/bin/python" -c '
import os, sys, wandb
viewer = wandb.Api().viewer
allowed = {viewer.username} | {edge["node"]["name"] for edge in viewer._attrs.get("teams", {}).get("edges", [])}
entity = os.environ.get("WANDB_ENTITY") or viewer.entity
if entity not in allowed:
    sys.exit(f"ERROR: WANDB_ENTITY {entity!r} is not your user or one of your teams {sorted(allowed)}; fix .env")
print(f"W&B preflight: logged in as {viewer.username}, entity {entity}")' 2>&1 | grep -v "^wandb:") \
    || { echo "ERROR: W&B preflight failed (above); set WANDB_MODE=offline to skip it" >&2; exit 1; }
fi

RUNS="$ROOT/expert_training/runs/$TASK"
if [ -n "$RUN_ID" ]; then
  [ -d "$RUNS/$RUN_ID" ] || { echo "ERROR: --run-id $RUN_ID: $RUNS/$RUN_ID does not exist" >&2; exit 1; }
else
  RUN_ID="$(date +%Y%m%d-%H%M)-$HW"
  [ ! -e "$RUNS/$RUN_ID" ] || { echo "ERROR: $RUNS/$RUN_ID exists; wait a minute or pass --run-id" >&2; exit 1; }
fi
RUN_DIR="$RUNS/$RUN_ID"
mkdir -p "$RUN_DIR"  # Slurm writes slurm-%j.out here and does not create directories

cd "$ROOT"  # train_expert.sbatch finds the project through SLURM_SUBMIT_DIR
SUBMIT=(sbatch --parsable --job-name "a2u-$TASK" --output "$RUN_DIR/slurm-%j.out"
        --export "ALL,TASK=$TASK,RUN_ID=$RUN_ID,HW=$HW,OVERRIDES=$OVERRIDES")
[ -z "$TIME" ] || SUBMIT+=(--time "$TIME")
JOB="$("${SUBMIT[@]}" expert_training/slurm/train_expert.sbatch)"
JOB="${JOB%%;*}" IDS="$JOB"
for _ in $(seq 1 "$CHAIN"); do
  JOB="$("${SUBMIT[@]}" --dependency "afterany:$JOB" expert_training/slurm/train_expert.sbatch)"
  JOB="${JOB%%;*}" IDS="$IDS $JOB"
done
echo "Run: $RUN_DIR"
echo "Overrides: ${OVERRIDES:-none}; wall time per job: ${TIME:-3-00:00:00 (sbatch default)}"
echo "Jobs: $IDS"
