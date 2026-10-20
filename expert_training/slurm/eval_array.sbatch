#!/bin/bash
# Closed-loop evaluations as a Slurm array: one GPU per array task; task i runs baseline_eval.run_baseline with
# the arguments on line i of a job-list file, launching its own policy server on port 5555 + i (tasks may
# share a node). Comparison numbers must come from one GPU type: pass the partition (and, to keep the
# evaluations on one node, -w <node>) when submitting, from the project root:
#   sbatch -p b200 [-w gb2NN] --array=1-<lines> -o <dir>/eval-%A_%a.out expert_training/slurm/eval_array.sbatch <list>
# A list line, e.g. (paths relative to the project root):
#   --pairing pnp_cabinet_drawer --roles i1 --checkpoint checkpoints/experts/<Task>/step_10 --seed-start 1000
#   --num-episodes 20 --results-root expert_training/results/<Task>/select/step_10
#SBATCH --job-name=a2u-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail

LIST="${1:?usage: sbatch ... eval_array.sbatch <job-list file>}"
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
[ -f baseline_eval/run_baseline.py ] || { echo "ERROR: submit from the project root ($ROOT is not)" >&2; exit 1; }
[ -f "$LIST" ] || { echo "ERROR: job list $LIST not found" >&2; exit 1; }
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as an array: --array=1-<number of lines>}"
LINE="$(sed -n "${TASK_ID}p" "$LIST")"
[ -n "$LINE" ] || { echo "ERROR: $LIST has no line $TASK_ID" >&2; exit 1; }
read -r -a ARGS <<<"$LINE"
for arg in "${ARGS[@]}"; do
  case "$arg" in --port|--launch-server) echo "ERROR: line $TASK_ID sets $arg; the job chooses the port" >&2; exit 1 ;; esac
done

set +u  # conda's activate scripts reference unset variables
source "$ROOT/third_party/miniforge3/etc/profile.d/conda.sh"
conda activate a2u-sim  # simulator + client; the server runs in a2u-policy (evaluation/serve_policy.sh)
set -u
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset PYTHONPATH
echo "== $(date '+%F %T') job ${SLURM_ARRAY_JOB_ID}_${TASK_ID} on $(hostname) ($SLURM_JOB_PARTITION):" \
     "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
echo "run_baseline ${ARGS[*]}"
python -m baseline_eval.run_baseline "${ARGS[@]}" --launch-server --port $((5555 + TASK_ID))
