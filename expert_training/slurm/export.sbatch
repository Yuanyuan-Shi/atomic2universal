#!/bin/bash
# Export raw RL checkpoints of one expert run as eval-ready GR00T folders (checkpoints/experts/<Task>/step_<N>/)
# with expert_training/export_expert.py, one step after another, on one GPU (the verification loads the
# model on the GPU). Steps: global step numbers, 'latest' or 'best'. From the project root:
#   sbatch -p b200 -o <dir>/export-%j.out expert_training/slurm/export.sbatch <Task> <run_dir> <step> [<step> ...]
#SBATCH --job-name=a2u-export
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail

TASK="${1:?usage: sbatch ... export.sbatch <Task> <run_dir> <step> [<step> ...]}"
RUN_DIR="${2:?usage: sbatch ... export.sbatch <Task> <run_dir> <step> [<step> ...]}"
shift 2
[ $# -gt 0 ] || { echo "ERROR: give at least one step" >&2; exit 1; }
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
[ -f expert_training/export_expert.py ] || { echo "ERROR: submit from the project root ($ROOT is not)" >&2; exit 1; }

set +u  # conda's activate scripts reference unset variables
source "$ROOT/third_party/miniforge3/etc/profile.d/conda.sh"
conda activate a2u-sim  # export_expert.py runs the converter and the verifier in their own envs
set -u
unset PYTHONPATH
echo "== $(date '+%F %T') job $SLURM_JOB_ID on $(hostname): $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
for step in "$@"; do
  echo "== $(date '+%F %T') export $TASK step $step"
  python expert_training/export_expert.py --task "$TASK" --run-dir "$RUN_DIR" --step "$step"
done
echo "== $(date '+%F %T') done"
