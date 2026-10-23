#!/usr/bin/env bash
# Stage 3a toy sweep: the three loss variants (lambda_IDA 0.99, K_psi 1) and the DMD + FM IDA sweep
# (lambda_IDA in {1, 0.99, 0.9, 0} x K_psi in {1, 5}); dmd_fm is shared by both, so 10 runs in total.
# CPU only, at most 3 runs at a time with 4 threads each; K_psi = 5 runs go first because they take longest.
# Usage (from anywhere): bash distillation/toy/run_sweep.sh   -> runs/<run>.json, runs/<run>.pt, runs/<run>.out
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="${PY:-third_party/conda_envs/a2u-rl-blackwell/bin/python}"
mkdir -p distillation/toy/runs

# run name           alpha_dmd alpha_fm lambda_ida k_fake
CONFIGS="dmd_fm_lam1_k5      0.5 0.5 1    5
dmd_fm_lam0.99_k5   0.5 0.5 0.99 5
dmd_fm_lam0.9_k5    0.5 0.5 0.9  5
dmd_fm_lam0_k5      0.5 0.5 0    5
fm_only             0   1   0.99 1
dmd_only            1   0   0.99 1
dmd_fm              0.5 0.5 0.99 1
dmd_fm_lam1_k1      0.5 0.5 1    1
dmd_fm_lam0.9_k1    0.5 0.5 0.9  1
dmd_fm_lam0_k1      0.5 0.5 0    1"

echo "$CONFIGS" | xargs -P 3 -L 1 sh -c \
  '"$0" -m distillation.toy.train_toy --run "$1" --alpha-dmd "$2" --alpha-fm "$3" --lambda-ida "$4" --k-fake "$5" \
   > "distillation/toy/runs/$1.out" 2>&1 && echo "done: $1"' "$PY"
