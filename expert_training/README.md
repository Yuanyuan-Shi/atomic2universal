# expert_training

Trains one PPO expert per atomic task from the GR00T N1.5 joint SFT checkpoint-60000, with StructRL's
GR00T-N1.5 Sparse-RL recipe (terminal-only reward) on RLinf; the experts are the teachers for distillation.
Keyed by task: `runs/<Task>/<run_id>/` (resolved_config.yaml, train.log, raw checkpoints),
`checkpoints/experts/<Task>/step_<N>/`, `results/<Task>/<label>/`. Another expert needs only another `--task`.

`build_config.py` (config + checks: `task_registry.py`, `batch_rules.py`), `train_expert.sh` (launch, prune via
`raw_checkpoints.py`, scene seeding via `seed_hook/`), `export_expert.py` (StructRL converter, then
`verify_expert.py`), `make_step0.py` (untrained control). Settings: `configs/rl/`. Background:
`docs/rl_recon.md`, `docs/environment.md` (Stage 2 sections and deviations).

```bash
cd /mnt1/atomic2universal
RL="conda run --no-capture-output -p third_party/conda_envs/a2u-rl-blackwell"   # AWS H200: .../a2u-rl
env -u PYTHONPATH $RL python expert_training/build_config.py --task PickPlaceCounterToCabinet --hw workstation --dry-run
# Smoke test (1 GPU, 8 envs, 3 iterations, checkpoint every iteration); prints the run dir at the end
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw workstation
# Step-0 control: checkpoint-60000 through RLinf's load -> bf16 -> save, no training (CPU)
env -u PYTHONPATH $RL python expert_training/make_step0.py --task PickPlaceCounterToCabinet --hw workstation
# Export (run in a2u-sim; converter in the RL env, verification on the GPU in a2u-policy)
conda activate a2u-sim
python expert_training/export_expert.py --task PickPlaceCounterToCabinet --run-dir <run_dir> --step latest
python expert_training/export_expert.py --task PickPlaceCounterToCabinet \
  --run-dir expert_training/runs/PickPlaceCounterToCabinet/step0-control --step 0
# Evaluate on seeds 0-9 (a2u-sim), same seeds as the Stage 1c baseline
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 --launch-server \
  --checkpoint checkpoints/experts/PickPlaceCounterToCabinet/step_<N> --seed-start 0 --num-episodes 10 \
  --results-root expert_training/results/PickPlaceCounterToCabinet/_smoke
# AWS node (8 GPUs, 100 iterations, checkpoint every 10); --resume continues the latest run
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw aws_node [--resume]
```
Seeds: training envs use 100000 + env id. Select checkpoints on 1000-1019 (`--seed-start 1000 --num-episodes 20`),
run the final test on 0-99; never select on 0-99. Ctrl-C stops the whole training process group (<= 60 s).
