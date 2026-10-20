# expert_training

Trains one PPO expert per atomic task from the GR00T N1.5 joint SFT checkpoint-60000, with StructRL's
GR00T-N1.5 Sparse-RL recipe (terminal-only reward) on RLinf; the experts are the teachers for distillation.
Keyed by task: `runs/<Task>/<run_id>/` (resolved_config.yaml, train.log, raw checkpoints, seeds/, slurm-*.out,
jobs.tsv), `checkpoints/experts/<Task>/step_<N>/`, `results/<Task>/<label>/`. Another expert needs only another `--task`.

`build_config.py` (config + checks: `task_registry.py`, `batch_rules.py`), `train_expert.sh` (launch, prune via
`raw_checkpoints.py`, startup hooks in `startup_hooks/`: scene seeds `a2u_scene_seeds.py`, W&B metadata and extra
metrics `a2u_run_metrics.py` + `a2u_resources.py`, best checkpoint `a2u_best_checkpoint.py`), `export_expert.py` (StructRL converter, then `verify_expert.py`),
`make_step0.py` (untrained control), `wandb_log_eval.py` (evaluation results to the W&B run), `slurm/` (cluster
jobs). Settings: `configs/rl/`. Background: `docs/rl_recon.md`, `docs/environment.md`, `docs/cluster.md`.

## Schmidt Sciences cluster (project root /scratch/schmidt/ssci-ys-shigroup/atomic2universal)
```bash
source third_party/miniforge3/etc/profile.d/conda.sh
RL="conda run --no-capture-output -p third_party/conda_envs/a2u-rl-blackwell"
env -u PYTHONPATH $RL python expert_training/build_config.py --task PickPlaceCounterToCabinet --hw node8 --dry-run
# Training: one node x 8 GPUs (B200 or B300), chained jobs resume from the newest raw checkpoint
bash expert_training/slurm/submit_training.sh --task PickPlaceCounterToCabinet --chain <N>
bash expert_training/slurm/submit_training.sh --task PickPlaceCounterToCabinet --override iterations=2 \
  --override save_every=1                                     # the 2-iteration pilot
bash expert_training/slurm/submit_training.sh --task <Task> --run-id <run_id> [--override ...]  # continue a run
# Evaluations, one GPU each: one line of run_baseline arguments per array task
sbatch -p b200 --array=1-<lines> -o <dir>/eval-%A_%a.out expert_training/slurm/eval_array.sbatch <list>
# Results to the training run's W&B page (RL env)
env -u PYTHONPATH $RL python expert_training/wandb_log_eval.py --run-dir <run_dir> \
  --select expert_training/results/<Task>/select/step_*/pnp_cabinet_drawer/i1/summary.json \
  --final-expert <summary.json> --final-baseline <summary.json> --final-step0 <summary.json>
```
W&B: project `atomic2universal`, entity `WANDB_ENTITY` from `.env` (else the key's default), group `expert/<Task>`,
job type `train`, run id = run name = run directory name, `resume=allow` (one W&B run across chained jobs), tags
task, GPU type, `sparse-ppo`, `frozen-vlm`. Metrics are plotted against `iteration` (= checkpoint step). RLinf's
names: `env/success_once` (= `train/rollout_success`, training rollouts with exploration noise, not evaluation),
`env/return`, `env/episode_len`, `train/actor/{policy_loss,approx_kl,clip_fraction,grad_norm,lr}`,
`train/critic/value_loss`, `time/generate_rollouts` (rollout s), `time/actor/run_training` (train s); added:
`time/save_checkpoint`, `speed/episodes_per_second`, `resources/peak_gpu_mem_gib/gpu<i>`,
`resources/peak_host_ram_gib` (job cgroup), `seeds/{distinct,min,max}` (expect 512 and >= 100000 per iteration),
`best/{score,step}`. **Best checkpoint**: besides `global_step_N` every `save_every`, `<run_dir>/checkpoints/best/`
holds the weights at the best mean `train/rollout_success` over `save_best_window` (5) iterations (`best.json`: step,
score; all scores in `rollout_success.tsv`); export it with `--step best`. A training-time pick only: the final-test
checkpoint is still chosen by the selection eval on seeds 1000-1019.
**Offline fallback** (compute nodes without W&B access): submit with `WANDB_MODE=offline` in the environment, then
after each job, on the login node: `$RL wandb sync <run_dir>/wandb/wandb/offline-run-*` (keys from `.env`:
`source setup/dotenv.sh && load_dotenv .env` first).

## Workstation (smoke test, `hw_workstation.yaml`)
```bash
cd /mnt1/atomic2universal
RL="conda run --no-capture-output -p third_party/conda_envs/a2u-rl-blackwell"   # H200 nodes: .../a2u-rl
env -u PYTHONPATH $RL python expert_training/build_config.py --task PickPlaceCounterToCabinet --hw workstation --dry-run
# Smoke test (1 GPU, 2 envs x 4 epochs, 3 iterations, checkpoint every iteration); prints the run dir at the end
bash expert_training/train_expert.sh --task PickPlaceCounterToCabinet --hw workstation
# Step-0 control: checkpoint-60000 through RLinf's load -> bf16 -> save, no training (CPU)
env -u PYTHONPATH $RL python expert_training/make_step0.py --task PickPlaceCounterToCabinet --hw workstation
# Export (run in a2u-sim; converter in the RL env, verification on the GPU in a2u-policy)
conda activate a2u-sim
python expert_training/export_expert.py --task PickPlaceCounterToCabinet --run-dir <run_dir> --step latest  # or N, or best
python expert_training/export_expert.py --task PickPlaceCounterToCabinet \
  --run-dir expert_training/runs/PickPlaceCounterToCabinet/step0-control --step 0
# Evaluate on seeds 0-9 (a2u-sim), same seeds as the Stage 1c baseline
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 --launch-server \
  --checkpoint checkpoints/experts/PickPlaceCounterToCabinet/step_<N> --seed-start 0 --num-episodes 10 \
  --results-root expert_training/results/PickPlaceCounterToCabinet/_smoke
```
Seeds: reset k of training env i uses seed 100000 + i + num_envs * k (`a2u_scene_seeds.py`). Select checkpoints on
1000-1019 (`--seed-start 1000 --num-episodes 20`), run the final test on 0-99; never select on 0-99. Ctrl-C (or
Slurm's TERM before the wall time) stops the whole training process group (<= 60 s).
