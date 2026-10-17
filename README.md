# atomic2universal

Research question: is **specialize-then-combine** (PPO-fine-tune one expert per pick-and-place
task, then distill the experts into one student with multi-teacher distillation, MTD) better than
**combine-then-train** (fine-tune one policy on all the tasks together)? The task family is the 5
pick-and-place tasks among RoboCasa365's 18 fine-tuned atomic tasks, starting with two:
PickPlaceCounterToCabinet and PickPlaceDrawerToCounter. Simulator: robosuite + RoboCasa365.
Starting policy: GR00T N1.5 joint SFT checkpoint `target_posttraining/atomic_seen/checkpoint-60000`.

## Stages
- `baseline_eval/` (Stage 1): closed-loop baseline of the SFT checkpoint on a task set (atomic roles i1, i2, ..., optional composite). Trains nothing.
- `expert_training/` (next): per-task PPO experts fine-tuned from checkpoint-60000.
- `distillation/` (later): multi-teacher distillation of the experts into one universal policy.

## Layout
- `setup/`: environment and checkpoint scripts. `docs/`: `environment.md` and `eval_entrypoint.md`.
- `configs/pairings/`: WHAT is evaluated. `configs/eval/baseline.yaml`: HOW it is evaluated.
- `evaluation/`: shared closed-loop eval library. `third_party/`, `checkpoints/`: git-ignored.

## Run
```bash
cp .env.example .env                 # set HF_TOKEN; never commit .env
bash setup/setup_sim.sh              # conda env a2u-sim: robosuite + RoboCasa365 + assets
bash setup/setup_policy.sh           # conda env a2u-policy: GR00T N1.5 (RoboCasa365 fork)
bash setup/fetch_checkpoint.sh       # checkpoint -> checkpoints/sft_joint/
conda activate a2u-sim               # all project code runs here
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --launch-server \
  [--roles i1 i2] [--num-episodes 10] [--port 5555] [--overwrite]
python -m baseline_eval.build_report         # one tab per task set -> baseline_eval/results/report.html
python -m http.server 8080 --directory baseline_eval/results
# open http://localhost:8080/report.html (remote: ssh -L 8080:localhost:8080 <user>@<host>)
```
`--launch-server` starts the policy server (`evaluation/serve_policy.sh`, env a2u-policy) on
`--port` and stops it on exit; without it, start the server yourself first.

Environment choices, conda env names, repo commits and deviations are recorded in
`docs/environment.md`; the official eval path (task registry, env construction, success check,
policy loading, cameras) is in `docs/eval_entrypoint.md`. Both are written in Phase 2a.

With 10 episodes per task the 95% CI spans roughly ±25–30 points: a sanity check, not a benchmark.
