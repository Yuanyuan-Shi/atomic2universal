# baseline_eval

Purpose: measure closed-loop success of the GR00T N1.5 joint SFT checkpoint-60000 on one task set
(atomic roles i1, i2, ... and an optional composite), 10 seeded episodes per role, one MP4 each.

Inputs: `configs/pairings/<task set>.yaml` (roles, tasks, modes, reference SRs), `configs/eval/baseline.yaml`
(checkpoint, split, episodes, seeds, action horizon, video), `checkpoints/sft_joint/...`.

Outputs (episode video paths are relative to the pairing directory):
```
baseline_eval/results/<pairing>/<role>/episodes.jsonl        # one line per finished episode
baseline_eval/results/<pairing>/<role>/summary.json          # once all episodes exist; Wilson ci95
baseline_eval/results/<pairing>/<role>/videos/ep{episode:02d}_seed{seed}.mp4
baseline_eval/results/<pairing>/policy_server_port<N>.log    # with --launch-server
baseline_eval/results/report.html                            # one tab per pairing
baseline_eval/results/_rl_experts/<pairing>/<role>           # symlinks to the expert results (git-ignored)
```

Two conda envs: the sim/eval code runs in `a2u-sim`; the policy runs in `a2u-policy` behind the
official GR00T server. `--launch-server` starts it on `--port` (default 5555) and always stops it;
without it, start one yourself: `bash evaluation/serve_policy.sh --port 5555`.
```bash
conda activate a2u-sim     # or prefix: conda run --no-capture-output -n a2u-sim
# smoke test: 1 episode per role, every role of the config in order (git-ignored output)
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer \
  --num-episodes 1 --results-root baseline_eval/results/_smoke --launch-server
# full run, one process + server per role (re-running resumes; --overwrite starts over)
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 --port 5555 --launch-server
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i2 --port 5556 --launch-server
# large runs without MP4s (episode "video": null, summary "video": false), e.g. the 100-episode Stage 2b runs
python -m baseline_eval.run_baseline --pairing pnp_cabinet_drawer --roles i1 --seed-start 0 --num-episodes 100 \
  --no-video --launch-server --checkpoint <model dir> --results-root eval_n100/<model>
```

Report (one tab per task set with results; `_smoke` and other `_` folders skipped). Below the baseline,
a tab gets an "RL Expert closed-loop evaluation" section when `configs/eval/rl_experts.yaml` lists the task
set: per role, the `--results-root` of an expert evaluation (same seeds as the baseline), or `null` for a
pending row. Above it, a 100-episode RL expert vs baseline table appears for every role with both
`eval_n100/{baseline,experts}/<pairing>/<role>/summary.json` (Newcombe CI of the gain). Code: `build_report.py`
(tabs, page), `report_common.py` (style, rows, video cards), `report_rl.py` (RL expert and 100-episode parts).
```bash
python -m baseline_eval.build_report             # or: --pairings pnp_cabinet_drawer
python -m http.server 8080 --directory baseline_eval/results
# open http://localhost:8080/report.html (remote: ssh -L 8080:localhost:8080 <user>@<host>)
```
