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
```

Report (one tab per task set with results; `_smoke` and other `_` folders skipped):
```bash
python -m baseline_eval.build_report             # or: --pairings pnp_cabinet_drawer
python -m http.server 8080 --directory baseline_eval/results
# open http://localhost:8080/report.html (remote: ssh -L 8080:localhost:8080 <user>@<host>)
```
