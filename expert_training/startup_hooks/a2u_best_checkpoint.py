"""Keep a copy of the best weights of an expert run by training rollout success (called by a2u_run_metrics).

Score after iteration N = mean train/rollout_success (env/success_once) of iterations N-W+1 .. N, where
W = a2u_expert.save_best_window in resolved_config.yaml (configs/rl/ppo_sparse.yaml save_best_window;
absent in older runs: off). One iteration's 512 training episodes have a standard error of about 0.02, so
the maximum of single iterations mostly picks a lucky draw; the W-iteration mean is steadier.
When the score beats the stored best, the actor's current weights are saved to <run>/checkpoints/best/
in the same layout as global_step_N (actor/model_state_dict/full_weights.pt; the resume state is
removed), so `export_expert.py --step best` exports it, plus best.json (step, score, window, scores).
The rollouts of iteration N come from the weights of step N-1, so the saved step-N weights are one PPO
update past the last scored ones (approx_kl about 0.002 per update).
Every score is appended to <run>/rollout_success.tsv (step<TAB>value); after --resume a repeated step
takes its newest value and the stored best is kept. This is a training-time pick: the checkpoint used
in the final test is still chosen by the selection eval (seeds 1000-1019).
"""

import json
import shutil
from pathlib import Path


def window_scores(run_dir: Path, step: int, value: float, window: int) -> list[float]:
    """Record this step's success; return the last `window` steps' values (fewer if some are missing)."""
    path = run_dir / "rollout_success.tsv"
    with path.open("a") as out:
        out.write(f"{step}\t{value}\n")
    scores = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 2:
            scores[int(fields[0])] = float(fields[1])
    return [scores[s] for s in range(step - window + 1, step + 1) if s in scores]


def update(runner, run_dir: Path, step: int, value: float, window: int) -> dict:
    """Save the best checkpoint if this step's windowed score is a new best; return best/* metrics."""
    best_dir = run_dir / "checkpoints" / "best"
    info_path = best_dir / "best.json"
    best = json.loads(info_path.read_text()) if info_path.is_file() else None
    recent = window_scores(run_dir, step, value, window)
    if len(recent) == window and (best is None or sum(recent) / window > best["score"]):
        partial, old = best_dir.with_name("best.partial"), best_dir.with_name("best.old")
        shutil.rmtree(partial, ignore_errors=True)
        actor = partial / "actor"
        actor.mkdir(parents=True)
        runner.actor.save_checkpoint(str(actor), step).wait()
        for path in actor.iterdir():
            if path.name != "model_state_dict":
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        best = {"step": step, "score": sum(recent) / window, "window": window, "scores": recent}
        (partial / "best.json").write_text(json.dumps(best, indent=2) + "\n")
        shutil.rmtree(old, ignore_errors=True)
        if best_dir.exists():
            best_dir.rename(old)
        partial.rename(best_dir)
        shutil.rmtree(old, ignore_errors=True)
        print(f"[a2u best] step {step}: new best {best['score']:.4f} (mean of {window}); saved {best_dir}", flush=True)
    return {} if best is None else {"best/score": best["score"], "best/step": best["step"]}
