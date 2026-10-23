"""Compare the baseline and the RL experts on the 100-episode evaluation (Stage 2b).

Reads <root>/<model>/pnp_cabinet_drawer/<role>/summary.json (and episodes.jsonl) for both models and both
roles (each expert only on its own task), checks each run (100 episodes on seeds 0-99, the intended
checkpoint, task and split, no video), and writes <root>/comparison.md and <root>/comparison_table.tex:
successes, success rate and Wilson 95% CI per model and task, then each expert's gain over the baseline
with a Newcombe (hybrid score) 95% CI for the difference. (The step0 control was dropped by user decision.)

Usage (project root, a2u-sim): python expert_training/compare_models.py [--root eval_n100]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.stats import newcombe_interval, wilson_interval  # noqa: E402

PAIRING, SPLIT, NUM_EPISODES, SEED_START = "pnp_cabinet_drawer", "target", 100, 0
TASKS = {"i1": "PickPlaceCounterToCabinet", "i2": "PickPlaceDrawerToCounter"}
SFT = "checkpoints/sft_joint/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
CHECKPOINTS = {  # model -> role -> intended checkpoint
    "baseline": {"i1": SFT, "i2": SFT},
    "experts": {"i1": "checkpoints/experts/PickPlaceCounterToCabinet/step_95",
                "i2": "checkpoints/experts/PickPlaceDrawerToCounter/step_100"},
}


def load_run(root: Path, model: str, role: str) -> int:
    """Return the successes of one run after checking its summary against the plan and its episodes."""
    role_dir = root / model / PAIRING / role
    if not (role_dir / "summary.json").is_file():
        raise SystemExit(f"ERROR: {role_dir}/summary.json is missing; finish that run_baseline run first")
    summary = json.loads((role_dir / "summary.json").read_text())
    expected = {"pairing": PAIRING, "role": role, "task": TASKS[role], "split": SPLIT, "video": False,
                "checkpoint": CHECKPOINTS[model][role], "seed_start": SEED_START, "num_episodes": NUM_EPISODES}
    wrong = {key: summary.get(key) for key, value in expected.items() if summary.get(key) != value}
    if wrong:
        raise SystemExit(f"ERROR: {role_dir}/summary.json has {wrong}, expected {expected}")
    episodes = [json.loads(line) for line in (role_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    seeds = sorted(episode["seed"] for episode in episodes)
    successes = sum(episode["success"] for episode in episodes)
    if seeds != list(range(SEED_START, SEED_START + NUM_EPISODES)) or successes != summary["num_success"]:
        raise SystemExit(f"ERROR: {role_dir}/episodes.jsonl does not match its summary (seeds or successes)")
    return successes


def percent(value: float) -> str:
    return f"{value:.0%}"


def interval(low: float, high: float) -> str:
    return f"[{low:.0%}, {high:.0%}]"


def points(value: float) -> str:
    return f"{value * 100:+.0f} pp"


def point_interval(low: float, high: float) -> str:
    return f"[{low * 100:+.0f}, {high * 100:+.0f}] pp"


def gain(results: dict, role: str) -> tuple[float, float, float]:
    """(expert SR - baseline SR, CI low, CI high) for one task."""
    expert, baseline = results["experts"][role], results["baseline"][role]
    low, high = newcombe_interval(expert, NUM_EPISODES, baseline, NUM_EPISODES)
    return (expert - baseline) / NUM_EPISODES, low, high


def write_markdown(results: dict, path: Path) -> None:
    """Write comparison.md: the per-run table, then the expert gains."""
    lines = [f"# 100-episode comparison ({PAIRING}, split {SPLIT}, seeds 0–99, no video)", "",
             "| Model | Task | Checkpoint | Successes | SR | Wilson 95% CI |", "|---|---|---|---|---|---|"]
    for model, by_role in results.items():
        for role, successes in by_role.items():
            rate = successes / NUM_EPISODES
            lines.append(f"| {model} | {TASKS[role]} | `{CHECKPOINTS[model][role]}` | {successes}/{NUM_EPISODES} | "
                         f"{percent(rate)} | {interval(*wilson_interval(successes, NUM_EPISODES))} |")
    lines += ["", "## Expert gain over baseline (Newcombe hybrid-score 95% CI)", "",
              "| Task | Expert − baseline | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    for role in TASKS:
        delta, low, high = gain(results, role)
        lines.append(f"| {TASKS[role]} | {points(delta)} | {point_interval(low, high)} | "
                     f"{'yes' if low > 0 or high < 0 else 'no'} |")
    lines += ["", "The models share seeds 0–99, but the policy samples noise on every query, so the runs are treated "
              "as independent samples; with positively correlated outcomes these CIs are conservative."]
    path.write_text("\n".join(lines) + "\n")


def task_header(width: int) -> list[str]:
    """Task names spanning `width` columns each, with a rule under each name."""
    names = " & ".join(rf"\multicolumn{{{width}}}{{c}}{{{TASKS[role]}}}" for role in TASKS)
    rules = " ".join(rf"\cmidrule(lr){{{2 + width * index}-{1 + width * (index + 1)}}}" for index in range(len(TASKS)))
    return [rf" & {names} \\", rules]


def write_latex(results: dict, path: Path) -> None:
    """Write comparison_table.tex: one tabular per section of comparison.md (needs booktabs)."""
    lines = [r"\begin{tabular}{lrrcrrc}", r"\toprule", *task_header(3),
             r"Model" + r" & Succ. & SR & 95\% CI" * len(TASKS) + r" \\", r"\midrule"]
    for model, by_role in results.items():
        cells = [f"{s}/{NUM_EPISODES} & {percent(s / NUM_EPISODES)} & {interval(*wilson_interval(s, NUM_EPISODES))}"
                 for s in by_role.values()]
        lines.append(f"{model} & " + " & ".join(cells).replace("%", r"\%") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\begin{tabular}{lrcrc}", r"\toprule", *task_header(2),
              r"Difference" + r" & $\Delta$ (pp) & 95\% CI" * len(TASKS) + r" \\", r"\midrule"]
    cells = [f"{delta * 100:+.0f} & [{low * 100:+.0f}, {high * 100:+.0f}]"
             for delta, low, high in (gain(results, role) for role in TASKS)]
    lines += [r"experts $-$ baseline & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "eval_n100")
    args = parser.parse_args()
    results = {model: {role: load_run(args.root, model, role) for role in TASKS} for model in CHECKPOINTS}
    write_markdown(results, args.root / "comparison.md")
    write_latex(results, args.root / "comparison_table.tex")
    print((args.root / "comparison.md").read_text())
    print(f"Wrote {args.root / 'comparison.md'} and {args.root / 'comparison_table.tex'}")


if __name__ == "__main__":
    main()
