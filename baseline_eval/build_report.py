"""Build one localhost HTML report with a tab per task set, and print a console summary.

A task set is a folder <results-root>/<name>/ whose role subfolders (i1, i2, ..., and an
optional composite) each hold summary.json and episodes.jsonl. Writes <results-root>/report.html.
Video paths in the page are relative to <results-root>, so serve that directory.

Below the baseline, each tab gets an "RL Expert closed-loop evaluation" section when the experts
config (configs/eval/rl_experts.yaml) lists the task set, and above it a 100-episode RL expert vs
baseline table when eval_n100/ holds both runs for a role (see report_rl.py).

Usage:
    python -m baseline_eval.build_report                  # every task set with results
    python -m baseline_eval.build_report --pairings pnp_cabinet_drawer
    python -m http.server 8080 --directory baseline_eval/results
"""

import argparse
import html
from pathlib import Path

from baseline_eval.report_common import (CSS, SCRIPT, SMALL_SAMPLE_NOTE, TABLE_HEADER, format_interval, format_percent,
                                         load_role, role_label, role_section, summary_row)
from baseline_eval.report_rl import (DEFAULT_EVAL_N100_ROOT, DEFAULT_EXPERTS_CONFIG, eval_n100_block, expert_block,
                                     load_eval_n100, load_experts, load_experts_config)


def role_sort_key(role: str) -> tuple[int, str]:
    """Order atomic roles (i1, i2, ...) first and the composite role last."""
    return (1 if role == "composite" else 0, role)


def find_roles(pairing_dir: Path) -> list[str]:
    """Return the role subfolders of one task set that contain a summary.json."""
    roles = [folder.name for folder in pairing_dir.iterdir() if (folder / "summary.json").exists()]
    return sorted(roles, key=role_sort_key)


def discover_pairings(results_root: Path) -> list[str]:
    """Return task-set folders (not starting with '_') that contain at least one role summary."""
    pairings = []
    for folder in sorted(results_root.iterdir()):
        if folder.is_dir() and not folder.name.startswith("_") and find_roles(folder):
            pairings.append(folder.name)
    return pairings


def pairing_panel(pairing: str, loaded: list[tuple[dict, list[dict]]], experts: dict | None, eval_n100: dict) -> str:
    summaries = [summary for summary, _ in loaded]
    rows = "\n".join(summary_row(summary, role_label(summary["role"])) for summary in summaries)
    sections = "\n".join(
        role_section(summary, episodes, pairing, role_label(summary["role"])) for summary, episodes in loaded
    )
    checkpoint = html.escape(summaries[0]["checkpoint"])
    expert_html = expert_block(pairing, loaded, experts) if experts is not None else ""
    eval_n100_html = eval_n100_block(eval_n100) if eval_n100 else ""
    return f"""<div class="panel" id="{html.escape(pairing)}">
{eval_n100_html}
<h1 class="section-title">Baseline closed-loop evaluation</h1>
<p class="meta">Pairing: {html.escape(pairing)} · checkpoint: {checkpoint}</p>
<table>{TABLE_HEADER}
{rows}
</table>
{SMALL_SAMPLE_NOTE}
{sections}
{expert_html}
</div>"""


def tab_button(pairing: str, num_tasks: int) -> str:
    return (
        f'<button class="tab" data-tab="{html.escape(pairing)}">'
        f"{html.escape(pairing)}<small>{num_tasks} tasks</small></button>"
    )


def render_page(panels: list[tuple[str, int, str]]) -> str:
    buttons = "\n".join(tab_button(pairing, num_tasks) for pairing, num_tasks, _ in panels)
    bodies = "\n".join(body for _, _, body in panels)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-loop evaluation</title><style>{CSS}</style></head>
<body>
<h1>Closed-loop evaluation</h1>
<nav class="tabs">{buttons}</nav>
{bodies}
<script>{SCRIPT}</script>
</body></html>"""


def print_console_table(heading: str, summaries: list[dict]) -> None:
    print(f"\n== {heading} ==")
    print(f"{'Role':<10} {'Task':<28} {'Split':<8} {'Success':>8} {'SR':>6} {'95% CI':>14} {'Ref':>6}")
    for summary in summaries:
        success_text = f"{summary['num_success']}/{summary['num_episodes']}"
        print(
            f"{summary['role']:<10} {summary['task']:<28} {summary['split']:<8} {success_text:>8} "
            f"{format_percent(summary['success_rate']):>6} {format_interval(summary):>14} "
            f"{format_percent(summary.get('reference_sr')):>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairings", nargs="+", help="task-set folder names; default: all with results")
    parser.add_argument("--results-root", type=Path, default=Path("baseline_eval/results"))
    parser.add_argument("--experts-config", type=Path, default=DEFAULT_EXPERTS_CONFIG)
    parser.add_argument("--eval-n100-root", type=Path, default=DEFAULT_EVAL_N100_ROOT)
    args = parser.parse_args()
    experts_config = load_experts_config(args.experts_config)

    pairings = args.pairings or discover_pairings(args.results_root)
    if not pairings:
        raise SystemExit(f"ERROR: no task set with results found under {args.results_root}.")

    panels = []
    for pairing in pairings:
        pairing_dir = args.results_root / pairing
        roles = find_roles(pairing_dir) if pairing_dir.is_dir() else []
        if not roles:
            raise SystemExit(f"ERROR: no role results found in {pairing_dir}. Run run_baseline first.")
        loaded = [load_role(pairing_dir / role) for role in roles]
        summaries = [summary for summary, _ in loaded]
        experts = None
        if pairing in experts_config:
            experts = load_experts(args.results_root, pairing, experts_config[pairing], summaries)
        eval_n100 = load_eval_n100(args.eval_n100_root, pairing, roles)
        panels.append((pairing, len(roles), pairing_panel(pairing, loaded, experts, eval_n100)))
        print_console_table(pairing, summaries)
        if experts:
            print_console_table(f"{pairing} · RL experts", [summary for summary, _ in experts.values()])

    report_path = args.results_root / "report.html"
    report_path.write_text(render_page(panels), encoding="utf-8")
    tab_word = "tab" if len(panels) == 1 else "tabs"
    print(f"\nReport: {report_path}  ({len(panels)} {tab_word})")
    print(f"View:   python -m http.server 8080 --directory {args.results_root}")
    print("        then open http://localhost:8080/report.html")


if __name__ == "__main__":
    main()
