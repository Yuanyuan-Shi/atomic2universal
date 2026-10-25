"""RL expert parts of the HTML report.

The "RL Expert closed-loop evaluation" section of a tab lists one run_baseline --results-root folder
per role from the experts config (configs/eval/rl_experts.yaml), or null for a pending row; expert
videos are reached through symlinks in <results-root>/_rl_experts/<pairing>/<role>, refreshed here.
The 100-episode table at the top of a tab compares <eval-n100-root>/{baseline,experts}/<pairing>/<role>
(Stage 2b runs, no video) for each role that has both runs.
"""

import html
import json
import os
from pathlib import Path

import yaml

from baseline_eval.report_common import (SMALL_SAMPLE_NOTE, TABLE_HEADER, expert_label, format_interval, format_percent,
                                         load_role, role_section, summary_row)
from evaluation.config import PROJECT_ROOT
from evaluation.stats import newcombe_interval

DEFAULT_EXPERTS_CONFIG = PROJECT_ROOT / "configs" / "eval" / "rl_experts.yaml"
EXPERT_LINKS_DIR = "_rl_experts"
DEFAULT_EVAL_N100_ROOT = PROJECT_ROOT / "eval_n100"


def load_experts_config(path: Path) -> dict[str, dict[str, str | None]]:
    """Return {pairing: {role: results folder or None}}; an absent file means no expert sections."""
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text()) or {}
    if not isinstance(content, dict) or not all(isinstance(roles, dict) for roles in content.values()):
        raise SystemExit(f"ERROR: {path} must map each task set to {{role: results folder or null}}.")
    return content


def link_expert_role(results_root: Path, pairing: str, role: str, role_dir: Path) -> None:
    """Point <results-root>/_rl_experts/<pairing>/<role> at the expert's role folder (relative symlink)."""
    link = results_root / EXPERT_LINKS_DIR / pairing / role
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise SystemExit(f"ERROR: {link} exists and is not a symlink; move it away.")
    link.symlink_to(os.path.relpath(role_dir.resolve(), link.parent.resolve()), target_is_directory=True)


def load_experts(
    results_root: Path, pairing: str, roles: dict[str, str | None], baselines: list[dict]
) -> dict[str, tuple[dict, list[dict]]]:
    """Load each configured expert evaluation of one task set (null roles stay pending)."""
    tasks = {summary["role"]: summary["task"] for summary in baselines}
    experts = {}
    for role, folder in roles.items():
        if folder is None:
            continue
        if role not in tasks:
            raise SystemExit(f"ERROR: experts config lists role {role} of {pairing}, which has no baseline results.")
        folder_path = Path(folder) if Path(folder).is_absolute() else PROJECT_ROOT / folder
        role_dir = folder_path / pairing / role
        summary, episodes = load_role(role_dir)
        if summary["task"] != tasks[role]:
            raise SystemExit(f"ERROR: {role_dir} evaluates {summary['task']}, but the baseline {role} is {tasks[role]}.")
        link_expert_role(results_root, pairing, role, role_dir)
        experts[role] = (summary, episodes)
    return experts


def pending_row(baseline: dict) -> str:
    """Placeholder row for a role whose expert has no evaluation yet."""
    return (
        f'<tr class="pending"><td>{expert_label(baseline["role"])}</td><td>{html.escape(baseline["task"])}</td>'
        f"<td>{html.escape(baseline['split'])}</td><td>pending (expert training)</td><td>—</td><td>—</td>"
        f"<td>{format_percent(baseline.get('reference_sr'))}</td>"
        f"<td>{html.escape(baseline.get('reference_note') or '')}</td></tr>"
    )


def expert_block(pairing: str, loaded: list[tuple[dict, list[dict]]], experts: dict) -> str:
    """The RL expert section of one tab: a row per baseline role, videos for evaluated experts."""
    rows, sections, checkpoints = [], [], []
    for baseline, _ in loaded:
        role = baseline["role"]
        if role not in experts:
            rows.append(pending_row(baseline))
            checkpoints.append(f"{role.upper()} pending")
            continue
        summary, episodes = experts[role]
        rows.append(summary_row(summary, expert_label(role)))
        video_root = f"{EXPERT_LINKS_DIR}/{pairing}"
        sections.append(role_section(summary, episodes, video_root, expert_label(role), baseline))
        seeds = f"seeds {summary['seed_start']}–{summary['seed_start'] + summary['num_episodes'] - 1}"
        checkpoints.append(f"{role.upper()} {html.escape(summary['checkpoint'])} ({seeds})")
    return f"""<h1 class="section-title">RL Expert closed-loop evaluation</h1>
<p class="meta">Pairing: {html.escape(pairing)} · checkpoints: {" · ".join(checkpoints)}</p>
<table>{TABLE_HEADER}
{chr(10).join(rows)}
</table>
{SMALL_SAMPLE_NOTE}
{chr(10).join(sections)}"""


def load_eval_n100(root: Path, pairing: str, roles: list[str]) -> dict[str, tuple[dict, dict]]:
    """{role: (baseline summary, expert summary)} for the roles with both 100-episode runs."""
    runs = {}
    for role in roles:
        paths = [root / model / pairing / role / "summary.json" for model in ("baseline", "experts")]
        if not all(path.is_file() for path in paths):
            continue
        baseline, expert = (json.loads(path.read_text()) for path in paths)
        keys = ("task", "split", "seed_start", "num_episodes")
        if any(baseline[key] != expert[key] for key in keys):
            raise SystemExit(f"ERROR: the baseline and expert runs of {pairing} {role} in {root} differ in {keys}.")
        runs[role] = (baseline, expert)
    return runs


def run_cell(summary: dict) -> str:
    return (f"{summary['num_success']}/{summary['num_episodes']} "
            f"({format_percent(summary['success_rate'])}, {format_interval(summary)})")


def eval_n100_block(runs: dict[str, tuple[dict, dict]]) -> str:
    """The 100-episode RL expert vs baseline table (first in the tab)."""
    rows = []
    for baseline, expert in runs.values():
        low, high = newcombe_interval(expert["num_success"], expert["num_episodes"],
                                      baseline["num_success"], baseline["num_episodes"])
        gain = expert["success_rate"] - baseline["success_rate"]
        rows.append(f"<tr><td>{html.escape(baseline['task'])}</td><td>{run_cell(baseline)}</td>"
                    f"<td>{html.escape(Path(expert['checkpoint']).name)}: {run_cell(expert)}</td>"
                    f"<td>{gain * 100:+.0f} pp [{low * 100:+.0f}, {high * 100:+.0f}]</td></tr>")
    first = next(iter(runs.values()))[0]
    baselines = " / ".join(sorted({html.escape(Path(baseline["checkpoint"]).name) for baseline, _ in runs.values()}))
    seeds = f"seeds {first['seed_start']}–{first['seed_start'] + first['num_episodes'] - 1}"
    return f"""<h1 class="section-title">100-episode closed-loop evaluation: RL expert vs baseline</h1>
<p class="meta">{seeds} · {first['num_episodes']} episodes per run · split {html.escape(first['split'])} · no video · eval_n100/</p>
<table><tr><th>Task</th><th>Baseline ({baselines})</th><th>RL expert</th><th>Expert − baseline (Newcombe 95% CI)</th></tr>
{chr(10).join(rows)}
</table>
<p class="note">Successes with Wilson 95% CIs. The gain CI treats the two runs as independent samples; they share
seeds, so it is conservative.</p>"""
