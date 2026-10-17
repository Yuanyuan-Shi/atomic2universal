"""Build one localhost HTML report with a tab per task set, and print a console summary.

A task set is a folder <results-root>/<name>/ whose role subfolders (i1, i2, ..., and an
optional composite) each hold summary.json and episodes.jsonl. Writes <results-root>/report.html.
Video paths in the page are relative to <results-root>, so serve that directory.

Usage:
    python -m baseline_eval.build_report                  # every task set with results
    python -m baseline_eval.build_report --pairings pnp_cabinet_drawer
    python -m http.server 8080 --directory baseline_eval/results
"""

import argparse
import html
import json
from pathlib import Path

def role_sort_key(role: str) -> tuple[int, str]:
    """Order atomic roles (i1, i2, ...) first and the composite role last."""
    return (1 if role == "composite" else 0, role)


def role_label(role: str) -> str:
    return "Composite" if role == "composite" else f"{role.upper()} · atomic"

CSS = """
:root { color-scheme: light dark; --bg:#f6f8fa; --card:#ffffff; --text:#1f2328; --muted:#656d76;
        --line:#d0d7de; --ok:#1a7f37; --bad:#cf222e; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --card:#161b22; --text:#e6edf3; --muted:#8d96a0;
          --line:#30363d; --ok:#3fb950; --bad:#f85149; --accent:#4493f8; } }
body { margin:0; padding:24px; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }
h1 { font-size:20px; margin:0 0 16px; }
.tabs { display:flex; flex-wrap:wrap; gap:4px; border-bottom:1px solid var(--line); margin-bottom:20px; }
.tab { font:inherit; color:var(--muted); background:none; border:1px solid transparent; border-bottom:none;
       border-radius:6px 6px 0 0; padding:8px 14px; cursor:pointer; margin-bottom:-1px; }
.tab.active { color:var(--text); background:var(--card); border-color:var(--line); font-weight:600; }
.tab small { color:var(--muted); font-weight:400; margin-left:6px; }
[hidden] { display:none !important; }
.meta { color:var(--muted); margin:0 0 16px; }
.note { color:var(--muted); font-size:12px; margin:8px 0 28px; }
table { border-collapse:collapse; background:var(--card); }
th, td { border:1px solid var(--line); padding:6px 10px; text-align:left; }
section { margin-bottom:32px; }
section header { display:flex; flex-wrap:wrap; align-items:baseline; gap:12px;
                 border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:12px; }
h2 { font-size:16px; margin:0; }
.tag { font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:1px 6px; }
.rate { margin-left:auto; font-weight:600; }
.rate small { font-weight:400; color:var(--muted); margin-left:8px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; }
.card { margin:0; background:var(--card); border:2px solid var(--line); border-radius:8px; overflow:hidden; }
.card.success { border-color:var(--ok); }
.card.fail { border-color:var(--bad); }
video { display:block; width:100%; background:#000; }
figcaption { display:flex; justify-content:space-between; gap:8px; padding:6px 8px; font-size:12px; color:var(--muted); }
.badge { font-weight:700; }
.badge.success { color:var(--ok); }
.badge.fail { color:var(--bad); }
"""

SCRIPT = """
const tabs = Array.from(document.querySelectorAll(".tab"));
function showTab(tabId) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  document.querySelectorAll(".panel").forEach((panel) => {
    const isActive = panel.id === tabId;
    panel.hidden = !isActive;
    if (!isActive) panel.querySelectorAll("video").forEach((video) => video.pause());
  });
  history.replaceState(null, "", "#" + tabId);
}
tabs.forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
const requested = location.hash.slice(1);
showTab(tabs.some((tab) => tab.dataset.tab === requested) ? requested : tabs[0].dataset.tab);
"""


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


def load_role(role_dir: Path) -> tuple[dict, list[dict]]:
    """Load one role's summary and its episodes, sorted by episode index."""
    summary_path = role_dir / "summary.json"
    episodes_path = role_dir / "episodes.jsonl"
    if not summary_path.exists() or not episodes_path.exists():
        raise SystemExit(f"ERROR: missing summary.json or episodes.jsonl in {role_dir}. Run run_baseline first.")
    summary = json.loads(summary_path.read_text())
    lines = episodes_path.read_text().splitlines()
    episodes = [json.loads(line) for line in lines if line.strip()]
    episodes.sort(key=lambda episode: episode["episode"])
    return summary, episodes


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def format_interval(summary: dict) -> str:
    low, high = summary["ci95"]
    return f"[{low:.0%}, {high:.0%}]"


def summary_row(summary: dict) -> str:
    return (
        f"<tr><td>{role_label(summary['role'])}</td><td>{html.escape(summary['task'])}</td>"
        f"<td>{html.escape(summary['split'])}</td>"
        f"<td>{summary['num_success']}/{summary['num_episodes']}</td>"
        f"<td>{format_percent(summary['success_rate'])}</td><td>{format_interval(summary)}</td>"
        f"<td>{format_percent(summary.get('reference_sr'))}</td>"
        f"<td>{html.escape(summary.get('reference_note') or '')}</td></tr>"
    )


def episode_card(episode: dict, pairing: str) -> str:
    status = "success" if episode["success"] else "fail"
    video_path = html.escape(f"{pairing}/{episode['video']}")
    caption = f"episode {episode['episode']:02d} · seed {episode['seed']} · {episode['num_steps']} steps"
    return (
        f'<figure class="card {status}">'
        f'<video src="{video_path}" controls muted loop playsinline preload="metadata"></video>'
        f'<figcaption><span>{caption}</span><span class="badge {status}">{status.upper()}</span></figcaption>'
        f"</figure>"
    )


def role_section(summary: dict, episodes: list[dict], pairing: str) -> str:
    cards = "\n".join(episode_card(episode, pairing) for episode in episodes)
    return (
        f"<section><header><h2>{html.escape(summary['task'])}</h2>"
        f'<span class="tag">{role_label(summary["role"])}</span>'
        f'<span class="rate">{format_percent(summary["success_rate"])}'
        f"<small>95% CI {format_interval(summary)} · reference {format_percent(summary.get('reference_sr'))}</small>"
        f'</span></header><div class="grid">{cards}</div></section>'
    )


def pairing_panel(pairing: str, loaded: list[tuple[dict, list[dict]]]) -> str:
    summaries = [summary for summary, _ in loaded]
    rows = "\n".join(summary_row(summary) for summary in summaries)
    sections = "\n".join(role_section(summary, episodes, pairing) for summary, episodes in loaded)
    checkpoint = html.escape(summaries[0]["checkpoint"])
    return f"""<div class="panel" id="{html.escape(pairing)}">
<p class="meta">Pairing: {html.escape(pairing)} · checkpoint: {checkpoint}</p>
<table><tr><th>Role</th><th>Task</th><th>Split</th><th>Success</th><th>SR</th><th>95% CI</th>
<th>Reference</th><th>Reference note</th></tr>
{rows}
</table>
<p class="note">With 10 episodes per task the 95% CI spans roughly ±25–30 points; treat these as a sanity check.</p>
{sections}
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
<title>Baseline closed-loop evaluation</title><style>{CSS}</style></head>
<body>
<h1>Baseline closed-loop evaluation</h1>
<nav class="tabs">{buttons}</nav>
{bodies}
<script>{SCRIPT}</script>
</body></html>"""


def print_console_table(pairing: str, summaries: list[dict]) -> None:
    print(f"\n== {pairing} ==")
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
    args = parser.parse_args()

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
        panels.append((pairing, len(roles), pairing_panel(pairing, loaded)))
        print_console_table(pairing, summaries)

    report_path = args.results_root / "report.html"
    report_path.write_text(render_page(panels), encoding="utf-8")
    tab_word = "tab" if len(panels) == 1 else "tabs"
    print(f"\nReport: {report_path}  ({len(panels)} {tab_word})")
    print(f"View:   python -m http.server 8080 --directory {args.results_root}")
    print("        then open http://localhost:8080/report.html")


if __name__ == "__main__":
    main()
