"""Shared pieces of the HTML report: page style and script, loading a role's results, and the
table rows, video cards and role sections used by the baseline and RL expert parts of a tab."""

import html
import json
from pathlib import Path


def role_label(role: str) -> str:
    return "Composite" if role == "composite" else f"{role.upper()} · atomic"


def expert_label(role: str) -> str:
    return f"{role.upper()} · RL expert"


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
h1.section-title { margin:40px 0 16px; padding-top:24px; border-top:2px solid var(--line); }
.panel > h1.section-title:first-child { margin-top:0; padding-top:0; border-top:none; }
tr.pending td { color:var(--muted); font-style:italic; }
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


def summary_row(summary: dict, label: str) -> str:
    return (
        f"<tr><td>{label}</td><td>{html.escape(summary['task'])}</td>"
        f"<td>{html.escape(summary['split'])}</td>"
        f"<td>{summary['num_success']}/{summary['num_episodes']}</td>"
        f"<td>{format_percent(summary['success_rate'])}</td><td>{format_interval(summary)}</td>"
        f"<td>{format_percent(summary.get('reference_sr'))}</td>"
        f"<td>{html.escape(summary.get('reference_note') or '')}</td></tr>"
    )


def episode_card(episode: dict, video_root: str) -> str:
    status = "success" if episode["success"] else "fail"
    video_path = html.escape(f"{video_root}/{episode['video']}")
    caption = f"episode {episode['episode']:02d} · seed {episode['seed']} · {episode['num_steps']} steps"
    return (
        f'<figure class="card {status}">'
        f'<video src="{video_path}" controls muted loop playsinline preload="metadata"></video>'
        f'<figcaption><span>{caption}</span><span class="badge {status}">{status.upper()}</span></figcaption>'
        f"</figure>"
    )


def role_section(summary: dict, episodes: list[dict], video_root: str, label: str, baseline: dict | None = None) -> str:
    cards = "\n".join(episode_card(episode, video_root) for episode in episodes)
    baseline_text = f" · baseline {format_percent(baseline['success_rate'])}" if baseline else ""
    return (
        f"<section><header><h2>{html.escape(summary['task'])}</h2>"
        f'<span class="tag">{label}</span>'
        f'<span class="rate">{format_percent(summary["success_rate"])}'
        f"<small>95% CI {format_interval(summary)}{baseline_text}"
        f" · reference {format_percent(summary.get('reference_sr'))}</small>"
        f'</span></header><div class="grid">{cards}</div></section>'
    )


TABLE_HEADER = (
    "<tr><th>Role</th><th>Task</th><th>Split</th><th>Success</th><th>SR</th><th>95% CI</th>\n"
    "<th>Reference</th><th>Reference note</th></tr>"
)
SMALL_SAMPLE_NOTE = (
    '<p class="note">With 10 episodes per task the 95% CI spans roughly ±25–30 points; treat these as a sanity check.</p>'
)
