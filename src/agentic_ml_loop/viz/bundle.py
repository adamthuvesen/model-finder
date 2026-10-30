"""Bundle the renderer and a replay script into one self-contained HTML file."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from agentic_ml_loop.final_evaluation import leaderboard_label
from agentic_ml_loop.viz.generate import replay_script

RENDERER_DIR = Path(__file__).parent / "renderer"
JS_FILES = ["room.js", "character.js", "sound.js", "ui.js", "script.js", "engine.js"]


def _script_tag_safe_json(raw_json: str) -> str:
    """Escape HTML-significant characters inside embedded JSON script tags."""
    return raw_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _label_banner(label: str) -> str:
    return (
        '<div id="exploratory-label" style="position:fixed;top:0;left:0;right:0;z-index:9999;'
        'padding:6px 12px;background:#7a4b00;color:#fff;font:14px sans-serif;text-align:center">'
        f"{escape(label)}</div>"
    )


def bundle(script_data: str, output: Path, label: str | None = None) -> None:
    html = (RENDERER_DIR / "index.html").read_text(encoding="utf-8")
    combined_js = "\n\n".join(
        f"// === {name} ===\n" + (RENDERER_DIR / name).read_text(encoding="utf-8")
        for name in JS_FILES
    )
    html = html.replace(
        '<script id="script-data" type="application/json">null</script>',
        f'<script id="script-data" type="application/json">{_script_tag_safe_json(script_data)}</script>',
    )
    for name in JS_FILES:
        html = html.replace(f'<script src="{name}"></script>', "")
    banner = _label_banner(label) if label else ""
    html = html.replace("</body>", f"{banner}<script>\n{combined_js}\n</script>\n</body>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def write_replay(experiment_dir: Path, output: Path) -> Path:
    script = json.dumps(replay_script(experiment_dir), indent=2)
    bundle(script, output, leaderboard_label(experiment_dir))
    return output
