"""Bundle the renderer and a replay script into one self-contained HTML file."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_ml_loop.viz.generate import replay_script

RENDERER_DIR = Path(__file__).parent / "renderer"
JS_FILES = ["room.js", "character.js", "sound.js", "ui.js", "script.js", "engine.js"]


def _script_tag_safe_json(raw_json: str) -> str:
    """Escape HTML-significant characters inside embedded JSON script tags."""
    return raw_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def bundle(script_data: str, output: Path) -> None:
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
    html = html.replace("</body>", f"<script>\n{combined_js}\n</script>\n</body>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def write_replay(experiment_dir: Path, output: Path) -> Path:
    bundle(json.dumps(replay_script(experiment_dir), indent=2), output)
    return output
