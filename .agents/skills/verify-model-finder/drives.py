"""The drives for verify-model-finder: each runs the real CLI in its own temp HOME and asserts what a user sees."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / ".venv" / "bin"
STEP_TIMEOUT = 45
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
LIVE: set[subprocess.Popen[str]] = set()


class DriveFailure(Exception):
    pass


def git(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env, check=False
    )
    return done.stdout.strip()


def kill_group(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


@dataclass
class Ctx:
    """One drive's scratch space: a temp HOME and data home, and its evidence directory."""

    name: str
    scratch: Path
    evidence: Path
    steps: int = 0

    @property
    def data_home(self) -> Path:
        return self.scratch / "home" / ".local" / "share"

    def env(self) -> dict[str, str]:
        home = self.scratch / "home"
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "VIRTUAL_ENV", "UV_"))
        }
        return env | {
            "HOME": str(home),
            "XDG_DATA_HOME": str(self.data_home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "PATH": f"{self.scratch / 'tripwire'}:{BIN}:/usr/bin:/bin",
            "PYTHONPATH": str(ROOT),
            "MPLBACKEND": "Agg",
        }

    def run(
        self,
        step: str,
        cmd: list[str],
        cwd: Path | None = None,
        stdin: str = "",
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        self.steps += 1
        stem = self.evidence / f"{self.steps:02d}-{step}"
        child_env, where = env or self.env(), cwd or self.scratch
        proc = subprocess.Popen(
            cmd,
            cwd=where,
            env=child_env,
            text=True,
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        LIVE.add(proc)
        try:
            out, err = proc.communicate(stdin, timeout=STEP_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            out, err = proc.communicate()
            err += f"\nverify: timed out after {STEP_TIMEOUT}s"
        finally:
            LIVE.discard(proc)
        shown = {
            k: "<redacted>" if any(m in k for m in SECRET_MARKERS) else v
            for k, v in child_env.items()
        }
        stem.with_suffix(".cmd").write_text(
            json.dumps({"cmd": cmd, "cwd": str(where), "env": shown}, indent=1)
        )
        stem.with_suffix(".out").write_text(out)
        stem.with_suffix(".err").write_text(err)
        stem.with_suffix(".exit").write_text(f"{proc.returncode}\n")
        return proc.returncode, out, err

    def expect(
        self,
        step: str,
        cmd: list[str],
        code: int,
        text: str,
        cwd: Path | None = None,
        stdin: str = "",
        env: dict[str, str] | None = None,
    ) -> str:
        actual, out, err = self.run(step, cmd, cwd, stdin, env)
        if actual != code or text not in out + err or "Traceback" in err:
            tail = (err.strip() or out.strip()).splitlines()[-1:] or [""]
            raise DriveFailure(
                f"{step}: expected exit {code} and {text!r}, got exit {actual}: {tail[0][:160]}"
            )
        return out


def require(condition: object, message: str) -> None:
    if not condition:
        raise DriveFailure(message)


def demo(*args: str) -> list[str]:
    return [str(BIN / "python"), "-m", "demos.demo_bootstrap", *args]


def cli(*args: str) -> list[str]:
    return [str(BIN / "experiment"), *args]


def py(ctx: Ctx, step: str, code: str) -> str:
    return ctx.expect(step, [str(BIN / "python"), "-c", code], 0, "")


def search(ctx: Ctx) -> Path:
    x = ctx.scratch / "run_a"
    ctx.expect("init-demo", demo("init-demo", "--experiment", str(x)), 0, "")
    for candidate in ("majority-baseline", "logreg-tiny"):
        ctx.expect(
            f"run-{candidate}",
            demo("run-candidate", "--experiment", str(x), "--candidate", candidate),
            0,
            "val_auc",
        )
    return x


def freeze_and_evaluate(ctx: Ctx, x: Path) -> None:
    ctx.expect(
        "freeze",
        demo("freeze-candidate", "--experiment", str(x), "--candidate", "logreg-tiny"),
        0,
        "selected_result",
    )
    ctx.expect(
        "evaluate-final", demo("evaluate-final", "--experiment", str(x)), 0, "holdout_sha256"
    )


class Fixture:
    """Builds the demo experiment once per run, searched and then evaluated. Drives copy a snapshot, since it holds no absolute paths."""

    lock = threading.Lock()
    snapshots: dict[str, Path] = {}  # noqa: RUF012
    ctx = Ctx("fixture", Path(), Path())

    @classmethod
    def copy(cls, ctx: Ctx, stage: str) -> Path:
        with cls.lock:
            if not cls.snapshots:
                x = search(cls.ctx)
                cls.snapshots["searched"] = cls.snapshot("searched", x)
                freeze_and_evaluate(cls.ctx, x)
                cls.snapshots["evaluated"] = cls.snapshot("evaluated", x)
        for part in ("run_a", "home"):
            shutil.rmtree(ctx.scratch / part, ignore_errors=True)
            shutil.copytree(cls.snapshots[stage] / part, ctx.scratch / part, symlinks=True)
        return ctx.scratch / "run_a"

    @classmethod
    def snapshot(cls, stage: str, x: Path) -> Path:
        target = cls.ctx.scratch / stage
        shutil.copytree(x, target / "run_a", symlinks=True)
        shutil.copytree(cls.ctx.scratch / "home", target / "home", symlinks=True)
        return target


def drive_holdout_flow(ctx: Ctx) -> str:
    x = search(ctx)
    require(
        any(ctx.data_home.glob("model-finder/holdouts/*/holdout-*.parquet")),
        "no sealed parquet in the temp data home",
    )
    rows = py(
        ctx,
        "search-rows",
        f"import pandas as p;print(sum(len(p.read_parquet('{x}/data/'+n+'.parquet')) for n in ('train','validation')))",
    ).strip()
    require(rows == "640", f"expected 640 search rows in data/, got {rows}")
    entries = json.loads((x / "results.json").read_text())
    require(
        all(set(e["metrics"]) == {"train", "validation"} for e in entries),
        "results.json holds a metric split other than train and validation",
    )
    ctx.expect(
        "final-before-freeze",
        demo("evaluate-final", "--experiment", str(x)),
        1,
        "No frozen selection",
    )
    freeze_and_evaluate(ctx, x)
    ctx.expect(
        "search-after-final",
        demo("run-candidate", "--experiment", str(x), "--candidate", "logreg-tiny"),
        1,
        "holdout-informed",
    )
    return "160 rows sealed away from 640 search rows; holdout read once, only after a freeze"


def drive_complete(ctx: Ctx) -> str:
    x = Fixture.copy(ctx, "evaluated")
    (x / "outputs/report.md").write_text("logreg-tiny beats the baseline.\n")
    (x / "outputs/reproduce.md").write_text("See README.md.\n")
    review = (
        "record-review",
        "--experiment",
        str(x),
        "--reviewer",
        "verify",
        "--verdict",
        "accepted",
        "--report",
        "outputs/report.md",
        "--limitation",
        "Synthetic data.",
    )
    ctx.expect("record-review", cli(*review), 0, "")
    completion = (
        "record-completion",
        "--experiment",
        str(x),
        "--outcome",
        "supported",
        "--conclusion",
        "logreg-tiny beats the baseline.",
        "--limitation",
        "Synthetic data.",
    )
    picks = (
        "--selected",
        "logreg-tiny",
        "--baseline",
        "majority-baseline",
        "--report",
        "outputs/report.md",
        "--reproduction",
        "outputs/reproduce.md",
    )
    ctx.expect("record-completion", cli(*completion, *picks), 0, "")
    ctx.expect("complete-check", cli("complete-check", str(x)), 0, "")
    with (x / "outputs/final_evaluation.json").open("a") as final:
        final.write(" ")
    ctx.expect("complete-check-edited", cli("complete-check", str(x)), 1, "Stale evaluation")
    return (
        "review and completion recorded; complete-check passes, then refuses an edited final file"
    )


def drive_reseal(ctx: Ctx) -> str:
    x = Fixture.copy(ctx, "evaluated")
    rows = ctx.scratch / "new_rows.parquet"
    # New rows must not repeat search or old holdout rows, so the dates move past both.
    make_rows = "import pandas as pd\nfrom demos.demo_bootstrap.generate_data import generate_demo_bootstrap_dataset as g\n"
    py(
        ctx,
        "new-rows",
        make_rows
        + f"g(n_rows=80, seed=901).assign(event_date=pd.date_range('2027-01-01', periods=80)).to_parquet('{rows}', index=False)",
    )
    ctx.expect(
        "reseal",
        cli("reseal", "--experiment", str(x), "--rows", str(rows), "--note", "Verification run"),
        0,
        "Sealed 80 new rows",
    )
    require(
        (x / "work/holdout_history/0001/consumed.json").is_file(),
        "reseal wrote no work/holdout_history/0001/consumed.json",
    )
    ctx.expect("validate", cli("validate", str(x)), 0, "holdouts retired")
    return "a read holdout is retired to history, 80 new rows sealed, validate warns"


def drive_new(ctx: Ctx) -> str:
    consumer = ctx.scratch / "consumer"
    consumer.mkdir()
    rows = [
        f"2025-{1 + i // 28:02d}-{1 + i % 28:02d},{(i * 37 % 100) / 50 - 1:.2f},{int(i * 37 % 100 > 45)}"
        for i in range(200)
    ]
    (consumer / "rows.csv").write_text("\n".join(["date,signal,label", *rows]) + "\n")
    out = ctx.expect("new", cli("new", "churn", "--csv", "rows.csv"), 0, "", consumer)
    require(
        "Sealed 40 holdout rows" in out and out.index("linear") < out.index("baseline"),
        "new did not seal 40 rows and rank linear above baseline",
    )
    x = consumer / "experiments/churn"
    ctx.expect(
        "candidate-new",
        cli("candidate", "new", "candidate_2", "--experiment", str(x)),
        0,
        "",
        consumer,
    )
    ctx.expect(
        "run-scaffolded",
        [
            str(BIN / "python"),
            str(x / "model/run.py"),
            "run-candidate",
            "--candidate",
            "candidate-2",
        ],
        0,
        "val_auc",
        consumer,
    )
    board = ctx.expect("leaderboard", cli("leaderboard", str(x)), 0, "candidate-2", consumer)
    ranked = [line for line in board.splitlines() if line.strip()[:1].isdigit()]
    require(len(ranked) == 3, f"expected 3 leaderboard rows, got {len(ranked)}")
    ctx.expect("next", cli("next", str(x)), 0, "", consumer)
    return "experiment new on a CSV seals 40 rows; a scaffolded candidate runs; the leaderboard shows 3"


# The control passes only when the missing-experiment check fails with the CLI's own refusal.
CONTROL_FAILURE = "missing-experiment: expected exit 0 and 'does not exist', got exit 1: "
CONTROL_REFUSAL = "/absent: experiment directory does not exist."


def check_missing_experiment(ctx: Ctx, code: int = 1) -> None:
    ctx.expect(
        "missing-experiment",
        demo("evaluate-final", "--experiment", str(ctx.scratch / "absent")),
        code,
        "does not exist",
    )


def drive_replay(ctx: Ctx) -> str:
    x = Fixture.copy(ctx, "searched")
    ctx.expect("replay", cli("replay", str(x)), 0, "replay.html")
    shutil.copy(x / "outputs/replay.html", ctx.evidence / "replay.html")
    require(
        "logreg-tiny" in (x / "outputs/replay.html").read_text(),
        "replay.html does not name logreg-tiny",
    )
    check_missing_experiment(ctx)
    return "replay.html names the leaderboard; a missing experiment is one error line, exit 1"


def drive_wheel(ctx: Ctx) -> str:
    consumer, dist, source = ctx.scratch / "consumer", ctx.scratch / "dist", ctx.scratch / "source"
    consumer.mkdir()
    # setuptools writes build/ and an egg-info next to the project, so build from a copy.
    for name in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
        if (ROOT / name).is_file():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, source / name)
    # uv keeps its managed Python and cache under the real HOME, so uv runs with the real environment, offline.
    real = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "VIRTUAL_ENV"))}
    ctx.expect(
        "build",
        ["uv", "build", "--offline", "--wheel", "--quiet", "--out-dir", str(dist)],
        0,
        "",
        source,
        env=real,
    )
    ctx.expect(
        "venv",
        [
            "uv",
            "venv",
            "--offline",
            "--quiet",
            "--python",
            str(BIN / "python"),
            str(consumer / ".venv"),
        ],
        0,
        "",
        env=real,
    )
    wheel = str(next(dist.glob("*.whl")))
    # Dependencies come from the project's locked environment, not from whatever the
    # shared uv cache holds; a plain path in a .pth adds it without its editable finder.
    consumer_site = next((consumer / ".venv/lib").glob("python*/site-packages"))
    locked_site = next((ROOT / ".venv/lib").glob("python*/site-packages"))
    (consumer_site / "locked-dependencies.pth").write_text(f"{locked_site}\n")
    ctx.expect(
        "install",
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--quiet",
            "--python",
            str(consumer / ".venv/bin/python"),
            wheel,
        ],
        0,
        "",
        env=real,
    )
    env = ctx.env() | {"PATH": f"{ctx.scratch / 'tripwire'}:{consumer / '.venv/bin'}:/usr/bin:/bin"}
    env.pop("PYTHONPATH")
    ctx.expect("git-init", ["git", "init", "--quiet"], 0, "", consumer, env=env)
    exp = str(consumer / ".venv/bin/experiment")
    ctx.expect(
        "setup",
        [exp, "setup", "--repo", str(consumer)],
        0,
        "wrote the guard hook for PreToolUse",
        consumer,
        env=env,
    )
    require(
        "name: research" in (consumer / ".agents/skills/research/SKILL.md").read_text(),
        "the research skill link does not resolve",
    )
    require(
        (consumer / ".claude/skills/experiment-spec/SKILL.md").is_file(), "no .claude/skills link"
    )
    require(
        "Read(//**/model-finder/holdouts/**)" in (consumer / ".claude/settings.json").read_text(),
        "no holdout deny rule in .claude/settings.json",
    )
    ctx.expect(
        "contract",
        [exp, "contract", "evidence"],
        0,
        "What blocks and what labels",
        consumer,
        env=env,
    )
    where = "import model_finder,pathlib;p=pathlib.Path(model_finder.__file__).parent;assert (p/'py.typed').is_file();print(p)"
    origin = ctx.expect(
        "origin", [str(consumer / ".venv/bin/python"), "-c", where], 0, "", consumer, env=env
    ).strip()
    require(
        origin.startswith(str(consumer)), f"engine resolves to {origin}, not the installed wheel"
    )
    hook = json.loads((consumer / ".codex/hooks.json").read_text())["hooks"]["PreToolUse"][0][
        "hooks"
    ][0]["command"]
    read = {
        "tool_name": "Bash",
        "tool_input": {"command": f"cat {ctx.data_home}/model-finder/holdouts/x/holdout-1.parquet"},
    }
    denied = ctx.expect("guard-holdout", ["sh", "-c", hook], 0, "", consumer, json.dumps(read), env)
    require(
        '"permissionDecision": "deny"' in denied,
        f"the Codex guard did not deny a holdout read: {denied[:120]!r}",
    )
    allowed = ctx.expect(
        "guard-allow",
        ["sh", "-c", hook],
        0,
        "",
        consumer,
        json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}}),
        env,
    )
    require(not allowed.strip(), f"the Codex guard refused a plain ls: {allowed[:120]!r}")
    (consumer / "results.json").write_text("[]")
    edit = json.dumps({"tool_input": {"file_path": str(consumer / "results.json")}})
    post_tool_use = json.loads((consumer / ".claude/settings.json").read_text())["hooks"][
        "PostToolUse"
    ]
    post_edit = [
        h["command"] for e in post_tool_use for h in e["hooks"] if "post_edit" in h["command"]
    ]
    require(
        len(post_edit) == 1, f"expected one post-edit hook in .claude/settings.json: {post_edit}"
    )
    ctx.expect(
        "post-edit",
        ["sh", "-c", post_edit[0]],
        0,
        "experiment rebuild-results",
        consumer,
        edit,
        env,
    )
    return "the wheel installs offline; setup links skills, deny rules, and the post-edit hook; the Codex guard refuses a holdout read"


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: tuple[str, ...]
    run: Callable[[Ctx], str]


DRIVES = {
    "holdout-flow": Drive(
        "A demo seals its holdout away from search rows and reads it once, after a freeze",
        ("src/model_finder/*.py", "src/model_finder/cli/runner.py", "demos/**"),
        drive_holdout_flow,
    ),
    "complete": Drive(
        "Review and completion records pass complete-check, which refuses edited evidence",
        (
            "src/model_finder/completion.py",
            "src/model_finder/final_evaluation.py",
            "src/model_finder/cli/main.py",
        ),
        drive_complete,
    ),
    "reseal": Drive(
        "Reseal retires a read holdout and seals new rows",
        (
            "src/model_finder/reseal.py",
            "src/model_finder/holdout.py",
            "src/model_finder/validation.py",
        ),
        drive_reseal,
    ),
    "new": Drive(
        "experiment new on a CSV scaffolds a runnable experiment and leaderboard",
        ("src/model_finder/*.py", "src/model_finder/cli/**", "src/model_finder/templates/**"),
        drive_new,
    ),
    "replay": Drive(
        "Replay renders HTML; a missing experiment fails with one line and no traceback",
        ("src/model_finder/viz/**", "src/model_finder/cli/errors.py"),
        drive_replay,
    ),
    "wheel": Drive(
        "The built wheel installs offline; setup links skills, deny rules, and a working guard",
        (
            "src/model_finder/cli/setup.py",
            "src/model_finder/cli/guard.py",
            "src/model_finder/cli/agent_rules.py",
            "src/model_finder/skills/**",
            "src/model_finder/contract/**",
            "src/model_finder/hooks/**",
            "src/model_finder/cli/hook.py",
            ".agents/hooks/**",
        ),
        drive_wheel,
    ),
}
SMOKE = "holdout-flow"
SHARED = ("uv.lock", "pyproject.toml", ".python-version", ".agents/skills/verify-model-finder/**")
