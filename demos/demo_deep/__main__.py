from __future__ import annotations

from pathlib import Path

from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.runner import ExperimentSeed
from demos.common import CODE_ROOT
from demos.demo_deep.data import TARGET_COLUMN, TIME_COLUMN
from demos.demo_deep.generate_data import generate_demo_dataset
from demos.demo_deep.modeling import CANDIDATE_RUNNERS

_HERE = Path(__file__).resolve().parent

CONFIG = RunnerConfig(
    seed=ExperimentSeed(
        experiment_id="demo_deep",
        template_path=_HERE / "experiment_template.md",
    ),
    candidate_runners=CANDIDATE_RUNNERS,
    rows_loader=lambda seed: generate_demo_dataset(seed=seed),
    rows_source=None,
    time_column=TIME_COLUMN,
    target_column=TARGET_COLUMN,
    code_root=CODE_ROOT,
)

if __name__ == "__main__":
    raise SystemExit(run_runner_main(CONFIG))
