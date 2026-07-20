"""End-to-end pipeline test. Requires a live LLM (set MODEL + the matching
API key env var) and actually calls the model, so it is excluded from the
default test run (see `addopts` in pyproject.toml) and only runs via:

    uv run pytest tests/e2e -m e2e

Set AUTO_APPROVE=1 so the human_input checkpoints don't block on stdin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_crew import settings
from ds_crew.main import main


@pytest.mark.e2e
def test_full_pipeline_runs_end_to_end_with_auto_approve(tmp_path, monkeypatch):
    # `settings.AUTO_APPROVE` is read from the environment once at import time, so
    # setting the env var here would have no effect on the already-imported module --
    # patch the attribute directly so crew.py's `not settings.AUTO_APPROVE` picks it up.
    monkeypatch.setattr(settings, "AUTO_APPROVE", True)
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.normal(5, 2, n),
            "cat_a": rng.choice(["x", "y", "z"], n),
            "target": rng.choice(["yes", "no"], n),
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    exit_code = main(["--data", str(csv_path), "--target", "target", "--task", "classification"])
    assert exit_code == 0
