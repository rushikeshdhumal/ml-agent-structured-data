from __future__ import annotations

import pandas as pd
import pytest

import ds_crew.tools.logging_tools as logging_tools
from ds_crew import settings
from ds_crew.main import build_arg_parser, main
from ds_crew.tools.logging_tools import estimate_cost_usd


def test_requires_data_and_target():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_rejects_bad_task_choice():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--data", "x.csv", "--target", "y", "--task", "nonsense"])


def test_accepts_valid_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--data", "x.csv", "--target", "y", "--task", "regression"])
    assert args.target == "y"
    assert args.task == "regression"
    assert str(args.data) == "x.csv"


def test_main_errors_on_missing_data_file(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(SystemExit):
        main(["--data", str(missing), "--target", "y"])
    assert "does not exist" in capsys.readouterr().err


def test_main_errors_on_unknown_target_column(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        main(["--data", str(csv_path), "--target", "not_a_column"])
    assert "not found in columns" in capsys.readouterr().err


def test_main_errors_on_metric_disallowed_for_resolved_task_type(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3], "target": [4, 5, 6]}).to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--data",
                str(csv_path),
                "--target",
                "target",
                "--task",
                "regression",
                "--metric",
                "accuracy",
            ]
        )
    assert "not allowed for task type" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# LLM usage / cost accounting
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens=1_000_000, completion_tokens=500_000):
        self.total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_prompt_tokens = 0
        self.successful_requests = 42


class _FakeCrew:
    def __init__(self, usage=None, raises=False):
        self._usage = usage or _FakeUsage()
        self._raises = raises

    def calculate_usage_metrics(self):
        if self._raises:
            raise RuntimeError("simulated usage-accounting failure")
        return self._usage


def test_estimate_cost_is_none_when_rates_unset(monkeypatch):
    # An unpriced run must record "not priced", never a $0.00 claim -- a
    # hardcoded zero would assert the run was free, which is usually wrong.
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    assert estimate_cost_usd(1_000_000, 500_000) is None


def test_estimate_cost_is_none_when_only_one_rate_set(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", 0.15)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    assert estimate_cost_usd(1_000_000, 500_000) is None


def test_estimate_cost_arithmetic(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", 0.15)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", 0.60)
    # 1.0M input @ 0.15 + 0.5M output @ 0.60 = 0.15 + 0.30
    assert estimate_cost_usd(1_000_000, 500_000) == pytest.approx(0.45)


def test_log_llm_usage_records_tokens_and_wall_clock(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    captured = {}
    monkeypatch.setattr(logging_tools, "log_metrics", lambda rid, m: captured.update(m))

    logging_tools.log_llm_usage("mlflow-run", _FakeCrew(), wall_clock_s=1326.94)

    assert captured["total_tokens"] == 1_500_000
    assert captured["successful_requests"] == 42
    assert captured["wall_clock_s"] == 1326.9
    assert "estimated_cost_usd" not in captured


def test_log_llm_usage_includes_cost_when_priced(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", 0.15)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", 0.60)
    captured = {}
    monkeypatch.setattr(logging_tools, "log_metrics", lambda rid, m: captured.update(m))

    logging_tools.log_llm_usage("mlflow-run", _FakeCrew(), wall_clock_s=10.0)

    assert captured["estimated_cost_usd"] == pytest.approx(0.45)


def test_log_llm_usage_never_raises(monkeypatch, capsys):
    # It runs inside a `finally`, so an exception escaping here would replace
    # whatever real error was propagating out of kickoff -- trading a
    # diagnosable pipeline failure for a confusing logging one.
    logging_tools.log_llm_usage("mlflow-run", _FakeCrew(raises=True), wall_clock_s=1.0)
    assert "could not record LLM usage" in capsys.readouterr().out
