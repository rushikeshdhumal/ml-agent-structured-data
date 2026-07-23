from __future__ import annotations

import pandas as pd
import pytest

from ds_crew.main import build_arg_parser, main


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
