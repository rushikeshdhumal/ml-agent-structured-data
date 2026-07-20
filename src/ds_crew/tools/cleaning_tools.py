"""Deterministic data-cleaning execution. Mutates the run's dataset.

Only ever called by the "execute" half of the cleaning HITL checkpoint, with
a plan a human has already approved (see config/tasks.yaml). The target
column can never be touched by a CleaningPlan -- refused both here and by
the propose-stage guardrail (defense in depth).
"""

from __future__ import annotations

import json

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel
from sklearn.impute import KNNImputer

from ds_crew.schemas import CleaningPlan, ColumnCleaningAction
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_metrics, log_plan_and_feedback

_DTYPE_CAST_MAP = {
    "int": "int64",
    "float": "float64",
    "str": "str",
    "category": "category",
    "bool": "bool",
}


def _apply_missing_strategy(df: pd.DataFrame, col: str, action: ColumnCleaningAction) -> pd.DataFrame:
    strategy = action.missing_strategy
    if strategy == "drop_rows":
        return df.dropna(subset=[col])
    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"missing_strategy 'mean' requires a numeric column; '{col}' is not.")
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"missing_strategy 'median' requires a numeric column; '{col}' is not.")
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
    elif strategy == "constant":
        if action.constant_fill_value is None:
            raise ValueError(f"missing_strategy 'constant' for '{col}' requires constant_fill_value.")
        df[col] = df[col].fillna(action.constant_fill_value)
    elif strategy == "ffill":
        df[col] = df[col].ffill()
    elif strategy == "bfill":
        df[col] = df[col].bfill()
    elif strategy == "knn":
        df = _knn_impute_column(df, col)
    return df


def _knn_impute_column(df: pd.DataFrame, col: str, n_neighbors: int = 5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if col not in numeric_cols:
        raise ValueError(f"missing_strategy 'knn' requires a numeric column; '{col}' is not.")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def _apply_outlier_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    if strategy == "none":
        return df
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"outlier_strategy '{strategy}' requires a numeric column; '{col}' is not.")
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    if strategy == "iqr_clip":
        df[col] = df[col].clip(lower, upper)
    elif strategy == "zscore_clip":
        mean, std = df[col].mean(), df[col].std()
        if std and std > 0:
            df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)
    elif strategy == "drop":
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def _apply_dtype_cast(df: pd.DataFrame, col: str, cast: str) -> pd.DataFrame:
    if cast == "datetime":
        df[col] = pd.to_datetime(df[col])
    else:
        df[col] = df[col].astype(_DTYPE_CAST_MAP[cast])
    return df


def apply_cleaning(df: pd.DataFrame, plan: CleaningPlan) -> pd.DataFrame:
    """Pure function: applies a validated CleaningPlan to a copy of df."""
    df = df.copy()
    if plan.drop_duplicate_rows:
        df = df.drop_duplicates()
    for action in plan.actions:
        if action.missing_strategy:
            df = _apply_missing_strategy(df, action.column, action)
        df = _apply_outlier_strategy(df, action.column, action.outlier_strategy)
        if action.dtype_cast:
            df = _apply_dtype_cast(df, action.column, action.dtype_cast)
    if plan.columns_to_drop:
        df = df.drop(columns=[c for c in plan.columns_to_drop if c in df.columns])
    return df


class ApplyCleaningPlanTool(BaseTool):
    name: str = "apply_cleaning_plan"
    description: str = (
        "MUTATES the run's dataset by applying a human-approved CleaningPlan. Validates "
        "every column and strategy against the current dataset first and refuses to touch "
        "the target column. Irreversible for this run -- call EXACTLY ONCE with the "
        "complete human-approved plan; a second call is refused."
    )
    args_schema: type[BaseModel] = CleaningPlan
    run_id: str = ""

    def _run(self, **plan_kwargs) -> str:
        plan_kwargs.setdefault("run_id", self.run_id)
        plan = CleaningPlan(**plan_kwargs)
        state = get_data_store().get(self.run_id)

        if state.cleaning_applied:
            return json.dumps(
                {
                    "error": "Cleaning has already been applied for this run. Do not call "
                    "apply_cleaning_plan again -- proceed to the next task."
                }
            )

        unknown = sorted(
            {a.column for a in plan.actions if a.column not in state.df.columns}
            | {c for c in plan.columns_to_drop if c not in state.df.columns}
        )
        if unknown:
            return json.dumps(
                {"error": f"Unknown columns: {unknown}", "valid_columns": list(state.df.columns)}
            )
        if state.target in plan.columns_to_drop or state.target in {
            a.column for a in plan.actions
        }:
            return json.dumps(
                {"error": f"Refusing to modify or drop the target column '{state.target}'."}
            )

        before_shape = state.df.shape
        try:
            state.df = apply_cleaning(state.df, plan)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        after_shape = state.df.shape
        state.cleaning_applied = True
        state.record("cleaning", "plan_applied", {"before": before_shape, "after": after_shape})
        log_plan_and_feedback(state.mlflow_run_id, plan, stage="cleaning")
        log_metrics(
            state.mlflow_run_id,
            {
                "rows_dropped": before_shape[0] - after_shape[0],
                "cols_dropped": before_shape[1] - after_shape[1],
            },
        )
        return json.dumps(
            {"before_shape": before_shape, "after_shape": after_shape, "status": "applied"}
        )
