"""Deterministic data-cleaning execution. Mutates the run's dataset.

Only ever called by the "execute" half of the cleaning HITL checkpoint, with
a plan a human has already approved (see config/tasks.yaml). The target
column can never be touched by a CleaningPlan -- refused both here and by
the propose-stage guardrail (defense in depth).

This is also where the train/test split happens (not in feature engineering)
so that every cleaning statistic -- imputation values, outlier bounds, KNN
imputer neighbors -- can be fit on the training split only and applied
identically to the test split. Splitting later, after cleaning has already
computed those statistics over the full dataset, would leak test-set values
into decisions that affect the training data.
"""

from __future__ import annotations

import json

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from ds_crew import settings
from ds_crew.schemas import CleaningPlan, ColumnCleaningAction, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_metrics, log_plan_and_feedback

_DTYPE_CAST_MAP = {
    "int": "int64",
    "float": "float64",
    "str": "str",
    "category": "category",
    "bool": "bool",
}


def split_df(
    df: pd.DataFrame, target: str, task_type: TaskType, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level split of the whole dataframe (target column stays in both
    halves -- feature engineering drops it later when building X). Stratified
    on the target for classification so class proportions are preserved.
    """
    stratify = df[target] if task_type == "classification" else None
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )


def apply_structural_cleaning(df: pd.DataFrame, plan: CleaningPlan) -> pd.DataFrame:
    """Row/column-identity operations applied to the FULL dataset, before the
    split: dropping exact-duplicate rows and dropping columns outright.
    Neither depends on a fitted statistic, but duplicate-row dropping
    specifically MUST happen pre-split -- otherwise an identical row could
    land in both train and test (the model would effectively be evaluated on
    a row it was trained on).
    """
    df = df.copy()
    if plan.drop_duplicate_rows:
        df = df.drop_duplicates()
    if plan.columns_to_drop:
        df = df.drop(columns=[c for c in plan.columns_to_drop if c in df.columns])
    return df


def _require_numeric(df: pd.DataFrame, col: str, strategy: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"missing_strategy '{strategy}' requires a numeric column; '{col}' is not.")


def _fit_transform_missing(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, action: ColumnCleaningAction
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = action.missing_strategy
    if strategy == "drop_rows":
        return df_train.dropna(subset=[col]), df_test.dropna(subset=[col])
    if strategy == "ffill":
        # No fittable scalar statistic here (each fill borrows a neighboring
        # row's value) -- fill train and test independently from their own
        # rows only, so a test row's value can never fill a train row's gap
        # (or vice versa).
        df_train[col] = df_train[col].ffill()
        df_test[col] = df_test[col].ffill()
        return df_train, df_test
    if strategy == "bfill":
        df_train[col] = df_train[col].bfill()
        df_test[col] = df_test[col].bfill()
        return df_train, df_test
    if strategy == "knn":
        return _knn_impute_fit_transform(df_train, df_test, col)

    value = None
    if strategy == "mean":
        _require_numeric(df_train, col, "mean")
        value = df_train[col].mean()
    elif strategy == "median":
        _require_numeric(df_train, col, "median")
        value = df_train[col].median()
    elif strategy == "mode":
        mode = df_train[col].mode(dropna=True)
        value = mode.iloc[0] if not mode.empty else None
    elif strategy == "constant":
        if action.constant_fill_value is None:
            raise ValueError(f"missing_strategy 'constant' for '{col}' requires constant_fill_value.")
        value = action.constant_fill_value

    if value is not None:
        df_train[col] = df_train[col].fillna(value)
        df_test[col] = df_test[col].fillna(value)
    return df_train, df_test


def _knn_impute_fit_transform(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, n_neighbors: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
    if col not in numeric_cols:
        raise ValueError(f"missing_strategy 'knn' requires a numeric column; '{col}' is not.")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_train[numeric_cols] = imputer.fit_transform(df_train[numeric_cols])
    test_numeric_cols = [c for c in numeric_cols if c in df_test.columns]
    df_test[test_numeric_cols] = imputer.transform(df_test[test_numeric_cols])
    return df_train, df_test


def _iqr_bounds(s: pd.Series) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _fit_transform_outliers(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, strategy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not pd.api.types.is_numeric_dtype(df_train[col]):
        raise ValueError(f"outlier_strategy '{strategy}' requires a numeric column; '{col}' is not.")

    if strategy == "iqr_clip":
        lower, upper = _iqr_bounds(df_train[col])
        df_train[col] = df_train[col].clip(lower, upper)
        df_test[col] = df_test[col].clip(lower, upper)
    elif strategy == "zscore_clip":
        mean, std = df_train[col].mean(), df_train[col].std()
        if std and std > 0:
            lower, upper = mean - 3 * std, mean + 3 * std
            df_train[col] = df_train[col].clip(lower, upper)
            df_test[col] = df_test[col].clip(lower, upper)
    elif strategy == "drop":
        lower, upper = _iqr_bounds(df_train[col])
        df_train = df_train[(df_train[col] >= lower) & (df_train[col] <= upper)]
        df_test = df_test[(df_test[col] >= lower) & (df_test[col] <= upper)]
    return df_train, df_test


def _apply_dtype_cast(df: pd.DataFrame, col: str, cast: str) -> pd.DataFrame:
    if cast == "datetime":
        df[col] = pd.to_datetime(df[col])
    else:
        df[col] = df[col].astype(_DTYPE_CAST_MAP[cast])
    return df


def fit_transform_cleaning(
    df_train: pd.DataFrame, df_test: pd.DataFrame, plan: CleaningPlan
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits every stateful cleaning statistic (imputation values, outlier
    bounds, KNN-imputer neighbors) on df_train only, then applies the
    identical fitted transform to both df_train and df_test. df_test's own
    values never influence what happens to df_train, or to itself.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    for action in plan.actions:
        col = action.column
        if action.missing_strategy:
            df_train, df_test = _fit_transform_missing(df_train, df_test, col, action)
        if action.outlier_strategy != "none":
            df_train, df_test = _fit_transform_outliers(df_train, df_test, col, action.outlier_strategy)
        if action.dtype_cast:
            df_train = _apply_dtype_cast(df_train, col, action.dtype_cast)
            df_test = _apply_dtype_cast(df_test, col, action.dtype_cast)
    return df_train, df_test


class ApplyCleaningPlanTool(BaseTool):
    name: str = "apply_cleaning_plan"
    description: str = (
        "MUTATES the run's dataset by applying a human-approved CleaningPlan. Validates "
        "every column and strategy against the current dataset first and refuses to touch "
        "the target column. Also performs the train/test split (every cleaning statistic is "
        "fit on the training split only). Irreversible for this run -- call EXACTLY ONCE "
        "with the complete human-approved plan; a second call is refused."
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
        if state.task_type is None:
            return json.dumps({"error": "task_type is not set on the run; run EDA first."})

        before_shape = state.df.shape
        try:
            structural = apply_structural_cleaning(state.df, plan)
            df_train, df_test = split_df(
                structural,
                state.target,
                state.task_type,
                state.test_size,
                random_state=settings.RANDOM_SEED,
            )
            df_train, df_test = fit_transform_cleaning(df_train, df_test, plan)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- e.g. a class too rare to stratify-split
            return json.dumps({"error": f"Could not split/clean the dataset: {exc}"})

        after_shape = (df_train.shape[0] + df_test.shape[0], df_train.shape[1])
        state.df_train = df_train
        state.df_test = df_test
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
            {
                "before_shape": before_shape,
                "after_shape": after_shape,
                "train_rows": df_train.shape[0],
                "test_rows": df_test.shape[0],
                "status": "applied",
            }
        )
