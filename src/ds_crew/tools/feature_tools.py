"""Deterministic feature engineering.

The train/test split itself happens earlier, in cleaning_tools.py (so that
cleaning statistics can also be fit on the training split only -- see that
module's docstring). This module consumes the already-split, already-cleaned
state.df_train/state.df_test and fits encoders/scalers on the training split
only: target-mean/frequency encoders must be *fit* on the training fold only,
or they leak target information into the engineered features. Model
selection and HPO only ever see X_train; X_test is *scored* exactly once, in
evaluation (see eval_tools.py's docstring for the precise form of that
invariant and why explain_tools.py's later read-only pass does not violate it).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ds_crew.tools.base import Tool
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ds_crew.schemas import ColumnFeaturePlan, FeatureEngineeringPlan, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_params, log_plan_and_feedback


class DatetimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Decomposes a single datetime-like column into year/month/day/dayofweek/is_weekend."""

    def fit(self, X, y=None):  # noqa: N803 (sklearn convention)
        self.n_features_in_ = 1
        return self

    def transform(self, X):  # noqa: N803
        s = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors="coerce")
        out = pd.DataFrame(
            {
                "year": s.dt.year,
                "month": s.dt.month,
                "day": s.dt.day,
                "dayofweek": s.dt.dayofweek,
                "is_weekend": (s.dt.dayofweek >= 5).astype("float"),
            }
        )
        return out.fillna(-1).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        base = input_features[0] if input_features is not None and len(input_features) else "dt"
        return np.array([f"{base}_{suffix}" for suffix in ("year", "month", "day", "dayofweek", "is_weekend")])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Maps each category to its relative frequency in the training data."""

    def fit(self, X, y=None):  # noqa: N803
        s = pd.DataFrame(X).iloc[:, 0]
        self.freq_map_ = s.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):  # noqa: N803
        s = pd.DataFrame(X).iloc[:, 0]
        return s.map(self.freq_map_).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        base = input_features[0] if input_features is not None and len(input_features) else "col"
        return np.array([f"{base}_freq"])


_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def _column_transformer_entry(cp: ColumnFeaturePlan, is_numeric: bool):
    steps: list[tuple[str, object]] = []
    if cp.datetime_decompose:
        steps.append(("dt", DatetimeFeaturesTransformer()))
    elif cp.encoding == "onehot":
        steps.append(("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif cp.encoding == "ordinal":
        steps.append(("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    elif cp.encoding == "target_mean":
        steps.append(("enc", TargetEncoder()))
    elif cp.encoding == "frequency":
        steps.append(("enc", FrequencyEncoder()))
    elif cp.encoding == "none":
        if not is_numeric:
            raise ValueError(
                f"Column '{cp.column}' is not numeric and encoding is 'none' -- choose an "
                "encoding strategy (onehot/ordinal/target_mean/frequency)."
            )

    if cp.scaling != "none" and not cp.datetime_decompose:
        steps.append((f"scale_{cp.scaling}", _SCALERS[cp.scaling]()))

    if not steps:
        return "passthrough"
    return Pipeline(steps)


def build_transformer(plan: FeatureEngineeringPlan, X: pd.DataFrame) -> ColumnTransformer:
    transformers = []
    for cp in plan.column_plans:
        is_numeric = pd.api.types.is_numeric_dtype(X[cp.column])
        entry = _column_transformer_entry(cp, is_numeric)
        transformers.append((cp.column, entry, [cp.column]))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    feature_names: list[str],
    plan: FeatureEngineeringPlan,
    task_type: TaskType,
):
    method = plan.feature_selection_method
    if method == "none":
        return X_train, X_test, feature_names

    if method == "variance_threshold":
        selector = VarianceThreshold()
    else:  # mutual_info
        k = min(plan.top_k or 10, X_train.shape[1])
        score_func = mutual_info_classif if task_type == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score_func, k=k)

    if method == "variance_threshold":
        X_train_sel = selector.fit_transform(X_train)
    else:
        X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    names_sel = [n for n, keep in zip(feature_names, selector.get_support()) if keep]
    return X_train_sel, X_test_sel, names_sel


class ApplyFeaturePlanTool(Tool):
    name: str = "apply_feature_plan"
    description: str = (
        "MUTATES the run: builds the feature matrix from the already-split, already-cleaned "
        "train/test data per a human-approved FeatureEngineeringPlan. Every non-target column "
        "must be explicitly covered by column_plans or features_to_drop. Refuses to treat the "
        "target as a feature. Irreversible for this run -- call EXACTLY ONCE with the complete "
        "human-approved plan; a second call is refused. Requires cleaning to have run first."
    )
    args_schema: type[BaseModel] = FeatureEngineeringPlan
    run_id: str = ""

    def _run(self, **plan_kwargs) -> str:
        plan_kwargs.setdefault("run_id", self.run_id)
        plan = FeatureEngineeringPlan(**plan_kwargs)
        state = get_data_store().get(self.run_id)

        if state.features_applied:
            return json.dumps(
                {
                    "error": "Feature engineering has already been applied for this run. Do "
                    "not call apply_feature_plan again -- proceed to the next task."
                }
            )

        if state.df_train is None or state.df_test is None:
            return json.dumps(
                {"error": "No train/test split found; run cleaning (apply_cleaning_plan) first."}
            )

        feature_cols = [c for c in state.df_train.columns if c != state.target]
        planned_cols = {cp.column for cp in plan.column_plans}

        if state.target in planned_cols:
            return json.dumps({"error": f"Refusing to treat target '{state.target}' as a feature."})

        unknown = sorted((planned_cols | set(plan.features_to_drop)) - set(feature_cols))
        if unknown:
            return json.dumps({"error": f"Unknown columns: {unknown}"})

        uncovered = sorted(set(feature_cols) - planned_cols - set(plan.features_to_drop))
        if uncovered:
            return json.dumps(
                {
                    "error": (
                        f"Columns not covered by column_plans or features_to_drop: {uncovered}. "
                        "Every feature column must be explicitly addressed."
                    )
                }
            )

        if state.task_type is None:
            return json.dumps({"error": "task_type is not set on the run; run EDA first."})

        X_train_raw = state.df_train.drop(columns=[state.target])
        y_train = state.df_train[state.target]
        X_test_raw = state.df_test.drop(columns=[state.target])
        y_test = state.df_test[state.target]

        try:
            transformer = build_transformer(plan, X_train_raw)
            X_train = transformer.fit_transform(X_train_raw, y_train)
            X_test = transformer.transform(X_test_raw)
            feature_names = list(transformer.get_feature_names_out())
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        X_train, X_test, feature_names = select_features(
            X_train, X_test, y_train, feature_names, plan, state.task_type
        )

        state.X_train, state.X_test = X_train, X_test
        state.y_train, state.y_test = y_train, y_test
        state.fitted_transformer = transformer
        state.feature_names = feature_names
        state.features_applied = True
        state.record("feature_engineering", "plan_applied", {"n_features": len(feature_names)})
        log_plan_and_feedback(state.mlflow_run_id, plan, stage="feature_engineering")
        log_params(state.mlflow_run_id, {"n_features_final": len(feature_names)})
        return json.dumps(
            {"n_features": len(feature_names), "feature_names": feature_names, "status": "applied"}
        )
