"""Deterministic model selection. Fixed, code-defined candidate registry --
the agent never chooses which model classes exist, only which leaderboard
entries to send on to hyperparameter optimization. Cross-validation runs on
X_train only; X_test is never touched here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from crewai.tools import BaseTool
from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import BaseModel, Field
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from ds_crew import settings
from ds_crew.schemas import Leaderboard, MetricChoice, ModelCandidateResult, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_params, log_stage_metrics


class _XGBClassifierWithLabelEncoding(XGBClassifier):
    """Unlike every other classifier in this registry, XGBoost's sklearn API
    requires y to already be integer-encoded 0..n_classes-1 and rejects
    arbitrary class labels (e.g. "yes"/"no") outright. Encode/decode here so
    it's a drop-in fit/predict match for the rest of the registry instead of
    needing special-casing in run_cross_validation/eval_tools/hpo_tools.
    """

    def fit(self, X, y, **kwargs):
        self._label_encoder = LabelEncoder().fit(y)
        return super().fit(X, self._label_encoder.transform(y), **kwargs)

    def predict(self, X, **kwargs):
        return self._label_encoder.inverse_transform(super().predict(X, **kwargs))


# Candidate models are chosen for a strong empirical track record on
# structured/tabular data, not just breadth: gradient boosting (xgboost/
# lightgbm/catboost) consistently matches or beats bagged-tree and kernel/
# neural alternatives on tabular benchmarks, so weaker or redundant model
# families (random forest, sklearn's own hist_gradient_boosting, SVM, MLP,
# naive Bayes) are deliberately left out rather than padding the leaderboard.
CANDIDATE_MODELS: dict[TaskType, dict[str, type]] = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "xgboost": _XGBClassifierWithLabelEncoding,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
    },
    "regression": {
        "ridge": Ridge,
        "elastic_net": ElasticNet,
        "knn": KNeighborsRegressor,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
        "catboost": CatBoostRegressor,
    },
}

METRIC_BY_TASK: dict[TaskType, str] = {"classification": "f1_macro", "regression": "r2"}

# The metrics a human may choose (via SetMetricTool) to drive cross-validation,
# HPO, ensembling, and evaluation instead of the METRIC_BY_TASK default.
# Deliberately restricted to bounded, higher-is-better metrics: HPO's Optuna
# studies hardcode direction="maximize", and the near-perfect-leakage heuristic
# in eval_tools.py checks `primary_metric >= NEAR_PERFECT_THRESHOLD` -- an
# error metric like MAE/RMSE (lower-is-better, unbounded) would silently break
# both without extra special-casing, so regression intentionally offers only r2.
ALLOWED_METRICS: dict[TaskType, set[str]] = {
    "classification": {
        "f1_macro",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "balanced_accuracy",
        "roc_auc",
    },
    "regression": {"r2"},
}

BASE_MODEL_KWARGS: dict[str, dict] = {
    "logistic_regression": {"max_iter": 1000, "random_state": settings.RANDOM_SEED},
    "knn": {"n_jobs": -1},
    "ridge": {"random_state": settings.RANDOM_SEED},
    "elastic_net": {"random_state": settings.RANDOM_SEED},
    "xgboost": {"random_state": settings.RANDOM_SEED, "n_jobs": -1, "verbosity": 0},
    "lightgbm": {"random_state": settings.RANDOM_SEED, "n_jobs": -1, "verbosity": -1},
    "catboost": {
        "random_state": settings.RANDOM_SEED,
        "verbose": False,
        "thread_count": -1,
        "iterations": 300,
    },
}


def make_cv_splitter(task_type: TaskType, n_splits: int, seed: int):
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def resolve_cv_scorer(metric: str, y: pd.Series) -> str:
    """Most allowed metric names are already valid sklearn `scoring=` strings.
    roc_auc is the one exception: sklearn requires "roc_auc_ovr" (not
    "roc_auc") once there are more than two classes.
    """
    if metric == "roc_auc" and len(set(y)) > 2:
        return "roc_auc_ovr"
    return metric


def run_cross_validation(
    X_train: np.ndarray,
    y_train: pd.Series,
    task_type: TaskType,
    cv_folds: int = 5,
    metric: str | None = None,
) -> Leaderboard:
    candidates = CANDIDATE_MODELS[task_type]
    metric = metric or METRIC_BY_TASK[task_type]
    scorer = resolve_cv_scorer(metric, y_train)
    cv = make_cv_splitter(task_type, cv_folds, settings.RANDOM_SEED)

    results: list[ModelCandidateResult] = []
    warnings: list[str] = []
    for name, cls in candidates.items():
        try:
            model = cls(**BASE_MODEL_KWARGS.get(name, {}))
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scorer)
        except Exception as exc:  # noqa: BLE001 -- isolate one bad candidate, not the whole leaderboard
            warnings.append(f"{name}: skipped after raising {type(exc).__name__}: {exc}")
            continue
        results.append(
            ModelCandidateResult(
                model_name=name,
                cv_mean_score=round(float(np.mean(scores["test_score"])), 4),
                cv_std=round(float(np.std(scores["test_score"])), 4),
                fit_time_s=round(float(np.mean(scores["fit_time"])), 4),
            )
        )
    if not results:
        raise RuntimeError(f"All candidate models failed: {'; '.join(warnings)}")
    results.sort(key=lambda r: r.cv_mean_score, reverse=True)
    return Leaderboard(
        run_id="", task_type=task_type, metric_name=metric, candidates=results, warnings=warnings
    )


class TrainCandidatesInput(BaseModel):
    cv_folds: int = Field(default=5, ge=2, le=20)


class TrainCandidateModelsTool(BaseTool):
    name: str = "train_candidate_models"
    description: str = (
        "Read-only w.r.t. the dataset (only reads X_train/y_train). Cross-validates a fixed, "
        "code-defined set of candidate models appropriate for the run's task type and returns "
        "a leaderboard sorted best-first. Call only after feature engineering."
    )
    args_schema: type[BaseModel] = TrainCandidatesInput
    run_id: str = ""

    def _run(self, cv_folds: int = 5) -> str:
        state = get_data_store().get(self.run_id)
        if state.X_train is None or state.y_train is None:
            return json.dumps({"error": "No feature matrix found; run feature engineering first."})
        try:
            leaderboard = run_cross_validation(
                state.X_train,
                state.y_train,
                state.task_type,
                cv_folds,
                metric=state.metric_name,
            )
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})
        leaderboard.run_id = self.run_id
        state.leaderboard = leaderboard
        state.record(
            "model_selection", "cv_complete", {"n_candidates": len(leaderboard.candidates)}
        )
        log_params(state.mlflow_run_id, {"cv_folds": cv_folds})
        log_stage_metrics(
            state.mlflow_run_id,
            "model_selection",
            {c.model_name: c.cv_mean_score for c in leaderboard.candidates},
        )
        return leaderboard.model_dump_json()


class SetMetricTool(BaseTool):
    name: str = "set_evaluation_metric"
    description: str = (
        "Records the human-approved optimization metric for this run. This metric drives "
        "cross-validation, hyperparameter tuning, ensembling, and evaluation for every "
        "later stage. Must be one of the metrics allowed for the run's task type. Call "
        "only after a human has reviewed the proposed metric."
    )
    args_schema: type[BaseModel] = MetricChoice
    run_id: str = ""

    def _run(self, **kwargs) -> str:
        kwargs.setdefault("run_id", self.run_id)
        choice = MetricChoice(**kwargs)
        state = get_data_store().get(self.run_id)

        allowed = ALLOWED_METRICS[state.task_type]
        if choice.metric not in allowed:
            return json.dumps(
                {
                    "error": (
                        f"'{choice.metric}' is not allowed for task type "
                        f"'{state.task_type}': {sorted(allowed)}"
                    )
                }
            )

        state.metric_name = choice.metric
        state.record("model_selection", "metric_set", {"metric": choice.metric})
        log_params(state.mlflow_run_id, {"evaluation_metric": choice.metric})
        return json.dumps({"status": "set", "metric": choice.metric})
