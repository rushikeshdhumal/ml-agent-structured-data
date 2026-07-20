"""Deterministic held-out evaluation. X_test is touched exactly once, here."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ds_crew import settings
from ds_crew.schemas import EvaluationBundle, EvaluationReport, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_json_artifact, log_metrics, log_tags
from ds_crew.tools.model_tools import BASE_MODEL_KWARGS, CANDIDATE_MODELS


def evaluate_candidate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    params: dict,
    task_type: TaskType,
    feature_names: list[str] | None,
    near_perfect_threshold: float,
) -> tuple[EvaluationReport, object]:
    cls = CANDIDATE_MODELS[task_type][model_name]
    base_kwargs = BASE_MODEL_KWARGS.get(model_name, {})
    model = cls(**{**base_kwargs, **params})
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics: dict[str, float] = {}
    confusion = None
    if task_type == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["f1_macro"] = round(float(f1_score(y_test, preds, average="macro")), 4)
        metrics["precision_macro"] = round(
            float(precision_score(y_test, preds, average="macro", zero_division=0)), 4
        )
        metrics["recall_macro"] = round(
            float(recall_score(y_test, preds, average="macro", zero_division=0)), 4
        )
        confusion = confusion_matrix(y_test, preds).tolist()
        primary_metric = metrics["f1_macro"]
    else:
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        primary_metric = metrics["r2"]

    feature_importances = None
    if feature_names:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            coef = getattr(model, "coef_", None)
            if coef is not None:
                importances = np.abs(np.ravel(coef))
        if importances is not None and len(importances) == len(feature_names):
            feature_importances = {
                name: round(float(val), 4)
                for name, val in sorted(
                    zip(feature_names, importances), key=lambda pair: -abs(pair[1])
                )
            }

    leakage_suspicion = primary_metric >= near_perfect_threshold
    notes = ""
    if leakage_suspicion:
        notes = (
            f"Primary metric ({primary_metric}) is suspiciously close to perfect -- "
            "double-check for target leakage before accepting this model."
        )

    report = EvaluationReport(
        model_name=model_name,
        metrics=metrics,
        confusion_matrix=confusion,
        feature_importances=feature_importances,
        leakage_suspicion=leakage_suspicion,
        notes=notes,
    )
    return report, model


class EvaluateModelsInput(BaseModel):
    model_names: list[str]


class EvaluateModelsTool(BaseTool):
    name: str = "evaluate_models"
    description: str = (
        "Fits each named model (using tuned hyperparameters from HPO if available, else "
        "defaults) on X_train and scores it ONCE on the held-out X_test. Flags "
        "leakage_suspicion when a metric looks too good to be true. model_names must be "
        "leaderboard candidates."
    )
    args_schema: type[BaseModel] = EvaluateModelsInput
    run_id: str = ""

    def _run(self, model_names: list[str]) -> str:
        state = get_data_store().get(self.run_id)
        if state.X_test is None or state.leaderboard is None:
            return json.dumps(
                {
                    "error": "No test split / leaderboard found; run feature engineering and "
                    "model selection first."
                }
            )

        allowed = {c.model_name for c in state.leaderboard.candidates}
        invalid = sorted(set(model_names) - allowed)
        if invalid:
            return json.dumps(
                {"error": f"{invalid} not in leaderboard candidates: {sorted(allowed)}"}
            )

        reports = []
        for name in model_names:
            params = state.hpo_results[name].best_params if name in state.hpo_results else {}
            report, fitted = evaluate_candidate(
                state.X_train,
                state.X_test,
                state.y_train,
                state.y_test,
                name,
                params,
                state.task_type,
                state.feature_names,
                settings.NEAR_PERFECT_THRESHOLD,
            )
            state.fitted_models[name] = fitted
            state.evaluation_reports[name] = report
            reports.append(report)
            log_json_artifact(state.mlflow_run_id, report, f"evaluation/{name}_report.json")
            log_metrics(
                state.mlflow_run_id, {f"{name}_{k}": v for k, v in report.metrics.items()}
            )
            if report.leakage_suspicion:
                log_tags(state.mlflow_run_id, {f"{name}_leakage_suspicion": "true"})

        state.record("evaluation", "scored", {"models": model_names})
        return EvaluationBundle(run_id=self.run_id, reports=reports).model_dump_json()
