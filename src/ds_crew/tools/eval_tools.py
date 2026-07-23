"""Deterministic held-out evaluation. X_test is touched exactly once, here."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ds_crew import settings
from ds_crew.schemas import EvaluationBundle, EvaluationReport, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_json_artifact, log_metrics, log_tags
from ds_crew.tools.model_tools import BASE_MODEL_KWARGS, CANDIDATE_MODELS, METRIC_BY_TASK


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
    metric: str | None = None,
    prefitted_model: object | None = None,
) -> tuple[EvaluationReport, object]:
    metric = metric or METRIC_BY_TASK[task_type]
    if prefitted_model is not None:
        # An ensemble (or any other pre-built estimator) fitted on this same
        # X_train elsewhere -- e.g. EnsembleModelsTool -- is scored here
        # without a second fit, so it still only ever touches X_test through
        # this one function, preserving "X_test touched exactly once".
        model = prefitted_model
    else:
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
        metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_test, preds)), 4)
        confusion = confusion_matrix(y_test, preds).tolist()

        # roc_auc needs predict_proba; not every estimator (or every fold's
        # label distribution) supports it, so this is best-effort -- absent
        # from the bundle rather than a hard failure if it can't be computed.
        # Column order is derived from y_train (sorted(unique(y_train))), NOT
        # model.classes_: _XGBClassifierWithLabelEncoding's classes_ reflects
        # the internal integer encoding XGBoost's sklearn API requires at fit
        # time (see model_tools.py), not the original label space -- but its
        # predict_proba columns still follow sorted(unique(y_train)) order,
        # same as every other classifier here, since that's the exact
        # encoding its internal LabelEncoder applied.
        predict_proba = getattr(model, "predict_proba", None)
        if predict_proba is not None:
            try:
                proba = predict_proba(X_test)
                classes_ = np.unique(y_train)
                if proba.shape[1] == 2:
                    y_true_binary = (np.asarray(y_test) == classes_[1]).astype(int)
                    metrics["roc_auc"] = round(
                        float(roc_auc_score(y_true_binary, proba[:, 1])), 4
                    )
                else:
                    metrics["roc_auc"] = round(
                        float(roc_auc_score(y_test, proba, multi_class="ovr", labels=classes_)),
                        4,
                    )
            except ValueError:
                pass

        primary_metric = metrics.get(metric, metrics["f1_macro"])
    else:
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        primary_metric = metrics.get(metric, metrics["r2"])

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
        "leaderboard candidates. Call EXACTLY ONCE for this run -- the held-out test set is "
        "only meaningful if it's scored a single time; a second call is refused."
    )
    args_schema: type[BaseModel] = EvaluateModelsInput
    run_id: str = ""

    def _run(self, model_names: list[str]) -> str:
        state = get_data_store().get(self.run_id)
        if state.evaluation_applied:
            return json.dumps(
                {
                    "error": "evaluate_models has already been called for this run. Do not "
                    "call it again -- proceed to the next task with the existing results."
                }
            )
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

        # Set before touching X_test, not after the loop completes: if
        # evaluate_candidate raises partway through model_names, X_test has
        # already been used for whichever models ran first. Leaving the flag
        # unset until a clean finish would let a retry after a partial
        # failure touch X_test again, breaking "touched exactly once" even
        # though the original call never fully succeeded.
        state.evaluation_applied = True

        # state.leaderboard.metric_name (not state.metric_name, which may
        # still be None if the metric-selection gate was never invoked) is
        # the ground truth for what metric these candidates were actually
        # cross-validated on -- run_cross_validation always resolves and
        # stamps a concrete metric string, even when called with metric=None.
        metric = state.leaderboard.metric_name
        reports = []
        for name in model_names:
            if name in CANDIDATE_MODELS[state.task_type]:
                params = state.hpo_results[name].best_params if name in state.hpo_results else {}
                prefitted = None
            else:
                # Not a registry model name (e.g. "ensemble" from
                # EnsembleModelsTool) -- it must already be fit on X_train
                # and stored in state.fitted_models; score it as-is instead
                # of trying to construct it from CANDIDATE_MODELS.
                params = {}
                prefitted = state.fitted_models[name]
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
                metric=metric,
                prefitted_model=prefitted,
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
