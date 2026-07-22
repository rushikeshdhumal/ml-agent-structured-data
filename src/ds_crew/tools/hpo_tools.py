"""Deterministic hyperparameter optimization via Optuna.

Search spaces are fixed, code-defined dictionaries per model -- the agent
picks *which* leaderboard model(s) to tune, never the search space itself.
Budget (n_trials/timeout_s) is capped in two layers: a Pydantic Field bound
on the tool input, and a hard `min()` clamp in the tool body, so the agent
cannot request more than the configured budget under any circumstance.

Trial and model failures are isolated, mirroring model_tools.py's
per-candidate leaderboard isolation: one bad hyperparameter draw (e.g. KNN's
n_neighbors exceeding a fold's sample count) fails that trial only, and one
model that exhausts its whole budget without a single successful trial is
skipped (recorded in HpoResults.warnings) rather than aborting every other
requested model in the same tune_model_hyperparameters call.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ds_crew import settings
from ds_crew.schemas import HpoResult, HpoResults, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_json_artifact, log_stage_metrics
from ds_crew.tools.model_tools import BASE_MODEL_KWARGS, CANDIDATE_MODELS, METRIC_BY_TASK

optuna.logging.set_verbosity(optuna.logging.WARNING)

HPO_CV_FOLDS = 3


def _suggest_params(trial: optuna.Trial, model_name: str, n_fold_train_samples: int) -> dict:
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
    if model_name == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "knn":
        # n_neighbors must be < the number of samples in each CV training fold, or
        # every trial that draws too high a value fails outright -- bound the search
        # space to what this dataset can actually support instead of wasting budget.
        max_k = max(2, min(30, n_fold_train_samples - 1))
        return {"n_neighbors": trial.suggest_int("n_neighbors", 2, max_k)}
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # subsample_freq must be >0 for LightGBM to actually apply subsample bagging.
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    raise ValueError(f"No search space defined for model '{model_name}'.")


def run_optuna_study(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_name: str,
    task_type: TaskType,
    n_trials: int,
    timeout_s: int,
    seed: int,
) -> HpoResult:
    cls = CANDIDATE_MODELS[task_type][model_name]
    metric = METRIC_BY_TASK[task_type]
    base_kwargs = BASE_MODEL_KWARGS.get(model_name, {})
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=seed)
    n_fold_train_samples = len(X_train) * (HPO_CV_FOLDS - 1) // HPO_CV_FOLDS

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name, n_fold_train_samples)
        model = cls(**{**base_kwargs, **params})
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    # catch=(Exception,) means one bad hyperparameter draw (a Optuna-suggested
    # combination that raises during fit/score) fails only that trial -- it's
    # recorded as a FAILed trial and the study moves on, instead of the whole
    # study aborting and losing every trial completed so far.
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"all {len(study.trials)} trials failed -- the search space may not be viable "
            "for this dataset size"
        )

    return HpoResult(
        model_name=model_name,
        best_params=study.best_params,
        best_score=round(float(study.best_value), 4),
        n_trials=len(study.trials),
    )


class TuneModelsInput(BaseModel):
    model_names: list[str] = Field(
        ..., description="Must be a subset of the top-K leaderboard candidates."
    )
    n_trials: int = Field(default=30, ge=1, le=settings.MAX_HPO_TRIALS)
    timeout_s: int = Field(default=300, ge=1, le=settings.MAX_HPO_TIMEOUT_S)


class TuneModelsTool(BaseTool):
    name: str = "tune_model_hyperparameters"
    description: str = (
        "Runs an Optuna study per requested model over a fixed, code-defined search space. "
        "model_names must be a subset of the top-K model-selection leaderboard candidates. "
        "n_trials/timeout_s are hard-capped server-side regardless of what is requested. "
        "Call only after model selection."
    )
    args_schema: type[BaseModel] = TuneModelsInput
    run_id: str = ""
    top_k: int = settings.DEFAULT_TOP_K_FOR_HPO

    def _run(self, model_names: list[str], n_trials: int = 30, timeout_s: int = 300) -> str:
        state = get_data_store().get(self.run_id)
        if state.leaderboard is None:
            return json.dumps({"error": "No leaderboard found; run model selection first."})

        allowed = {c.model_name for c in state.leaderboard.candidates[: self.top_k]}
        invalid = sorted(set(model_names) - allowed)
        if invalid:
            return json.dumps(
                {
                    "error": (
                        f"{invalid} not in top-{self.top_k} leaderboard candidates: "
                        f"{sorted(allowed)}"
                    )
                }
            )

        n_trials = min(n_trials, settings.MAX_HPO_TRIALS)
        timeout_s = min(timeout_s, settings.MAX_HPO_TIMEOUT_S)

        results: list[HpoResult] = []
        warnings: list[str] = []
        for name in model_names:
            try:
                result = run_optuna_study(
                    state.X_train,
                    state.y_train,
                    name,
                    state.task_type,
                    n_trials,
                    timeout_s,
                    seed=settings.RANDOM_SEED,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate one bad model, not the whole batch
                warnings.append(f"{name}: skipped after raising {type(exc).__name__}: {exc}")
                continue
            state.hpo_results[name] = result
            results.append(result)
            log_json_artifact(state.mlflow_run_id, result, f"hpo/{name}_best_params.json")

        if not results:
            return json.dumps({"error": f"All requested models failed HPO: {'; '.join(warnings)}"})

        state.record(
            "hpo",
            "studies_complete",
            {"models": [r.model_name for r in results], "n_trials": n_trials},
        )
        log_stage_metrics(
            state.mlflow_run_id,
            "hpo",
            {f"{r.model_name}_best_score": r.best_score for r in results},
        )
        return HpoResults(run_id=self.run_id, results=results, warnings=warnings).model_dump_json()
