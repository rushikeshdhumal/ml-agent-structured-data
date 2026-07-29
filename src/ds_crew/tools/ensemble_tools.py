"""Deterministic ensembling. Builds a metric-optimized ensemble (soft voting,
weighted voting, greedy Caruana selection, or stacking) from the strongest
model-selection/HPO leaderboard candidates -- the agent never hand-picks
members, weights, or a strategy; this module scores each candidate strategy
by cross-validating it against the run's chosen optimization metric
(state.leaderboard.metric_name) and keeps whichever wins, mirroring
model_tools.py's "agent proposes, deterministic code decides" philosophy.

Out-of-fold predictions for each member model are computed exactly once via
cross_val_predict and cached; every strategy then scores against those cached
arrays with plain NumPy, so weight/greedy search is cheap instead of repeated
model fitting. Ensemble size is capped at settings.MAX_ENSEMBLE_MEMBERS.

Runs on X_train only -- X_test is never touched here. The winning ensemble is
fit once on the full X_train and registered as an extra leaderboard candidate
named "ensemble", so it is scored by the existing single-pass evaluate_models
tool alongside the tuned single models, preserving "X_test scored exactly
once" (see eval_tools.py's docstring for the precise form of that invariant).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import optuna
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_validate

from ds_crew import settings
from ds_crew.schemas import EnsembleReport, ModelCandidateResult, TaskType
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_json_artifact, log_params, log_stage_metrics
from ds_crew.tools.model_tools import (
    BASE_MODEL_KWARGS,
    CANDIDATE_MODELS,
    make_cv_splitter,
    resolve_cv_scorer,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# A minimum-improvement epsilon looser than settings.MIN_ENSEMBLE_IMPROVEMENT --
# this one only decides whether adding a member to a greedy pick is worth the
# extra complexity, not whether the final ensemble beats the best single model.
_GREEDY_IMPROVEMENT_EPS = 1e-9


def _build_estimator(name: str, task_type: TaskType, params: dict) -> Any:
    cls = CANDIDATE_MODELS[task_type][name]
    base_kwargs = BASE_MODEL_KWARGS.get(name, {})
    return cls(**{**base_kwargs, **params})


def _oof_predictions(
    estimator: Any, X_train: np.ndarray, y_train, task_type: TaskType, cv
) -> np.ndarray:
    """One member's out-of-fold predictions: a (n_samples, n_classes) proba
    matrix for classification (columns aligned to sorted(np.unique(y_train)),
    which sklearn's cross_val_predict guarantees internally), or a
    (n_samples,) array of predicted values for regression.
    """
    if task_type == "classification":
        return cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")
    return cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict")


def _score_classification(y_true, pred, proba, metric: str, classes_: np.ndarray) -> float:
    if metric == "roc_auc":
        # roc_auc_score takes no pos_label kwarg -- binarize y_true against
        # the sorted-last class (classes_[1]) to match column 1's meaning,
        # the same "sorted, last = positive" convention proba columns follow
        # throughout this module (see _oof_predictions/model_tools.py).
        if len(classes_) == 2:
            y_true_binary = (np.asarray(y_true) == classes_[1]).astype(int)
            return float(roc_auc_score(y_true_binary, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes_))
    if metric == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, pred, average="macro"))
    if metric == "precision_macro":
        return float(precision_score(y_true, pred, average="macro", zero_division=0))
    if metric == "recall_macro":
        return float(recall_score(y_true, pred, average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    raise ValueError(f"Unsupported classification metric '{metric}'.")


def _combine(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() or 1.0)
    return np.tensordot(w, stacked, axes=(0, 0))


def _score_weights(
    member_arrays: list[np.ndarray],
    weights: list[float],
    y_train,
    task_type: TaskType,
    metric: str,
    classes_: np.ndarray | None,
) -> float:
    combined = _combine(member_arrays, weights)
    if task_type == "classification":
        pred = classes_[np.argmax(combined, axis=1)]
        return _score_classification(y_train, pred, combined, metric, classes_)
    return float(r2_score(y_train, combined))


def _optimize_weights(
    member_arrays: list[np.ndarray],
    y_train,
    task_type: TaskType,
    metric: str,
    classes_: np.ndarray | None,
    n_trials: int,
    seed: int,
) -> tuple[list[float], float]:
    n = len(member_arrays)

    def objective(trial: optuna.Trial) -> float:
        raw = [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n)]
        if sum(raw) <= 0:
            return -1e9
        return _score_weights(member_arrays, raw, y_train, task_type, metric, classes_)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        weights = [1.0 / n] * n
        return weights, _score_weights(member_arrays, weights, y_train, task_type, metric, classes_)

    raw = [study.best_params[f"w{i}"] for i in range(n)]
    total = sum(raw) or 1.0
    weights = [w / total for w in raw]
    return weights, float(study.best_value)


def _greedy_select(
    member_arrays: list[np.ndarray],
    y_train,
    task_type: TaskType,
    metric: str,
    classes_: np.ndarray | None,
    max_distinct: int,
) -> tuple[list[float], float]:
    """Caruana (2004) forward selection with replacement: start from the best
    single member, then repeatedly add whichever member (possibly a repeat)
    most improves the OOF metric, stopping on no improvement or once
    max_distinct distinct members have been picked. The pick multiset
    directly yields normalized ensemble weights.
    """
    n = len(member_arrays)
    singleton_scores = [
        _score_weights(
            member_arrays,
            [1.0 if i == j else 0.0 for j in range(n)],
            y_train,
            task_type,
            metric,
            classes_,
        )
        for i in range(n)
    ]
    best_idx = int(np.argmax(singleton_scores))
    picks = [best_idx]
    distinct = {best_idx}
    best_score = singleton_scores[best_idx]

    max_iterations = 20 * max(max_distinct, 1)
    for _ in range(max_iterations):
        candidates = []
        for i in range(n):
            if len(distinct | {i}) > max_distinct:
                continue
            trial_picks = picks + [i]
            weights = (np.bincount(trial_picks, minlength=n) / len(trial_picks)).tolist()
            score = _score_weights(member_arrays, weights, y_train, task_type, metric, classes_)
            candidates.append((score, i))
        if not candidates:
            break
        trial_best_score, trial_best_i = max(candidates, key=lambda c: c[0])
        if trial_best_score <= best_score + _GREEDY_IMPROVEMENT_EPS:
            break
        picks.append(trial_best_i)
        distinct.add(trial_best_i)
        best_score = trial_best_score

    weights = (np.bincount(picks, minlength=n) / len(picks)).tolist()
    return weights, best_score


def _stacking_oof_score(
    member_arrays: list[np.ndarray],
    y_train,
    task_type: TaskType,
    metric: str,
    classes_: np.ndarray | None,
    cv,
    seed: int,
) -> float:
    """Stacking's definition IS a meta-learner trained on out-of-fold base
    predictions -- reusing the already-cached member OOF arrays here scores
    it without refitting any base estimator, unlike a literal
    StackingClassifier/Regressor CV pass would.
    """
    if task_type == "classification":
        meta_X = np.concatenate(member_arrays, axis=1)
        final_estimator = LogisticRegression(max_iter=1000, random_state=seed)
        meta_proba = cross_val_predict(final_estimator, meta_X, y_train, cv=cv, method="predict_proba")
        meta_pred = classes_[np.argmax(meta_proba, axis=1)]
        return _score_classification(y_train, meta_pred, meta_proba, metric, classes_)
    meta_X = np.stack(member_arrays, axis=1)
    final_estimator = Ridge(random_state=seed)
    meta_pred = cross_val_predict(final_estimator, meta_X, y_train, cv=cv)
    return float(r2_score(y_train, meta_pred))


def _build_stacking(named_estimators: list[tuple[str, Any]], task_type: TaskType, cv, seed: int):
    if task_type == "classification":
        final_estimator = LogisticRegression(max_iter=1000, random_state=seed)
        return StackingClassifier(
            estimators=named_estimators, final_estimator=final_estimator, cv=cv, n_jobs=-1
        )
    final_estimator = Ridge(random_state=seed)
    return StackingRegressor(
        estimators=named_estimators, final_estimator=final_estimator, cv=cv, n_jobs=-1
    )


class BuildEnsembleInput(BaseModel):
    max_members: int = Field(
        default=settings.MAX_ENSEMBLE_MEMBERS,
        ge=2,
        le=settings.MAX_ENSEMBLE_MEMBERS,
        description="Distinct member models to consider, hard-capped server-side.",
    )


class EnsembleModelsTool(BaseTool):
    name: str = "build_ensemble"
    description: str = (
        "Combines the strongest model-selection/HPO candidates into a single ensemble "
        "(soft voting, weighted voting, greedy Caruana selection, or stacking), choosing "
        "whichever cross-validates best on the run's chosen optimization metric -- never "
        "raw accuracy. Reads only X_train/y_train; X_test is never touched. Registers the "
        "result as an extra leaderboard candidate named 'ensemble' so evaluate_models can "
        "score it later. Call only after hyperparameter tuning, exactly once."
    )
    args_schema: type[BaseModel] = BuildEnsembleInput
    run_id: str = ""

    def _run(self, max_members: int = settings.MAX_ENSEMBLE_MEMBERS) -> str:
        state = get_data_store().get(self.run_id)
        if state.ensemble_applied:
            return json.dumps({"error": "build_ensemble has already been called for this run."})
        if state.leaderboard is None or state.X_train is None:
            return json.dumps({"error": "No leaderboard found; run model selection first."})

        max_members = min(max_members, settings.MAX_ENSEMBLE_MEMBERS)
        task_type = state.task_type
        # The leaderboard's own metric_name is the ground truth for what its
        # cv_mean_score values actually measure -- always a concrete resolved
        # string, unlike state.metric_name which may still be None if the
        # metric-selection gate was skipped.
        metric = state.leaderboard.metric_name
        best_single = state.leaderboard.candidates[0]
        pool_names = [c.model_name for c in state.leaderboard.candidates[:max_members]]

        X_train, y_train = state.X_train, state.y_train
        cv = make_cv_splitter(task_type, settings.DEFAULT_CV_FOLDS, settings.RANDOM_SEED)
        classes_ = np.unique(y_train) if task_type == "classification" else None

        named_estimators: list[tuple[str, Any]] = []
        member_arrays: list[np.ndarray] = []
        warnings: list[str] = []
        for name in pool_names:
            params = state.hpo_results[name].best_params if name in state.hpo_results else {}
            try:
                oof = _oof_predictions(
                    _build_estimator(name, task_type, params), X_train, y_train, task_type, cv
                )
            except Exception as exc:  # noqa: BLE001 -- isolate one bad member, not the whole ensemble
                warnings.append(f"{name}: skipped after raising {type(exc).__name__}: {exc}")
                continue
            named_estimators.append((name, _build_estimator(name, task_type, params)))
            member_arrays.append(oof)

        if len(named_estimators) < 2:
            return json.dumps(
                {
                    "warning": "Fewer than 2 viable member models after out-of-fold scoring; "
                    "skipping ensembling.",
                    "warnings": warnings,
                    "skipped": True,
                }
            )

        seed = settings.RANDOM_SEED
        n = len(member_arrays)
        strategies: dict[str, tuple[list[float] | None, float]] = {}

        eq_weights = [1.0 / n] * n
        strategies["equal_voting"] = (
            eq_weights,
            _score_weights(member_arrays, eq_weights, y_train, task_type, metric, classes_),
        )
        strategies["weighted_voting"] = _optimize_weights(
            member_arrays, y_train, task_type, metric, classes_, settings.ENSEMBLE_WEIGHT_TRIALS, seed
        )
        strategies["greedy"] = _greedy_select(
            member_arrays, y_train, task_type, metric, classes_, max_members
        )
        try:
            strategies["stacking"] = (
                None,
                _stacking_oof_score(member_arrays, y_train, task_type, metric, classes_, cv, seed),
            )
        except Exception as exc:  # noqa: BLE001 -- stacking is one of several strategies, not required
            warnings.append(f"stacking: skipped after raising {type(exc).__name__}: {exc}")

        best_strategy = max(strategies, key=lambda k: strategies[k][1])
        weights, _ = strategies[best_strategy]

        member_names = [name for name, _ in named_estimators]
        final_estimator_name: str | None = None
        if best_strategy == "stacking":
            final_estimator = _build_stacking(named_estimators, task_type, cv, seed)
            final_estimator_name = "logistic_regression" if task_type == "classification" else "ridge"
        else:
            keep = [i for i, w in enumerate(weights) if w > 1e-6] or list(range(n))
            kept_estimators = [named_estimators[i] for i in keep]
            kept_weights = [weights[i] for i in keep]
            total = sum(kept_weights) or 1.0
            kept_weights = [w / total for w in kept_weights]
            member_names = [name for name, _ in kept_estimators]
            weights = kept_weights
            if task_type == "classification":
                final_estimator = VotingClassifier(
                    estimators=kept_estimators, voting="soft", weights=kept_weights, n_jobs=-1
                )
            else:
                final_estimator = VotingRegressor(
                    estimators=kept_estimators, weights=kept_weights, n_jobs=-1
                )

        scorer = resolve_cv_scorer(metric, y_train)
        scores = cross_validate(final_estimator, X_train, y_train, cv=cv, scoring=scorer)
        cv_mean = round(float(np.mean(scores["test_score"])), 4)
        cv_std = round(float(np.std(scores["test_score"])), 4)
        fit_time = round(float(np.mean(scores["fit_time"])), 4)

        final_estimator.fit(X_train, y_train)
        state.fitted_models["ensemble"] = final_estimator

        state.leaderboard.candidates.append(
            ModelCandidateResult(
                model_name="ensemble", cv_mean_score=cv_mean, cv_std=cv_std, fit_time_s=fit_time
            )
        )
        state.leaderboard.candidates.sort(key=lambda c: c.cv_mean_score, reverse=True)

        improved = cv_mean >= best_single.cv_mean_score + settings.MIN_ENSEMBLE_IMPROVEMENT
        report = EnsembleReport(
            run_id=self.run_id,
            strategy=best_strategy,
            member_models=member_names,
            weights=weights if best_strategy != "stacking" else None,
            final_estimator=final_estimator_name,
            metric_name=metric,
            cv_mean_score=cv_mean,
            best_single_model=best_single.model_name,
            best_single_cv_score=best_single.cv_mean_score,
            improved_over_best_single=improved,
            warnings=warnings,
            notes=(
                f"Ensemble ({best_strategy}) scored {cv_mean} vs best single "
                f"'{best_single.model_name}' at {best_single.cv_mean_score} on {metric}."
            ),
        )
        state.ensemble_report = report
        state.ensemble_applied = True
        state.record(
            "ensemble",
            "built",
            {"strategy": best_strategy, "cv_mean_score": cv_mean, "members": member_names},
        )
        log_params(state.mlflow_run_id, {"ensemble_strategy": best_strategy})
        log_stage_metrics(state.mlflow_run_id, "ensemble", {"cv_mean_score": cv_mean})
        log_json_artifact(state.mlflow_run_id, report, "ensemble/report.json")
        return report.model_dump_json()
