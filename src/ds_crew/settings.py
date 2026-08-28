"""Env-driven constants. Loaded once at import time via python-dotenv."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_float_or_none(name: str) -> float | None:
    val = os.getenv(name)
    return float(val) if val else None


# The Foundry *project* endpoint -- the project's control plane, which looks
# like https://<account>.services.ai.azure.com/api/projects/<project>. Used
# only by ds_crew.foundry, which drives agents hosted in that project over the
# OpenAI-compatible Responses API rather than calling a model directly. Auth
# is Entra (DefaultAzureCredential, i.e. `az login`), not an API key: the
# Agents surface does not accept one.
AZURE_FOUNDRY_PROJECT_ENDPOINT = os.getenv("AZURE_FOUNDRY_PROJECT_ENDPOINT") or None

# The same account's plain Azure-OpenAI-compatible endpoint (a different
# hostname from the project endpoint above, e.g.
# https://<account>.cognitiveservices.azure.com/ vs.
# https://<account>.services.ai.azure.com/api/projects/<project> -- not
# derivable from one another). Used only by ds_crew.maf.azure_evaluation to
# reach a plain chat-completions deployment as an LLM judge; the Azure AI
# Evaluation SDK's evaluators need that, not the Agents surface. Auth is
# Entra, same as above.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or None
# Deployment used as the judge model for on-demand evaluation
# (`ds-crew-maf --evaluate`). Deliberately the same tier most pipeline
# stages already run on, not the cheapest available -- a judge scoring
# other agents' work benefits from at least their own capability.
AZURE_OPENAI_JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", "ds-standard")

# Management-plane coordinates for `ds-crew-maf --check-models`
# (ds_crew.maf.model_lifecycle). Not derivable from the data-plane endpoints
# above -- deployment/model-catalog lifecycle data (deprecation dates,
# retirement status) only exists on the `management.azure.com` control
# plane, scoped by subscription/resource group/location rather than by
# account hostname. The account name itself *is* derived, from
# AZURE_OPENAI_ENDPOINT's hostname.
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") or None
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") or None
AZURE_LOCATION = os.getenv("AZURE_LOCATION", "eastus")

# Optional per-million-token rates for whichever deployment a Foundry stage
# uses (ds_crew.foundry.stages pins one per stage), used only to turn the
# token counts the Responses API reports into an estimated USD figure. Left
# unset by default and deliberately so: with no rates configured no cost
# metric is logged at all, which honestly represents an unpriced or free-tier
# endpoint rather than asserting a misleading $0.00. Rates are provider- and
# model-specific and go stale, so they belong in .env, not in code.
LLM_PRICE_PER_1M_INPUT = _env_float_or_none("LLM_PRICE_PER_1M_INPUT")
LLM_PRICE_PER_1M_OUTPUT = _env_float_or_none("LLM_PRICE_PER_1M_OUTPUT")

# Shared key gating the HTTP tool service (ds_crew.service). Required to create a
# run; creating one mints a per-run token that every subsequent tool call must
# present. Left unset the service refuses to start, rather than defaulting to an
# open endpoint that can mutate datasets and train models.
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY") or None

# Absolute public base URL the service is reachable at, emitted as the OpenAPI
# `servers` entry. Required by Azure AI Foundry's OpenAPI tool, which resolves
# operation paths against it and has no other way to learn the host: the spec
# FastAPI generates carries only relative paths. Set it to whatever fronts the
# service (a dev tunnel while iterating, a Container Apps ingress once deployed).
# Unset is fine for local use, where a relative spec works.
SERVICE_PUBLIC_URL = os.getenv("SERVICE_PUBLIC_URL") or None

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "structured-ml-crew")

# Application Insights connection string for ds_crew.maf's OTel traces and
# metrics (agent_framework emits spans by default, and ds_crew.maf.telemetry
# adds its own refusal/denial/retry counters -- this only decides where both
# are exported to): the operational record (per-call spans, tokens, latency,
# the model version actually served behind each stage's deployment). MLflow
# above is the decision record (leaderboard, chosen model, human verdict) --
# its lifecycle is tied to a run's HTTP request cycle, see logging_tools.py.
# Unset is fine here: spans/metrics are still created, just dropped instead
# of exported, which is today's behavior and not a regression.
APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") or None

# Off by default, matching agent_framework's own default: sensitive telemetry
# additionally exports raw prompt/response content and tool-call arguments,
# not just metadata (token counts, durations, operation names). Meant for
# debugging a specific run, not left on against a shared App Insights resource.
ENABLE_SENSITIVE_TELEMETRY = _env_bool("ENABLE_SENSITIVE_TELEMETRY", False)

AUTO_APPROVE = _env_bool("AUTO_APPROVE", False)

MAX_HPO_TRIALS = _env_int("MAX_HPO_TRIALS", 100)
MAX_HPO_TIMEOUT_S = _env_int("MAX_HPO_TIMEOUT_S", 1800)
NEAR_PERFECT_THRESHOLD = _env_float("NEAR_PERFECT_THRESHOLD", 0.995)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)

DEFAULT_TEST_SIZE = _env_float("DEFAULT_TEST_SIZE", 0.2)
DEFAULT_CV_FOLDS = _env_int("DEFAULT_CV_FOLDS", 5)
DEFAULT_TOP_K_FOR_HPO = _env_int("DEFAULT_TOP_K_FOR_HPO", 2)

# Ensembling: a hard cap on distinct member models (diminishing returns and
# rising weight-overfit risk beyond ~5 diverse tabular learners), a bounded
# Optuna trial budget for weight optimization (cheap -- it scores cached
# out-of-fold predictions, not refit models), and a minimum CV-metric margin
# an ensemble must beat the best single model by to be recommended over it.
MAX_ENSEMBLE_MEMBERS = _env_int("MAX_ENSEMBLE_MEMBERS", 5)
ENSEMBLE_WEIGHT_TRIALS = _env_int("ENSEMBLE_WEIGHT_TRIALS", 60)
MIN_ENSEMBLE_IMPROVEMENT = _env_float("MIN_ENSEMBLE_IMPROVEMENT", 0.0)

# Explainability: every knob here exists to keep the stage's cost bounded and
# its output reviewable by a human, the same way MAX_HPO_TRIALS/
# MAX_ENSEMBLE_MEMBERS bound their stages. EXPLAIN_MAX_ROWS subsamples X_test
# before any SHAP call (exact tree SHAP is fast, but the kernel/linear paths
# and permutation importance both scale with row count); EXPLAIN_TOP_K_FEATURES
# bounds how much of a wide feature matrix reaches LLM context, mirroring
# EDA_DETAILED_COLUMN_LIMIT's role for the EDA stage.
EXPLAIN_MAX_ROWS = _env_int("EXPLAIN_MAX_ROWS", 500)
EXPLAIN_PERMUTATION_REPEATS = _env_int("EXPLAIN_PERMUTATION_REPEATS", 10)
EXPLAIN_TOP_K_FEATURES = _env_int("EXPLAIN_TOP_K_FEATURES", 15)
EXPLAIN_LOCAL_EXAMPLES = _env_int("EXPLAIN_LOCAL_EXAMPLES", 3)
EXPLAIN_SURROGATE_MAX_DEPTH = _env_int("EXPLAIN_SURROGATE_MAX_DEPTH", 3)

# Columns with more unique values than this (relative to row count) are
# treated as classification only if within MAX_CLASSIFICATION_CLASSES;
# otherwise the target is treated as regression.
MAX_CLASSIFICATION_CLASSES = _env_int("MAX_CLASSIFICATION_CLASSES", 20)

# EDA truncation policy: detailed per-column profiling above this width falls
# back to a summarized view to keep LLM context bounded on wide datasets.
EDA_DETAILED_COLUMN_LIMIT = _env_int("EDA_DETAILED_COLUMN_LIMIT", 50)
