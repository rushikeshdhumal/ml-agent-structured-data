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


def _env_int_or_none(name: str) -> int | None:
    val = os.getenv(name)
    return int(val) if val else None


def _env_float_or_none(name: str) -> float | None:
    val = os.getenv(name)
    return float(val) if val else None


MODEL = os.getenv("MODEL", "gpt-4o")

# Optional: point MODEL at any OpenAI-compatible endpoint (NVIDIA NIM, Together,
# Groq, a local vLLM server, ...) that isn't one of CrewAI's built-in named
# providers. When LLM_BASE_URL is set, crew.py routes through CrewAI's native
# custom_openai path instead of passing MODEL as a bare provider-prefixed string.
# LLM_API_KEY falls back to the provider's own env var (e.g. OPENAI_API_KEY) if unset.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY") or None

# Azure AI Foundry. Set the endpoint copied from the Foundry portal (any of the
# forms crew.foundry_base_url normalizes) and a key, and the crew runs against a
# Foundry deployment. Takes precedence over LLM_BASE_URL when both are set.
#
# This deliberately reuses the OpenAI-compatible path rather than adding
# crewai[azure-ai-inference] or crewai[litellm]: Foundry exposes an
# OpenAI-compatible `/openai/v1` surface, so the provider CrewAI already ships
# natively can talk to it unchanged. Given how much of this project's
# dependency surface is already load-bearing (see pyproject.toml's numpy/numba
# comments), not adding a heavy provider SDK for an endpoint that speaks a
# protocol we already support is the cheaper and more stable trade.
#
# MODEL means different things per Foundry flavour, and this is the most common
# misconfiguration: for an Azure OpenAI deployment it is the *deployment name*
# you chose, not the model name; for Foundry Models it is the catalog model name.
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT") or None
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY") or None

# Caps aggregate LLM calls per minute across the whole crew (CrewAI's RPMController).
# Leave unset for no cap; set it to stay under a provider's free-tier rate limit.
MAX_RPM = _env_int_or_none("MAX_RPM")

# Optional per-million-token rates for the configured MODEL, used only to turn
# the token counts CrewAI reports into an estimated USD figure in MLflow. Left
# unset by default and deliberately so: with no rates configured no cost metric
# is logged at all, which honestly represents a free-tier endpoint (e.g. NVIDIA
# NIM) rather than asserting a misleading $0.00. Rates are provider- and
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
