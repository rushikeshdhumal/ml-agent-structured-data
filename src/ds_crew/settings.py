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


MODEL = os.getenv("MODEL", "gpt-4o")

# Optional: point MODEL at any OpenAI-compatible endpoint (NVIDIA NIM, Together,
# Groq, a local vLLM server, ...) that isn't one of CrewAI's built-in named
# providers. When LLM_BASE_URL is set, crew.py routes through CrewAI's native
# custom_openai path instead of passing MODEL as a bare provider-prefixed string.
# LLM_API_KEY falls back to the provider's own env var (e.g. OPENAI_API_KEY) if unset.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY") or None

# Caps aggregate LLM calls per minute across the whole crew (CrewAI's RPMController).
# Leave unset for no cap; set it to stay under a provider's free-tier rate limit.
MAX_RPM = _env_int_or_none("MAX_RPM")

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

# Columns with more unique values than this (relative to row count) are
# treated as classification only if within MAX_CLASSIFICATION_CLASSES;
# otherwise the target is treated as regression.
MAX_CLASSIFICATION_CLASSES = _env_int("MAX_CLASSIFICATION_CLASSES", 20)

# EDA truncation policy: detailed per-column profiling above this width falls
# back to a summarized view to keep LLM context bounded on wide datasets.
EDA_DETAILED_COLUMN_LIMIT = _env_int("EDA_DETAILED_COLUMN_LIMIT", 50)
