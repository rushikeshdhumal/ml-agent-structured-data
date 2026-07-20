# DS-Crew

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Built with CrewAI](https://img.shields.io/badge/built%20with-CrewAI-6f42c1.svg)](https://docs.crewai.com)

A [CrewAI](https://docs.crewai.com) multi-agent system that runs the full
data-science lifecycle for structured (tabular) data -- EDA, cleaning,
feature engineering, model selection, hyperparameter optimization,
evaluation, and metadata logging -- on top of deterministic, strictly-typed
Python tools instead of trusting an LLM to touch your data directly.

Raw data acquisition is explicitly out of scope: you hand it a CSV and a
target column, and it takes it from there.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rushikeshdhumal/ml-agent-structured-data.git
cd ml-agent-structured-data
uv sync --extra dev
cp .env.example .env   # set MODEL + the matching API key (see comments in the file)
```

Run the crew against a dataset:

```bash
uv run ds-crew --data path/to/data.csv --target target_column
```

Each invocation is one dataset, one CrewAI run, one MLflow run. By default
you'll be prompted at the console to review/edit the cleaning plan, the
feature-engineering plan, and the final model before anything irreversible
happens; set `AUTO_APPROVE=1` in `.env` to skip these gates for
automated/headless runs (every auto-approved run is tagged
`auto_approve=true` in MLflow so it's never mistaken for a real sign-off).

Inspect results:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run the tests:

```bash
uv run pytest                    # unit tests only (default; no LLM calls)
uv run pytest -m e2e tests/e2e   # full pipeline, requires a live LLM key
```

## Architecture

Six agents run a fixed nine-task pipeline. Every task that touches the
dataset delegates to a deterministic tool -- the agent proposes, code
executes:

```mermaid
flowchart TD
    CSV["CSV + target column"]
    EDA["**eda_analyst**<br/>eda_task"]
    PC["**cleaning_strategist**<br/>propose_cleaning_task 👤"]
    EC["**cleaning_strategist**<br/>execute_cleaning_task"]
    PF["**feature_engineer**<br/>propose_feature_task 👤 🛡️"]
    EF["**feature_engineer**<br/>execute_feature_task<br/><sub>train/test split</sub>"]
    MS["**model_selector**<br/>model_selection_task<br/><sub>CV leaderboard</sub>"]
    HPO["**hpo_tuner**<br/>hpo_task<br/><sub>Optuna, budget-capped</sub>"]
    EV["**evaluator**<br/>evaluation_task 👤<br/><sub>X_test touched once</sub>"]
    FIN["**evaluator**<br/>finalize_task<br/><sub>sign-off</sub>"]

    CSV --> EDA --> PC --> EC --> PF --> EF --> MS --> HPO --> EV --> FIN

    subgraph Tools["Deterministic tools (pandas / scikit-learn / Optuna)"]
        T1[[cleaning_tools]]
        T2[[feature_tools]]
        T3[[model_tools]]
        T4[[hpo_tools]]
        T5[[eval_tools]]
        T6[[logging_tools]]
    end

    EC -.-> T1
    EF -.-> T2
    MS -.-> T3
    HPO -.-> T4
    EV -.-> T5
    FIN -.-> T6

    DS[("DataStore<br/><sub>per-run DataFrames --<br/>never enters LLM context</sub>")]
    MLF[("MLflow<br/><sub>sqlite:///mlflow.db</sub>")]

    T1 & T2 & T3 & T4 & T5 --> DS
    T1 & T2 & T3 & T4 & T5 & T6 --> MLF

    classDef gate fill:#fff3cd,stroke:#b38600
    class PC,PF,EV gate
```

👤 = human-in-the-loop gate &nbsp;&nbsp; 🛡️ = task guardrail

**Agents never manipulate data directly.** Every mutating action --
profiling, cleaning, encoding, training, tuning, evaluating -- goes through a
Pydantic-validated tool. This is enforced in layers:

1. Pydantic `args_schema` / `output_pydantic` on every tool and "propose"
   task rejects structurally invalid output before it reaches any logic.
2. Every mutating tool re-validates its input against the *actual current
   dataset* at call time (unknown columns, target-as-feature, disallowed
   strategies all come back as a structured error, never a silent no-op).
3. Task-level `guardrail` functions (`guardrails.py`) catch business-rule
   violations (e.g. target leakage) and trigger CrewAI's automatic
   agent-retry loop.
4. The corresponding mutating tool independently re-checks the same rule --
   a guardrail only covers one task's output, so the tool call is a separate
   trust boundary that must not blindly trust upstream validation.
5. Hyperparameter search budgets (`n_trials`, `timeout_s`) are hard-capped in
   code (`settings.MAX_HPO_TRIALS` / `MAX_HPO_TIMEOUT_S`), regardless of what
   an agent requests.

**Human-in-the-loop.** Applying a cleaning plan, applying a
feature-engineering plan (which also performs the train/test split), and
accepting a final model are gated by CrewAI's native `human_input` Task
flag: an agent first *proposes* a structured plan, a human reviews/edits it
at the console, and only then does a separate *execute* task call the
mutating tool with the approved plan.

**Metadata logging** (`tools/logging_tools.py`) is deterministic code, not
an agent -- it's wired to fire automatically from inside each tool and from
`main.py`'s run lifecycle, rather than depending on an agent remembering to
log something. Everything lands in MLflow against a local SQLite-backed
tracking store, no server required.

### Model registry

`model_tools.py` cross-validates a fixed, code-defined candidate set chosen
for empirical strength on tabular data rather than breadth:

| Task | Candidates |
|---|---|
| Classification | `logistic_regression`, `knn`, `xgboost`, `lightgbm`, `catboost` |
| Regression | `ridge`, `elastic_net`, `knn`, `xgboost`, `lightgbm`, `catboost` |

The agent never chooses which model classes exist, only which leaderboard
entries move on to HPO (`hpo_tools.py`, Optuna, one fixed search space per
model) and evaluation. A failing candidate is skipped and recorded in
`Leaderboard.warnings` rather than crashing the whole leaderboard.

## Project layout

```
src/ds_crew/
  main.py           CLI entrypoint; owns the MLflow run lifecycle
  crew.py           Agent/Task/Crew wiring (tools, guardrails, output schemas, HITL flags)
  state.py          DataStore/RunState -- the actual DataFrames live here, never in LLM context
  schemas.py        Pydantic models for every plan/report passed between tasks
  guardrails.py     Function-based CrewAI Task guardrails
  settings.py       Env-driven constants (model, budgets, MLflow config, AUTO_APPROVE)
  config/
    agents.yaml     Agent role/goal/backstory
    tasks.yaml      Task descriptions, context wiring
  tools/
    eda_tools.py       Read-only profiling
    cleaning_tools.py  Missing-value/outlier/dtype cleaning
    feature_tools.py   Train/test split, encoding, scaling, feature selection
    model_tools.py     Candidate model registry + cross-validation
    hpo_tools.py       Optuna hyperparameter search
    eval_tools.py      Held-out evaluation
    logging_tools.py   MLflow helpers + the finalize_run tool
tests/              Unit tests for every tool/guardrail/schema (no LLM calls)
tests/e2e/          Full pipeline test; requires a live LLM key, opt-in via `-m e2e`
```

## Configuration

`.env` (copy from `.env.example`) configures:

- **LLM** -- `MODEL`, provider-agnostic via CrewAI's native provider routing
  (`gpt-4o`, `anthropic/claude-sonnet-4-5-...`, `ollama/llama3`, ...), or
  `LLM_BASE_URL` + `LLM_API_KEY` for any other OpenAI-compatible endpoint
  (e.g. NVIDIA NIM). `MAX_RPM` caps aggregate LLM calls/minute across the crew.
- **MLflow** -- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`.
- **Guardrails/budgets** -- `AUTO_APPROVE`, `MAX_HPO_TRIALS`,
  `MAX_HPO_TIMEOUT_S`, `NEAR_PERFECT_THRESHOLD`, `RANDOM_SEED`.

## License

[MIT](LICENSE) © 2026 Rushikesh Dhumal
