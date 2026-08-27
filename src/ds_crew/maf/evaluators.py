"""Deterministic post-hoc checks over stage narration vs. a tool's own output.

Closes two named gaps: `evaluation_task` has no guardrail tying
`evaluate_models`' `leakage_suspicion` flag to what a human actually sees
before approving a model (the thinnest safety net in the pipeline -- see
`ds_crew.foundry.stages`' docstring on why sequencing is code, not an LLM's
judgment), and the CrewAI-era `make_explanation_grounded_guardrail`, removed
along with CrewAI and never replaced (see `explain_tools.py`'s module
docstring).

Both checks are deterministic string/JSON comparisons, not an LLM judge: the
risk here is a narration diverging from a tool's own structured output, which
is exactly checkable, and a judge model would add cost, latency, and its own
failure mode (a judge can itself hallucinate) to a safety net that doesn't
need either. `ds_crew.maf.executors.GroundingCheckExecutor` is the caller;
these are plain functions so that caller's wiring stays separate from the
actual check logic.

Neither function raises on malformed/missing input -- both degrade to `[]`,
matching this codebase's tone for optional safety nets elsewhere
(`logging_tools.log_llm_usage`: "losing a metric is strictly better than
losing a stack trace"). A tool result that never got captured (e.g. a stage
that was entirely a `ConversationPoisoned` restart) is not a reason to crash
a guardrail whose whole job is to be extra caution, not a hard dependency.
"""

from __future__ import annotations

import json
import re


def _load_reports(bundle_json: str | None) -> list[dict]:
    if not bundle_json:
        return []
    try:
        parsed = json.loads(bundle_json)
    except (json.JSONDecodeError, TypeError):
        return []
    reports = parsed.get("reports") if isinstance(parsed, dict) else None
    return reports if isinstance(reports, list) else []


def find_leakage_suspicions(evaluation_bundle_json: str | None) -> list[str]:
    """Every model `evaluate_models` itself flagged `leakage_suspicion` for.

    Unconditional -- surfaced regardless of whether the agent's own narration
    already mentioned it. Redundancy here costs nothing; a missed leakage
    warning reaching a human sign-off gate costs a lot.
    """
    findings = []
    for report in _load_reports(evaluation_bundle_json):
        if not report.get("leakage_suspicion"):
            continue
        name = report.get("model_name", "<unknown model>")
        notes = report.get("notes") or "metric looks too good to be true."
        findings.append(f"{name}: {notes}")
    return findings


def find_ungrounded_model_mentions(
    narration: str, evaluation_bundle_json: str | None, explanation_bundle_json: str | None
) -> list[str]:
    """Models the narration discusses by name that `explain_models` never
    produced a report for.

    The candidate vocabulary to scan for is this run's own evaluated models
    (from the evaluation bundle), not a hardcoded or imported model registry
    -- that would recouple `ds_crew.maf` to `ds_crew.tools` across what's
    otherwise a clean HTTP/MCP process boundary, for a check that doesn't
    need it. Evaluated-minus-explained is the direct MAF-era equivalent of
    the old `make_explanation_grounded_guardrail`'s "reported model names not
    in state.explanation_reports" check, adapted for prose narration instead
    of a validated `output_pydantic` object (which doesn't exist in this
    architecture -- see `foundry/stages.py`'s docstring).
    """
    if not narration:
        return []
    evaluated = {r.get("model_name") for r in _load_reports(evaluation_bundle_json)}
    explained = {r.get("model_name") for r in _load_reports(explanation_bundle_json)}
    unexplained = sorted(name for name in evaluated - explained if name)

    findings = []
    for name in unexplained:
        if re.search(rf"\b{re.escape(name)}\b", narration, re.IGNORECASE):
            findings.append(
                f"narration mentions '{name}', which was evaluated but explain_models "
                "produced no report for it -- the explanation may be describing the "
                "wrong model."
            )
    return findings


def format_warning_block(findings: list[str]) -> str:
    lines = "\n".join(f"  - {f}" for f in findings)
    return (
        "AUTOMATED SAFETY CHECK (ds_crew.maf.evaluators, not written by the "
        f"explainer) -- review before approving:\n{lines}"
    )
