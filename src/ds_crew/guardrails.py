"""Function-based CrewAI Task guardrails.

Per CrewAI's contract, a guardrail takes exactly one argument (the task's
TaskOutput) and returns `(bool, Any)` -- `(True, value)` on pass, `(False,
error)` to trigger CrewAI's automatic agent retry loop (up to
`guardrail_max_retries`).

These guardrails improve the propose-stage retry UX; they are NOT the actual
security boundary. Each mutating tool (ApplyCleaningPlanTool,
ApplyFeaturePlanTool) re-checks the identical rule independently, because a
guardrail only ever covers one Task's output -- the mutating tool call is a
separate trust boundary and must not blindly trust upstream validation.

Guardrails must never be used to express a human's "no": a `(False, ...)`
return triggers an automatic agent *retry*, which is the wrong semantics for
"the human rejected this." A human's rejection is instead modeled as a
structured `approved: bool` field (see `schemas.FinalSignOff`) that
`FinalizeRunTool` inspects and branches on.
"""

from typing import Any, Tuple

from ds_crew.schemas import CleaningPlan, FeatureEngineeringPlan
from ds_crew.state import get_data_store


def prevent_target_leakage_guardrail(output: Any) -> Tuple[bool, Any]:
    """Attached to propose_feature_task: refuses a plan that treats the target as a feature."""
    plan = output.pydantic
    if plan is None or not isinstance(plan, FeatureEngineeringPlan):
        return False, "Expected a FeatureEngineeringPlan; structured output missing or malformed."
    try:
        state = get_data_store().get(plan.run_id)
    except KeyError as exc:
        return False, str(exc)

    leaking = {cp.column for cp in plan.column_plans if cp.column == state.target}
    if leaking:
        return False, (
            f"Plan illegally includes target column '{state.target}' as a feature. "
            "Remove it from column_plans and resubmit."
        )
    return True, plan


def prevent_target_modification_guardrail(output: Any) -> Tuple[bool, Any]:
    """Attached to propose_cleaning_task: refuses a plan that touches the target column."""
    plan = output.pydantic
    if plan is None or not isinstance(plan, CleaningPlan):
        return False, "Expected a CleaningPlan; structured output missing or malformed."
    try:
        state = get_data_store().get(plan.run_id)
    except KeyError as exc:
        return False, str(exc)

    touched = {a.column for a in plan.actions} | set(plan.columns_to_drop)
    if state.target in touched:
        return False, (
            f"Plan illegally modifies or drops the target column '{state.target}'. "
            "Remove it and resubmit."
        )
    return True, plan


def make_finalize_called_guardrail(run_id: str):
    """Attached to finalize_task, which has no output_pydantic (its job is to
    make a tool call, not produce structured data) so there's nothing else to
    validate its output against. Observed live under AUTO_APPROVE: with no
    real human feedback to interpret, an agent sometimes responds with prose
    asking a human to reply "approved"/"rejected" instead of ever calling
    finalize_run -- the task then "completes" without the decision actually
    being recorded. This closes that gap by checking DataStore history for a
    completed finalize/sign_off record before letting the task pass.

    Needs `run_id` closed over from crew.py (there's no output.pydantic to
    read it from here, unlike the other guardrails).
    """

    def _guardrail(output: Any, _run_id: str = run_id) -> Tuple[bool, Any]:
        try:
            state = get_data_store().get(_run_id)
        except KeyError as exc:
            return False, str(exc)

        called = any(h["stage"] == "finalize" and h["action"] == "sign_off" for h in state.history)
        if not called:
            return False, (
                "finalize_run was not called. You must call finalize_run with an "
                "explicit approved=true or approved=false decision before this task "
                "can complete -- do not just describe what you would do or ask a "
                "question; call the tool now with your best-supported decision."
            )
        return True, output

    return _guardrail
