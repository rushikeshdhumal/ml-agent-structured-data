"""Drive the eight Foundry agents through one full pipeline run.

Ordering comes from `stages.STAGES`, context is carried forward explicitly, and
the four gated tools pause for a human unless `--auto-approve` is passed. That
mirrors DS-Crew's existing `AUTO_APPROVE` setting rather than inventing a second
convention for the same idea.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ds_crew import settings
from ds_crew.foundry.runner import AgentRunner, ApprovalRequest, StageResult
from ds_crew.foundry.stages import GATED_TOOLS, STAGES, STAGES_BY_KEY, Stage


class PreflightError(RuntimeError):
    """Raised when the environment cannot support a run that would complete."""


@dataclass
class RunReport:
    run_id: str
    results: dict[str, StageResult] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results.values())

    @property
    def transport_retries(self) -> int:
        return sum(r.transport_retries for r in self.results.values())

    def cost_usd(self) -> float | None:
        """Only when both rates are configured; otherwise no number at all.

        Reporting $0.00 against unconfigured rates would be a lie dressed as a
        measurement, the same reasoning `settings.LLM_PRICE_PER_1M_*` already
        documents for the MLflow cost metric.
        """
        cin, cout = settings.LLM_PRICE_PER_1M_INPUT, settings.LLM_PRICE_PER_1M_OUTPUT
        if cin is None or cout is None:
            return None
        return self.input_tokens * cin / 1e6 + self.output_tokens * cout / 1e6


def preflight(*, timeout_s: float = 10.0, attempts: int = 3) -> None:
    """Refuse to start a run the environment cannot finish.

    A pipeline that dies at stage 6 of 9 leaves a run with cleaning and features
    applied and no way to resume: the `*_applied` guards are one-shot by design,
    so the half-finished run can only be abandoned. Checking first is much
    cheaper than discovering it later, and the dev tunnel has been measured
    dropping requests outright.
    """
    if not settings.AZURE_FOUNDRY_PROJECT_ENDPOINT:
        raise PreflightError(
            "AZURE_FOUNDRY_PROJECT_ENDPOINT is not set. It looks like "
            "https://<account>.services.ai.azure.com/api/projects/<project> and is "
            "distinct from AZURE_FOUNDRY_ENDPOINT."
        )
    if not settings.SERVICE_PUBLIC_URL:
        raise PreflightError(
            "SERVICE_PUBLIC_URL is not set, so Foundry has no address for the tool "
            "service."
        )

    health = settings.SERVICE_PUBLIC_URL.rstrip("/") + "/healthz"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(health, timeout=timeout_s) as resp:
                if resp.status == 200:
                    return
                last = RuntimeError(f"HTTP {resp.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(2)
    raise PreflightError(
        f"The tool service is not reachable at {health} after {attempts} attempts "
        f"({last}). Foundry reaches it over the same path, so a run would fail "
        "part-way. Check the tunnel is forwarded and public, and that the service "
        "is running."
    )


def _auto_decider(log: Any) -> Any:
    def decide(req: ApprovalRequest) -> tuple[bool, str]:
        log(f"    [auto-approved] {req.tool}")
        return True, "AUTO_APPROVE is set."

    return decide


def _interactive_verdict(explanation_text: str, log: Any) -> str:
    """Show the explanation and collect the human's actual sign-off, live.

    Without this, the finalize stage's verdict has to be written before the run
    starts and before the explanation exists to react to -- which contradicts
    the entire thesis of this pipeline (a human reviews the evidence, then
    decides). `finalize_task`'s instructions correctly refuse to record
    approval without an unambiguous statement in the conversation, so skipping
    this prompt does not fail loudly: it silently produces a rejection.
    """
    log("")
    log("  " + "=" * 68)
    log("  EXPLANATION -- read this before deciding")
    log("  " + "=" * 68)
    for line in explanation_text.splitlines():
        log(f"    {line}")
    log("")
    while True:
        answer = input(
            "  Approve the recommended model for production? "
            "[y]es / [n]o (reasoning optional after either): "
        ).strip()
        if not answer:
            continue
        head, _, tail = answer.partition(" ")
        head = head.lower()
        if head in ("y", "yes"):
            return f"I approve the recommended model. {tail.strip()}".strip()
        if head in ("n", "no"):
            return f"I do NOT approve the recommended model. {tail.strip()}".strip()
        log("  Please answer y or n.")


def _interactive_decider(log: Any) -> Any:
    def decide(req: ApprovalRequest) -> tuple[bool, str]:
        log("")
        log("  " + "=" * 68)
        log(f"  HUMAN APPROVAL REQUIRED -- {req.agent} wants to call {req.tool}")
        log("  " + "=" * 68)
        for line in req.pretty_arguments().splitlines():
            log(f"    {line}")
        log("")
        while True:
            answer = input("  Approve? [y]es / [n]o (reason optional after n): ").strip()
            if not answer:
                continue
            head, _, tail = answer.partition(" ")
            head = head.lower()
            if head in ("y", "yes"):
                return True, tail.strip()
            if head in ("n", "no"):
                return False, tail.strip() or "Rejected by the operator."
            log("  Please answer y or n.")

    return decide


def _context_block(stage: Stage, report: RunReport) -> str:
    parts: list[str] = []
    for key in stage.needs:
        result = report.results.get(key)
        if result is None or not result.text:
            continue
        parts.append(
            f"### Output of the {STAGES_BY_KEY[key].agent} stage\n\n{result.text}"
        )
    return "\n\n".join(parts)


def _build_prompt(stage: Stage, report: RunReport, verdict: str) -> str:
    body = stage.prompt.format(
        run_id=report.run_id,
        explanation=(report.results["explanation"].text if "explanation" in report.results else ""),
        verdict=verdict,
    )
    # `finalize` embeds the explanation in its own template, so prefixing the
    # generic context block as well would send it twice.
    if stage.key == "finalize":
        return body
    context = _context_block(stage, report)
    return f"{context}\n\n---\n\n{body}" if context else body


# How many follow-up turns a stage gets, whether it's stalling or being told
# "no" -- either way, this is the budget before the orchestrator gives up.
MAX_STAGE_ATTEMPTS = 3

_NUDGE = (
    "You have not completed the tool call yet. Do not ask for permission in prose: "
    "call {tools} now with the arguments you described. Gated tools are paused for "
    "human approval by the platform, which shows me your arguments before anything "
    "runs, so state your reasoning in the same turn as the call rather than waiting "
    "for a reply first."
)

_REVISE = (
    "The human reviewing this did not approve your last call to {tools}. Their "
    "reasoning is already in this conversation, attached to that decision. Revise "
    "your proposal to address it and call {tools} again in this same turn -- do not "
    "just repeat the same arguments, and do not merely respond in prose and wait."
)


class StageDidNotAct(RuntimeError):
    """A stage produced prose but never ran the tool it exists to run."""


class GateNotApproved(RuntimeError):
    """A human declined a gate on every revision attempt offered.

    Distinct from StageDidNotAct: here the agent acted and a human decided,
    every round. That is the system working as designed, not a stall, so
    run_pipeline stops cleanly with a summary rather than crashing on it.
    """


class ForbiddenToolCalled(RuntimeError):
    """A stage ran a gated tool that belongs to a later, separate turn.

    Raised immediately rather than caught by the next stage's precondition
    check, because the call already happened and is often irreversible
    (finalize_run is one-shot server-side). Observed live: a per-turn "do not
    call this yet" instruction lost to the evaluator's own persistent job
    description, which lists the forbidden tool as its natural next step. The
    run recorded a rejection three stages before the explanation it was
    supposed to wait for, and only surfaced as a confusing failure two stages
    later, once the tool's one-shot guard refused the real, correctly-timed
    call.
    """


def _run_stage_until_it_acts(runner, stage, prompt, decide, previous, log):
    """Run a stage, following up when it stalls or is turned down, until it acts.

    Two different silences look alike (no expected tool call yet) but need
    different follow-ups. An agent that stalls -- instructed to explain a gated
    action before taking it, which with a human typing reads as good manners --
    needs telling to act instead of asking; no `mcp_approval_request` was ever
    raised, because no call was attempted. An agent whose proposal was denied
    already has the human's reason in its own conversation (Foundry attaches it
    to the `mcp_approval_response`) and needs telling to revise, not to repeat
    the same call or wait for a different answer to the same one.

    Treating a denial as a hard stop here was the earlier bug: a human typing a
    substantive reason ("will f1 score be a better metric") got the whole run
    killed instead of a revised proposal to react to -- the same silent-failure
    shape as the stall this function was already written to catch.
    """
    result = runner.run(
        agent=stage.agent,
        deployment=stage.deployment,
        prompt=prompt,
        decide=decide,
        previous_response_id=previous,
    )
    _check_forbidden(stage, result)

    expected = set(stage.expects_tools)
    if not expected:
        return result

    for attempt in range(MAX_STAGE_ATTEMPTS):
        missing = expected - result.succeeded_tools()
        if not missing:
            return result

        refused = sorted(set(result.refused_tools) & expected)
        if refused:
            # The tool ran and said no. Nudging cannot help: the refusal is a
            # decision about run state, and repeating the call just repeats it.
            raise StageDidNotAct(
                f"Stage '{stage.key}': {', '.join(refused)} refused the call. "
                f"The agent's account was:\n\n{result.text[:800]}"
            )

        denied = sorted(set(result.denied) & missing)
        if denied:
            log(
                f"    ~ human denied {', '.join(denied)}; asking the agent to revise "
                f"({attempt + 1}/{MAX_STAGE_ATTEMPTS})"
            )
            follow_up_prompt = _REVISE.format(tools=", ".join(sorted(missing)))
        else:
            log(
                f"    ~ agent stalled without calling {', '.join(sorted(missing))}; nudging "
                f"({attempt + 1}/{MAX_STAGE_ATTEMPTS})"
            )
            follow_up_prompt = _NUDGE.format(tools=", ".join(sorted(missing)))

        follow_up = runner.run(
            agent=stage.agent,
            deployment=stage.deployment,
            prompt=follow_up_prompt,
            decide=decide,
            previous_response_id=result.response_id,
        )
        _check_forbidden(stage, follow_up)
        _merge(result, follow_up)

    missing = expected - result.succeeded_tools()
    if set(result.denied) & missing:
        raise GateNotApproved(
            f"Stage '{stage.key}': the human did not approve {', '.join(sorted(missing))} "
            f"after {MAX_STAGE_ATTEMPTS} revisions. Nothing was applied."
        )
    raise StageDidNotAct(
        f"Stage '{stage.key}' never called {', '.join(sorted(missing))} after "
        f"{MAX_STAGE_ATTEMPTS} attempts. Nothing was applied, so the run cannot continue. "
        f"The agent's last message was:\n\n{result.text[:800]}"
    )


def _check_forbidden(stage: Stage, result) -> None:
    hit = set(stage.forbidden_tools) & result.succeeded_tools()
    if hit:
        raise ForbiddenToolCalled(
            f"Stage '{stage.key}' ({stage.agent}) called {', '.join(sorted(hit))}, which "
            f"belongs to a later turn. This may be a one-shot, irreversible action already "
            f"recorded against the run. The agent said:\n\n{result.text[:800]}"
        )


def _merge(base, extra) -> None:
    base.response_id = extra.response_id
    base.input_tokens += extra.input_tokens
    base.output_tokens += extra.output_tokens
    base.tool_calls.extend(extra.tool_calls)
    base.refused_tools.extend(extra.refused_tools)
    base.approvals.extend(extra.approvals)
    base.denied.extend(extra.denied)
    base.transport_retries += extra.transport_retries
    if extra.text:
        base.text = f"{base.text}\n\n{extra.text}".strip() if base.text else extra.text


def run_pipeline(
    run_id: str,
    *,
    project_client: Any,
    auto_approve: bool | None = None,
    verdict: str = "",
    log: Any = print,
    read_verdict: Any = None,
) -> RunReport:
    """Drive every stage in order, returning a report over the whole run.

    `verdict` supplied here is a decision made before the run even starts, and
    is meant for unattended use (`--auto-approve --verdict "..."`). In the
    interactive case leaving it unset is deliberate: the actual sign-off is
    collected live, after the explanation exists to react to, by
    `read_verdict` (defaults to `_interactive_verdict`, which prints the
    explanation and prompts). Collecting it upfront instead would mean writing
    a verdict on a model nobody has explained yet, which is the opposite of
    what the human gate is for.
    """
    auto = settings.AUTO_APPROVE if auto_approve is None else auto_approve
    decide = _auto_decider(log) if auto else _interactive_decider(log)
    collect_verdict = read_verdict or _interactive_verdict

    if not auto and not verdict:
        # Left empty here; collected live just before the finalize stage below,
        # once report.results["explanation"] exists.
        pass
    elif not verdict:
        # Unattended with nothing to say: recording a sign-off nobody gave
        # would be worse than recording an explicit, honest rejection.
        verdict = "No human reviewed this run; it was executed unattended. Record it as NOT approved."

    runner = AgentRunner(project_client, log=log)
    report = RunReport(run_id=run_id)

    log(f"\nRun {run_id} -- {len(STAGES)} stages, {'auto-approving' if auto else 'human'} gates\n")

    for index, stage in enumerate(STAGES, 1):
        missing = [k for k in stage.critical_context if k not in report.results]
        if missing:
            raise RuntimeError(
                f"Stage '{stage.key}' needs output from {missing}, which did not run. "
                "That context is load-bearing, not advisory: the tool refuses without it."
            )

        if stage.key == "finalize" and not auto and not verdict:
            verdict = collect_verdict(report.results["explanation"].text, log)

        previous = report.results[stage.resume_from].response_id if stage.resume_from else None
        prompt = _build_prompt(stage, report, verdict)

        started = time.time()
        gates = [t for t in stage.tasks if t]
        log(f"[{index}/{len(STAGES)}] {stage.agent:<20} ({', '.join(gates)})")

        try:
            result = _run_stage_until_it_acts(runner, stage, prompt, decide, previous, log)
        except GateNotApproved as exc:
            log(f"    REJECTED: {exc} -- stopping.")
            break
        report.results[stage.key] = result

        tools = ", ".join(sorted(result.succeeded_tools())) or "none"
        log(
            f"    done in {time.time() - started:5.1f}s | tools: {tools} | "
            f"tokens {result.input_tokens}/{result.output_tokens}"
        )
        if result.refused_tools:
            log(f"    refused: {', '.join(sorted(set(result.refused_tools)))}")
        if result.denied:
            revised = sorted(set(result.denied))
            log(f"    (revised after the human first declined: {', '.join(revised)})")

    report.finished_at = time.time()
    _summarize(report, log)
    return report


def _summarize(report: RunReport, log: Any) -> None:
    elapsed = (report.finished_at or time.time()) - report.started_at
    log("")
    log("=" * 78)
    log(f"Run {report.run_id} finished in {elapsed / 60:.1f} min")
    log("=" * 78)
    log(f"{'stage':<16}{'agent':<22}{'in':>8}{'out':>8}  tools")
    for key, result in report.results.items():
        stage = STAGES_BY_KEY[key]
        log(
            f"{key:<16}{stage.agent:<22}{result.input_tokens:>8}{result.output_tokens:>8}  "
            f"{', '.join(result.tool_calls) or '-'}"
        )
    log("")
    log(f"total tokens      : {report.input_tokens} in / {report.output_tokens} out")
    cost = report.cost_usd()
    if cost is None:
        log("estimated cost    : not computed (set LLM_PRICE_PER_1M_INPUT/OUTPUT)")
    else:
        log(f"estimated cost    : ${cost:.4f}")
    if report.transport_retries:
        log(f"transport retries : {report.transport_retries} (tunnel faults, not agent errors)")

    succeeded = {t for r in report.results.values() for t in r.succeeded_tools()}
    gated_called = succeeded & GATED_TOOLS
    log(f"gated tools run   : {', '.join(sorted(gated_called)) or 'none'}")

    expected_gates = {t for s in STAGES_BY_KEY.values() for t in s.expects_tools} & GATED_TOOLS
    if gated_called != expected_gates:
        log(
            f"WARNING: expected {len(expected_gates)} gated tools to run, "
            f"{len(gated_called)} did. Missing: "
            f"{', '.join(sorted(expected_gates - gated_called))}"
        )


def build_project_client() -> Any:
    """Construct an AIProjectClient, with the preview flag agent endpoints need."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    return AIProjectClient(
        endpoint=settings.AZURE_FOUNDRY_PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
        # get_openai_client(agent_name=...) raises ValueError without this.
        allow_preview=True,
    )


def create_run(csv_path: str, target: str, *, task: str = "auto", metric: str | None = None) -> str:
    """Create a run on the tool service and return its id."""
    import pathlib

    payload: dict[str, Any] = {
        "csv_text": pathlib.Path(csv_path).read_text(encoding="utf-8"),
        "target": target,
        "task": task,
    }
    if metric:
        payload["metric"] = metric

    request = urllib.request.Request(
        settings.SERVICE_PUBLIC_URL.rstrip("/") + "/runs",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": settings.SERVICE_API_KEY or ""},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["run_id"]
