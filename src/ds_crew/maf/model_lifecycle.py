"""On-demand model-lifecycle and content-safety visibility check.

`ds-crew-maf --check-models` reports two things Foundry's own portal shows
per-resource but this project never surfaced anywhere as code:

* **Deployment health.** Cross-references this account's deployments
  (`ds-standard`, `ds-evaluator`) against Azure's model catalog for
  retirement status and deprecation dates, and lists still-available,
  tool-calling-capable alternatives when one is close to or past its
  inference-deprecation date. Directly motivated by a real incident: a live
  run failed with HTTP 410 on 2026-08-25 because `z-ai/glm-5.2` had reached
  end of life four days earlier. This data only exists on Azure's
  *management* plane (`management.azure.com`, scoped by subscription/
  resource group/location) -- the data-plane deployment-list endpoint this
  project otherwise uses was retired -- so this is the one place in
  `ds_crew` that needs `AZURE_SUBSCRIPTION_ID`/`AZURE_RESOURCE_GROUP`/
  `AZURE_LOCATION`.
* **Content-safety visibility.** Reads back whatever RAI (Responsible AI)
  policy is already attached to each of the eight agents'
  latest version, via the same `agents.get()` call
  `ds_crew.maf.transport_foundry.FoundryTransport` already makes. Read-only,
  deliberately: agents in this repo are created once, by hand, in the
  Foundry portal (see README.md), not from code, so *setting* `rai_config`
  here would mean adopting programmatic agent versioning -- a materially
  bigger change than this check. `None` back from the SDK means the agent
  inherits the account's default content filter, not that nothing is
  configured.

Deliberately out of scope: managed identity for the tool service (Phase
11's remaining sub-item). That item was written as "once [the tool service
is] on Container Apps", which is Phase 12, undeployed, and previously
recommended to skip for the demo -- the tool service still runs locally
behind a dev tunnel with a static `SERVICE_API_KEY`. Nothing here changes
that.

On-demand only, like `ds_crew.maf.azure_evaluation`: this needs a
credential scope (`https://management.azure.com/.default`) this project
has never exercised before, so it's isolated to one explicit CLI flag
rather than run automatically, until a live call confirms what permissions
the current Entra role actually has.

Live-verification note: the model-catalog response shape below
(`_normalize_catalog_entry`) is built from Microsoft's published REST
reference for `Models - List` (`.../locations/{location}/models`), not
from a captured real response -- field casing is ARM's documented
camelCase, but this has not yet been exercised against the real
`ds-crew-resource` account. If a live `--check-models` run shows different
keys, fix the normalizer here; the cross-referencing logic below it
(`deployment_health`) is unit-tested against synthetic data in that
already-normalized shape and shouldn't need to change.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

from ds_crew import settings
from ds_crew.foundry.stages import STAGES

_MANAGEMENT_API_VERSION = "2024-10-01"
# A deployment within this many days of its inference-deprecation date (or
# already past it) gets migration candidates listed alongside it.
_DEPRECATION_WARNING_DAYS = 90


def _account_name() -> str:
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set -- required to derive the Cognitive "
            "Services account name for --check-models. See .env.example."
        )
    host = urlparse(settings.AZURE_OPENAI_ENDPOINT).hostname or ""
    name = host.split(".")[0]
    if not name:
        raise RuntimeError(
            f"Could not parse an account name out of AZURE_OPENAI_ENDPOINT="
            f"{settings.AZURE_OPENAI_ENDPOINT!r}"
        )
    return name


def _require_mgmt_settings() -> tuple[str, str]:
    if not settings.AZURE_SUBSCRIPTION_ID or not settings.AZURE_RESOURCE_GROUP:
        raise RuntimeError(
            "AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP must both be set for "
            "--check-models -- deployment/model lifecycle data only exists on Azure's "
            "management plane, scoped by subscription and resource group, and isn't "
            "derivable from the data-plane endpoints this project otherwise uses. "
            "See .env.example."
        )
    return settings.AZURE_SUBSCRIPTION_ID, settings.AZURE_RESOURCE_GROUP


def _management_get(path: str, credential: Any) -> dict[str, Any]:
    token = credential.get_token("https://management.azure.com/.default").token
    request = urllib.request.Request(
        f"https://management.azure.com{path}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code == 403:
            raise RuntimeError(
                f"Management-plane request to {path} was forbidden (403). The current "
                "credential needs at least Reader on the resource group (for the "
                "deployments list) and on the subscription (for the location's model "
                f"catalog). Response: {body}"
            ) from exc
        raise RuntimeError(f"Management-plane request to {path} failed ({exc.code}): {body}") from exc


def fetch_deployments(credential: Any) -> list[dict[str, Any]]:
    """This account's deployments, straight from Azure -- unnormalized."""
    sub, rg = _require_mgmt_settings()
    account = _account_name()
    path = (
        f"/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices"
        f"/accounts/{account}/deployments?api-version={_MANAGEMENT_API_VERSION}"
    )
    return _management_get(path, credential).get("value", [])


def fetch_model_catalog(credential: Any) -> list[dict[str, Any]]:
    """The model catalog for AZURE_LOCATION, straight from Azure -- unnormalized."""
    sub, _rg = _require_mgmt_settings()
    path = (
        f"/subscriptions/{sub}/providers/Microsoft.CognitiveServices/locations/"
        f"{settings.AZURE_LOCATION}/models?api-version={_MANAGEMENT_API_VERSION}"
    )
    return _management_get(path, credential).get("value", [])


@dataclass(frozen=True)
class _DeployedModel:
    deployment_name: str
    model_name: str
    model_version: str | None


@dataclass(frozen=True)
class _CatalogModel:
    model_name: str
    model_version: str | None
    lifecycle_status: str | None
    inference_deprecation: date | None
    supports_tool_calling: bool


def _normalize_deployment(raw: dict[str, Any]) -> _DeployedModel:
    model = (raw.get("properties") or {}).get("model") or {}
    return _DeployedModel(
        deployment_name=raw.get("name", "?"),
        model_name=model.get("name", "?"),
        model_version=model.get("version"),
    )


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _normalize_catalog_entry(raw: dict[str, Any]) -> _CatalogModel:
    # See the module docstring's live-verification note -- this key shape
    # (nested under "model", camelCase) is Microsoft's documented REST
    # reference, not yet confirmed against a real response.
    model = raw.get("model", raw)
    deprecation = model.get("deprecation") or {}
    capabilities = model.get("capabilities") or {}
    return _CatalogModel(
        model_name=model.get("name", "?"),
        model_version=model.get("version"),
        lifecycle_status=model.get("lifecycleStatus") or model.get("lifecycle_status"),
        inference_deprecation=_parse_date(deprecation.get("inference")),
        supports_tool_calling=str(capabilities.get("toolCalling", "")).lower() == "true",
    )


@dataclass(frozen=True)
class DeploymentHealth:
    deployment_name: str
    model_name: str
    model_version: str | None
    lifecycle_status: str
    inference_deprecation: date | None
    days_until_deprecation: int | None
    migration_candidates: list[str] = field(default_factory=list)


def deployment_health(
    deployments: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> list[DeploymentHealth]:
    """Cross-reference this account's deployments against the model catalog.

    Pure function of the raw Azure JSON lists -- unit-tested directly, no
    live calls. A deployment whose model isn't in the catalog at all (a
    preview or since-delisted model) reports `lifecycle_status="unknown"`
    rather than raising: that's a real, non-fatal case.
    """
    today = today or date.today()
    catalog_models = [_normalize_catalog_entry(c) for c in catalog]
    by_name_version = {(c.model_name, c.model_version): c for c in catalog_models}
    by_name: dict[str, _CatalogModel] = {}
    for c in catalog_models:
        by_name.setdefault(c.model_name, c)

    rows: list[DeploymentHealth] = []
    for raw in deployments:
        d = _normalize_deployment(raw)
        catalog_entry = by_name_version.get((d.model_name, d.model_version)) or by_name.get(d.model_name)
        if catalog_entry is None:
            rows.append(
                DeploymentHealth(
                    deployment_name=d.deployment_name,
                    model_name=d.model_name,
                    model_version=d.model_version,
                    lifecycle_status="unknown",
                    inference_deprecation=None,
                    days_until_deprecation=None,
                )
            )
            continue

        days_until = None
        if catalog_entry.inference_deprecation is not None:
            days_until = (catalog_entry.inference_deprecation - today).days

        candidates: list[str] = []
        if days_until is not None and days_until <= _DEPRECATION_WARNING_DAYS:
            candidates = sorted(
                {
                    c.model_name
                    for c in catalog_models
                    if c.supports_tool_calling
                    and c.model_name != d.model_name
                    and (c.lifecycle_status or "").lower() not in ("deprecated", "legacy", "retired")
                    and (
                        c.inference_deprecation is None
                        or (c.inference_deprecation - today).days > _DEPRECATION_WARNING_DAYS
                    )
                }
            )[:5]

        rows.append(
            DeploymentHealth(
                deployment_name=d.deployment_name,
                model_name=d.model_name,
                model_version=d.model_version,
                lifecycle_status=catalog_entry.lifecycle_status or "unknown",
                inference_deprecation=catalog_entry.inference_deprecation,
                days_until_deprecation=days_until,
                migration_candidates=candidates,
            )
        )
    return rows


@dataclass(frozen=True)
class AgentRaiStatus:
    agent: str
    rai_policy_name: str | None


def agent_rai_policies(project_client: Any, agent_names: Iterable[str]) -> list[AgentRaiStatus]:
    """One `agents.get()` per unique agent name (same call
    `FoundryTransport._agent_version` already makes), reading back
    `rai_config` off its latest version. `None` means "inherits the
    account's default content filter", not "unconfigured" -- this read-only
    check can't tell those apart.
    """
    rows = []
    for name in sorted(set(agent_names)):
        details = project_client.agents.get(name)
        rai_config = details.versions.latest.definition.rai_config
        rows.append(
            AgentRaiStatus(
                agent=name,
                rai_policy_name=getattr(rai_config, "rai_policy_name", None),
            )
        )
    return rows


@dataclass(frozen=True)
class LifecycleReport:
    deployments: list[DeploymentHealth]
    agents: list[AgentRaiStatus]

    def has_past_due_deprecation(self) -> bool:
        return any(
            d.days_until_deprecation is not None and d.days_until_deprecation < 0
            for d in self.deployments
        )


def run_lifecycle_check(project_client: Any, credential: Any) -> LifecycleReport:
    """Real, on-demand Azure calls -- deployments + model catalog (management
    plane) and one `agents.get()` per agent (Foundry data plane). Never
    called from a live pipeline run; only from `ds-crew-maf --check-models`.
    """
    deployments = deployment_health(fetch_deployments(credential), fetch_model_catalog(credential))
    agents = agent_rai_policies(project_client, (stage.agent for stage in STAGES))
    return LifecycleReport(deployments=deployments, agents=agents)
