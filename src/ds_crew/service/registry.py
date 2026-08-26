"""The set of tools the HTTP service exposes, derived from the tool classes themselves.

Nothing here restates a tool's name, argument schema or description: all three are
read off the `Tool` subclass itself, so the HTTP surface and the OpenAPI spec
generated from it cannot drift from what a caller actually invokes. Adding
a tool to `TOOL_CLASSES` is the only step needed to publish it.
"""

from __future__ import annotations

from pydantic import BaseModel

from ds_crew.tools.base import Tool

from ds_crew.tools.cleaning_tools import ApplyCleaningPlanTool
from ds_crew.tools.eda_tools import EdaSummaryTool
from ds_crew.tools.ensemble_tools import EnsembleModelsTool
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.explain_tools import ExplainModelsTool
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool
from ds_crew.tools.hpo_tools import TuneModelsTool
from ds_crew.tools.logging_tools import FinalizeRunTool
from ds_crew.tools.model_tools import SetMetricTool, TrainCandidateModelsTool

# Ordered as the pipeline runs them. The service does not enforce this ordering
# -- the tools already refuse out-of-order invocation themselves (e.g.
# ExplainModelsTool requires evaluation_applied, and the *_applied flags reject a
# second application) -- but listing them in pipeline order makes the generated
# OpenAPI spec readable to whoever is wiring an agent against it.
TOOL_CLASSES: tuple[type[Tool], ...] = (
    EdaSummaryTool,
    ApplyCleaningPlanTool,
    ApplyFeaturePlanTool,
    SetMetricTool,
    TrainCandidateModelsTool,
    TuneModelsTool,
    EnsembleModelsTool,
    EvaluateModelsTool,
    ExplainModelsTool,
    FinalizeRunTool,
)


def tool_name_of(tool_cls: type[Tool]) -> str:
    """The tool's LLM-facing name, read off the class rather than duplicated here.

    `name` is a Pydantic field with a default on every tool class, so the default
    is the value an instance would carry; reading it avoids instantiating a tool
    (which would need a run_id) purely to learn what it is called.
    """
    return tool_cls.model_fields["name"].default


def args_schema_of(tool_cls: type[Tool]) -> type[BaseModel]:
    """The tool's argument schema, which becomes the HTTP request body model."""
    return tool_cls.model_fields["args_schema"].default


def description_of(tool_cls: type[Tool]) -> str:
    return tool_cls.model_fields["description"].default


TOOLS_BY_NAME: dict[str, type[Tool]] = {tool_name_of(c): c for c in TOOL_CLASSES}
