"""Base class for this project's deterministic tools.

Replaces `crewai.tools.BaseTool`. Every tool here is invoked exactly one way,
by `service/app.py` and `service/mcp_app.py`: construct `tool_cls(run_id=...)`
then call `._run(**kwargs)`. `service/registry.py` reads `name`, `description`
and `args_schema` off the class itself via `model_fields[...].default`,
without instantiating. That is the entire surface a base class needs to
provide -- crewai's version additionally carries async wrapping, checkpoint
serialization and LLM-facing schema formatting that nothing on this branch
ever calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class Tool(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def _run(self, **kwargs: object) -> str: ...
