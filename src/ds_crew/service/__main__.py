"""Entrypoint for the HTTP tool service: `ds-crew-service`.

This is the tool layer's only caller on this branch: it serves the tools in
`ds_crew.tools` to whatever drives them -- `ds_crew.foundry` over HTTP/MCP,
or a Foundry agent directly.
"""

from __future__ import annotations

import argparse
import sys

from ds_crew import settings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ds-crew-service",
        description="Serve the DS-Crew tool layer over HTTP (OpenAPI at /openapi.json).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--reload", action="store_true", help="Reload on code changes (development only)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not settings.SERVICE_API_KEY:
        # Refuse at startup rather than serving an endpoint whose every route
        # 503s, which looks like an outage instead of a missing setting.
        print(
            "SERVICE_API_KEY is not set. The tool service mutates datasets and trains "
            "models, so it will not start without one. Set it in .env.",
            file=sys.stderr,
        )
        return 1

    # Imported here, not at module scope: the `service` extra is optional, and a
    # missing uvicorn should surface as this message rather than an ImportError
    # traceback from `ds-crew-service --help`.
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install the service extra: uv sync --extra service",
            file=sys.stderr,
        )
        return 1

    # Single worker deliberately. RunState holds DataFrames and fitted models in
    # this process's memory (see state.DataStore) and run tokens live alongside
    # them, so a second worker would serve requests that cannot see either.
    # Scaling out means externalizing that state first.
    uvicorn.run(
        "ds_crew.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
