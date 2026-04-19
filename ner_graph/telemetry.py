"""Arize Phoenix tracing setup for LlamaIndex pipeline runs."""

import os

import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register


def _to_bool(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def setup_phoenix_tracing() -> None:
    """
    Configure and start Arize Phoenix + OpenInference tracing.

    Environment variables:
    - PHOENIX_ENABLED: enable tracing ("true"/"1"/"yes")
    - PHOENIX_PROJECT_NAME: trace project name in Phoenix
    - PHOENIX_COLLECTOR_ENDPOINT: optional OTLP endpoint (remote Phoenix)
    - PHOENIX_API_KEY: optional API key for remote collector
    - PHOENIX_LAUNCH_LOCAL: launch local Phoenix UI process when enabled
    - PHOENIX_HOST / PHOENIX_PORT: optional local host/port for launch_app
    """
    if not _to_bool(os.getenv("PHOENIX_ENABLED")):
        print("[telemetry] Phoenix tracing disabled")
        return

    project_name = os.getenv("PHOENIX_PROJECT_NAME", "ner-graph-rag")
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    api_key = os.getenv("PHOENIX_API_KEY")

    # Keep local launch opt-in to avoid unicode-print issues in some Windows terminals.
    if _to_bool(os.getenv("PHOENIX_LAUNCH_LOCAL", "false")):
        host = os.getenv("PHOENIX_HOST")
        port_raw = os.getenv("PHOENIX_PORT")
        port = int(port_raw) if port_raw is not None and port_raw.strip() != "" else None
        print(f"[telemetry] Launching Phoenix UI host={host!r} port={port!r}")
        try:
            px.launch_app(host=host, port=port, run_in_thread=True)
        except RuntimeError as exc:
            print(
                "[telemetry] Phoenix UI launch failed; continuing without local UI. "
                f"error={exc}"
            )

    headers: dict[str, str] | None = None
    if api_key is not None and api_key.strip() != "":
        headers = {"api_key": api_key.strip()}

    try:
        tracer_provider = register(
            endpoint=collector_endpoint,
            project_name=project_name,
            batch=True,
            headers=headers,
            verbose=True,
            auto_instrument=False,
            api_key=api_key,
        )
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:
        print(
            "[telemetry] Phoenix tracing setup failed; continuing without tracing. "
            f"error={exc}"
        )
        return

    print(
        f"[telemetry] Phoenix tracing enabled project={project_name!r} "
        f"endpoint={collector_endpoint!r}"
    )
