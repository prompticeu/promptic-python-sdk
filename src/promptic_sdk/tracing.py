"""Convenience wrapper for configuring OpenTelemetry to send traces to Promptic."""

from __future__ import annotations

import contextvars
import os
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_DEFAULT_ENDPOINT = "https://app.promptic.eu"

PROMPTIC_COMPONENT_ATTR = "promptic.ai_component"

# Context variable that holds the current AI component name.
_current_component: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_ai_component", default=None
)

# Instrumentors that we try to auto-detect and enable.
# Each entry: (module_path, class_name)
_INSTRUMENTORS: list[tuple[str, str]] = [
    ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
    ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
    ("opentelemetry.instrumentation.google_generativeai", "GoogleGenerativeAiInstrumentor"),
    ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
    ("opentelemetry.instrumentation.cohere", "CohereInstrumentor"),
]


class _ComponentAttributeProcessor(SpanProcessor):
    """Add ``promptic.ai_component`` attribute to spans inside :func:`ai_component`."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        name = _current_component.get()
        if name is not None:
            span.set_attribute(PROMPTIC_COMPONENT_ATTR, name)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D102
        pass

    def shutdown(self) -> None:  # noqa: D102
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D102
        return True


def init(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    auto_instrument: bool = True,
    service_name: str | None = None,
) -> None:
    """Configure OpenTelemetry to send traces to Promptic.

    Args:
        api_key: Promptic API key. Falls back to ``PROMPTIC_API_KEY`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://app.promptic.eu``.
        auto_instrument: If True, auto-detect installed LLM client libraries and
            instrument them.
        service_name: OpenTelemetry ``service.name`` resource attribute.
    """
    api_key = api_key or os.environ.get("PROMPTIC_API_KEY")
    if not api_key:
        msg = (
            "Promptic API key is required. "
            "Pass api_key= or set the PROMPTIC_API_KEY environment variable."
        )
        raise ValueError(msg)

    endpoint = endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
    traces_endpoint = f"{endpoint.rstrip('/')}/api/v1/traces"

    exporter = OTLPSpanExporter(
        endpoint=traces_endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    resource_attrs = {}
    if service_name:
        resource_attrs["service.name"] = service_name

    provider = TracerProvider(
        resource=Resource.create(resource_attrs) if resource_attrs else Resource.create(),
    )
    provider.add_span_processor(_ComponentAttributeProcessor())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if auto_instrument:
        _auto_instrument()


@contextmanager
def ai_component(name: str) -> Iterator[None]:
    """Tag all spans created within this context with an AI Component name.

    The server matches the name against AI Components in the workspace
    and links traces accordingly.

    Example::

        with promptic_sdk.ai_component("customer-support-agent"):
            response = openai_client.chat.completions.create(...)
    """
    token = _current_component.set(name)
    try:
        yield
    finally:
        _current_component.reset(token)


def _auto_instrument() -> None:
    """Try to import and enable each known instrumentor."""
    import importlib

    for module_path, class_name in _INSTRUMENTORS:
        try:
            mod = importlib.import_module(module_path)
            instrumentor_cls = getattr(mod, class_name)
            instrumentor_cls().instrument()
        except (ImportError, AttributeError):
            # Instrumentor package not installed — skip silently
            continue
