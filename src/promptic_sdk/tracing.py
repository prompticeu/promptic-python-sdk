"""Convenience wrapper for configuring OpenTelemetry to send traces to Promptic."""

from __future__ import annotations

import atexit
import contextvars
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger("promptic_sdk")

_DEFAULT_ENDPOINT = "https://promptic.eu"

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


class _LoggingExporter(SpanExporter):
    """Wraps an exporter to log failures instead of silently dropping spans."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner
        self._warned = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = self._inner.export(spans)
        if result != SpanExportResult.SUCCESS:
            if not self._warned:
                logger.warning(
                    "Promptic: failed to export %d span(s). "
                    "Check your API key and endpoint. "
                    "(Further export errors will be logged at DEBUG level.)",
                    len(spans),
                )
                self._warned = True
            else:
                logger.debug("Promptic: failed to export %d span(s).", len(spans))
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


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
            then to ``https://promptic.eu``.
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

    exporter = _LoggingExporter(
        OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
        )
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

    # Ensure all spans are flushed when the process exits.
    atexit.register(provider.shutdown)

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
    import importlib.util

    for module_path, class_name in _INSTRUMENTORS:
        try:
            mod = importlib.import_module(module_path)
            instrumentor_cls = getattr(mod, class_name)
            instrumentor_cls().instrument()
            logger.debug("Promptic: enabled %s.%s", module_path, class_name)
        except ImportError:
            # Distinguish "package not installed" from "package broken internally".
            # If the top-level package can be found on sys.path, the ImportError
            # is an internal compatibility issue that the user should know about.
            top_level = module_path.rsplit(".", 1)[0]
            if importlib.util.find_spec(top_level) is not None:
                logger.warning(
                    "Promptic: %s is installed but failed to import — "
                    "it may be incompatible with your current dependencies. "
                    "Try upgrading or pinning a compatible version.",
                    module_path,
                    exc_info=True,
                )
            # else: package genuinely not installed — skip silently
        except Exception:
            logger.warning(
                "Promptic: failed to enable %s.%s — the package may be incompatible.",
                module_path,
                class_name,
                exc_info=True,
            )
