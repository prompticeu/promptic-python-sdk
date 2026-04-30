"""Convenience wrapper for configuring OpenTelemetry to send traces to Promptic."""

from __future__ import annotations

import atexit
import contextvars
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger("promptic_sdk")

_DEFAULT_ENDPOINT = "https://promptic.eu"

PROMPTIC_COMPONENT_ATTR = "promptic.ai_component"
PROMPTIC_DATASET_ATTR = "promptic.dataset"
PROMPTIC_RUN_ATTR = "promptic.run"

# Context variable that holds the current AI component name.
_current_component: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_ai_component", default=None
)

# Context variable that holds the current dataset name.
_current_dataset: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_dataset", default=None
)

# Context variable that holds the current run name.
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_run", default=None
)

# Instrumentors that we try to auto-detect and enable.
# Each entry: (module_path, class_name)
#
# The first three cover direct LLM SDK calls (works across any framework).
# The rest cover framework-level instrumentation; all emit OTel-official
# ``gen_ai.*`` semantic conventions that the backend parser handles uniformly.
#
# Pydantic AI is intentionally absent — it ships its own OTel emitter; users
# opt in by constructing the Agent with ``instrument=True``.
_INSTRUMENTORS: list[tuple[str, str]] = [
    # LLM providers (direct SDK calls) — emit gen_ai.* on every chat/completion.
    ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
    ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
    ("opentelemetry.instrumentation.google_generativeai", "GoogleGenerativeAiInstrumentor"),
    ("opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor"),
    ("opentelemetry.instrumentation.bedrock", "BedrockInstrumentor"),
    ("opentelemetry.instrumentation.mistralai", "MistralAiInstrumentor"),
    ("opentelemetry.instrumentation.cohere", "CohereInstrumentor"),
    # Agent frameworks — emit chain/tool/llm spans with the full graph structure.
    ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
    ("opentelemetry.instrumentation.openai_agents", "OpenAIAgentsInstrumentor"),
    ("opentelemetry.instrumentation.claude_agent_sdk", "ClaudeAgentSdkInstrumentor"),
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


class _BodyTooLargeError(Exception):
    """Raised when the OTLP server rejects a batch with HTTP 413."""


class _OTLPSpanExporter413Aware(OTLPSpanExporter):
    """``OTLPSpanExporter`` that surfaces 413 responses as a typed exception.

    The base class only returns ``SpanExportResult.SUCCESS`` / ``FAILURE`` and
    swallows the HTTP status code. We need to distinguish "batch too big"
    (recoverable by bisecting) from every other failure (not recoverable
    that way), so we override ``_export`` to inspect the response and raise
    ``_BodyTooLargeError`` on 413. The exception propagates past the parent's
    ``RequestException`` handler and reaches our :class:`_BisectingExporter`.
    """

    def _export(self, serialized_data: bytes, timeout_sec: float | None = None):
        resp = super()._export(serialized_data, timeout_sec)
        if resp.status_code == 413:
            raise _BodyTooLargeError(
                f"OTLP server rejected payload of {len(serialized_data)} bytes "
                f"with HTTP 413 (Request Entity Too Large)"
            )
        return resp


class _BisectingExporter(SpanExporter):
    """Wraps an exporter so that oversized batches are halved and retried.

    The wrapped exporter is expected to raise :class:`_BodyTooLargeError`
    when the server rejects a batch with HTTP 413. This wrapper recursively
    bisects the batch until either each half fits or only one span is left
    (single span over the limit → drop it and log; one span can't be split).

    Other failure modes (auth errors, network errors, generic 5xx) are not
    retried here — the inner exporter's own retry policy handles those, and
    bisecting wouldn't help anyway.
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            return self._inner.export(spans)
        except _BodyTooLargeError as exc:
            if len(spans) <= 1:
                logger.error(
                    "Promptic: dropping a single oversized span — %s. "
                    "Reduce per-span attribute sizes (e.g. truncate large "
                    "tool inputs/outputs).",
                    exc,
                )
                return SpanExportResult.FAILURE
            mid = len(spans) // 2
            logger.debug(
                "Promptic: OTLP batch of %d spans exceeded server body limit; "
                "bisecting and retrying.",
                len(spans),
            )
            left = self.export(spans[:mid])
            right = self.export(spans[mid:])
            if left == SpanExportResult.SUCCESS and right == SpanExportResult.SUCCESS:
                return SpanExportResult.SUCCESS
            return SpanExportResult.FAILURE

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
        ds = _current_dataset.get()
        if ds is not None:
            span.set_attribute(PROMPTIC_DATASET_ATTR, ds)
        run = _current_run.get()
        if run is not None:
            span.set_attribute(PROMPTIC_RUN_ATTR, run)

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

    # Layered exporter:
    #   _LoggingExporter      → emits a one-time WARNING on the first failure
    #   _BisectingExporter    → on HTTP 413, halves the batch and retries
    #   _OTLPSpanExporter413Aware → raises _BodyTooLargeError for 413 so the
    #                                bisecter sees it (instead of the parent
    #                                swallowing it as a generic FAILURE)
    #
    # With this stack, oversized batches recover transparently. We keep
    # OTel's default `max_export_batch_size` (512) because the bisecter
    # makes overflow free.
    exporter = _LoggingExporter(
        _BisectingExporter(
            _OTLPSpanExporter413Aware(
                endpoint=traces_endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
            )
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


def _langsmith_tracing_context(
    component: str,
    dataset: str | None,
    run: str | None,
) -> AbstractContextManager | None:
    """Return a ``langsmith.tracing_context`` if available, else ``None``.

    Injects Promptic attributes into LangSmith run metadata so they appear
    as span attributes when the LangSmith OTel exporter converts runs to
    OTel spans.  Without this, LangSmith-created spans would lack the
    ``promptic.ai_component`` / ``promptic.dataset`` / ``promptic.run``
    attributes needed to link traces to AI components.
    """
    if os.environ.get("LANGSMITH_OTEL_ENABLED", "").lower() != "true":
        return None
    try:
        from langsmith import tracing_context
    except ImportError:
        return None

    metadata: dict[str, str] = {PROMPTIC_COMPONENT_ATTR: component}
    if dataset:
        metadata[PROMPTIC_DATASET_ATTR] = dataset
    if run:
        metadata[PROMPTIC_RUN_ATTR] = run
    return tracing_context(metadata=metadata)


@contextmanager
def ai_component(
    name: str,
    *,
    dataset: str | None = None,
    run: str | None = None,
) -> Iterator[None]:
    """Tag all spans created within this context with an AI Component name.

    The server matches the name against AI Components in the workspace
    and links traces accordingly.

    Args:
        name: AI Component name in the workspace.
        dataset: Optional dataset name. When set, traces are automatically
            added to the named dataset (created if it doesn't exist).
        run: Optional run name. When set alongside ``dataset``, traces are
            grouped into a named run within the dataset. Each unique run name
            creates a separate run, allowing you to compare different
            executions against the same dataset.

    Example::

        with promptic_sdk.ai_component("customer-support-agent"):
            response = openai_client.chat.completions.create(...)

        # With dataset and run tagging:
        with promptic_sdk.ai_component("my-agent", dataset="eval-set", run="v1-baseline"):
            agent.run(test_input)
    """
    comp_token = _current_component.set(name)
    ds_token = _current_dataset.set(dataset) if dataset else None
    run_token = _current_run.set(run) if run else None

    # When LangSmith OTel bridge is active, inject Promptic attributes into
    # LangSmith run metadata so they appear as span attributes after export.
    langsmith_ctx = _langsmith_tracing_context(name, dataset, run)

    try:
        if langsmith_ctx is not None:
            langsmith_ctx.__enter__()
        yield
    finally:
        if langsmith_ctx is not None:
            langsmith_ctx.__exit__(None, None, None)
        _current_component.reset(comp_token)
        if ds_token is not None:
            _current_dataset.reset(ds_token)
        if run_token is not None:
            _current_run.reset(run_token)


@contextmanager
def dataset(name: str) -> Iterator[None]:
    """Tag all spans created within this context with a dataset name.

    Traces with a dataset attribute are automatically added to the named
    dataset during OTLP ingestion (the dataset is created if it doesn't exist).

    Can be nested inside :func:`ai_component` for composability::

        with promptic_sdk.ai_component("my-agent"):
            with promptic_sdk.dataset("eval-round-1"):
                agent.run(test_input)
    """
    token = _current_dataset.set(name)
    try:
        yield
    finally:
        _current_dataset.reset(token)


def _auto_instrument() -> None:
    """Try to import and enable each known instrumentor.

    OpenLLMetry's instrumentors are the primary path. They emit OTel-official
    ``gen_ai.*`` semantic conventions (including ``gen_ai.tool.definitions``)
    and cover LangGraph / deepagents correctly as of
    ``opentelemetry-instrumentation-langchain>=0.60.0``.

    If the user manually sets ``LANGSMITH_OTEL_ENABLED=true`` in their env,
    that path is also supported — both sources coexist, and the backend
    de-duplicates/recognizes both. We do not auto-enable LangSmith anymore.
    """
    import importlib
    import importlib.util

    # Warn when LangSmith tracing is active without its OTel bridge.  LangSmith
    # installs a LangChain callback handler that intercepts model calls before
    # OpenLLMetry's LangchainInstrumentor can hook them — resulting in missing
    # ChatOpenAI.chat spans and silently broken agent-evaluation insights.
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    langsmith_otel = os.environ.get("LANGSMITH_OTEL_ENABLED", "").lower() == "true"
    if langsmith_tracing and not langsmith_otel:
        logger.warning(
            "Promptic: LANGSMITH_TRACING=true is set without LANGSMITH_OTEL_ENABLED=true. "
            "LangSmith's callback handler will intercept LangChain/LangGraph runs before "
            "OpenLLMetry can instrument them, so ChatOpenAI spans and tool definitions "
            "may be missing from your Promptic traces. "
            "Either unset LANGSMITH_TRACING, or set LANGSMITH_OTEL_ENABLED=true to "
            "route LangSmith spans through OTel into Promptic."
        )

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
            try:
                is_installed = importlib.util.find_spec(module_path) is not None
            except (ModuleNotFoundError, ValueError):
                is_installed = False
            if is_installed:
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
