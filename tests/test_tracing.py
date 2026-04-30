"""Tests for the tracing module."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from promptic_sdk.tracing import (
    PROMPTIC_COMPONENT_ATTR,
    _BisectingExporter,
    _BodyTooLargeError,
    _ComponentAttributeProcessor,
    _current_component,
    ai_component,
    init,
)


def _reset_tracer_provider():
    """Reset the global tracer provider so each test starts fresh."""
    # OTel guards against re-setting, so we need to reset the internal flag
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # noqa: SLF001
    trace._TRACER_PROVIDER = None  # noqa: SLF001


class TestInit:
    def teardown_method(self):
        """Reset global tracer provider after each test."""
        _reset_tracer_provider()

    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            init(api_key=None)

    def test_init_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test_key")
        with patch("promptic_sdk.tracing._OTLPSpanExporter413Aware") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init()

        mock_exporter.assert_called_once()
        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://promptic.eu/api/v1/traces"
        assert call_kwargs["headers"]["Authorization"] == "Bearer pk_test_key"

    def test_init_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with patch("promptic_sdk.tracing._OTLPSpanExporter413Aware") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init(endpoint="https://custom.example.com")

        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://custom.example.com/api/v1/traces"

    def test_init_sets_tracer_provider(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with patch(
            "promptic_sdk.tracing._OTLPSpanExporter413Aware",
            return_value=MagicMock(),
        ):
            init(auto_instrument=False)

        provider = trace.get_tracer_provider()
        assert provider is not None
        assert isinstance(provider, TracerProvider)

    def test_init_skips_auto_instrument_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with (
            patch(
                "promptic_sdk.tracing._OTLPSpanExporter413Aware",
                return_value=MagicMock(),
            ),
            patch("promptic_sdk.tracing._auto_instrument") as mock_auto,
        ):
            init(auto_instrument=False)

        mock_auto.assert_not_called()

    def test_init_calls_auto_instrument_when_enabled(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with (
            patch(
                "promptic_sdk.tracing._OTLPSpanExporter413Aware",
                return_value=MagicMock(),
            ),
            patch("promptic_sdk.tracing._auto_instrument") as mock_auto,
        ):
            init(auto_instrument=True)

        mock_auto.assert_called_once()

    def test_init_endpoint_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.setenv("PROMPTIC_ENDPOINT", "https://env.example.com")
        with patch("promptic_sdk.tracing._OTLPSpanExporter413Aware") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init()

        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://env.example.com/api/v1/traces"

    def test_init_wires_bisecting_exporter_chain(self, monkeypatch):
        """init() should wrap the OTLP exporter in our bisecting + logging chain."""
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with patch(
            "promptic_sdk.tracing._OTLPSpanExporter413Aware",
            return_value=MagicMock(),
        ):
            init(auto_instrument=False)

        provider = trace.get_tracer_provider()
        # The active provider should have our component processor + a
        # BatchSpanProcessor whose underlying exporter is the layered chain.
        # Smoke check via the public API: a tracer can be created and used.
        tracer = provider.get_tracer("promptic_sdk.test")
        with tracer.start_as_current_span("smoke"):
            pass


class TestBisectingExporter:
    """The bisecting exporter halves and retries on 413."""

    def _fake_inner(self, max_spans_per_request: int):
        """Fake inner exporter: succeeds if batch fits, raises 413 otherwise."""
        calls: list[int] = []

        class _Inner(SpanExporter):
            def export(self, spans):
                calls.append(len(spans))
                if len(spans) > max_spans_per_request:
                    raise _BodyTooLargeError(f"too many: {len(spans)}")
                return SpanExportResult.SUCCESS

            def shutdown(self) -> None:
                pass

            def force_flush(self, timeout_millis: int = 30000) -> bool:
                return True

        return _Inner(), calls

    def test_passthrough_when_inner_succeeds(self):
        inner, calls = self._fake_inner(max_spans_per_request=100)
        bisecter = _BisectingExporter(inner)
        spans = [MagicMock() for _ in range(50)]

        result = bisecter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert calls == [50]  # one call, no bisection

    def test_bisects_once_on_413(self):
        # Inner accepts batches of ≤4. With 8 spans, we expect: 8 → fails;
        # 4 + 4 → both succeed.
        inner, calls = self._fake_inner(max_spans_per_request=4)
        bisecter = _BisectingExporter(inner)
        spans = [MagicMock() for _ in range(8)]

        result = bisecter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert calls == [8, 4, 4]

    def test_recursively_bisects_until_each_half_fits(self):
        # Inner accepts only 1 span per request. With 8 spans we expect a
        # binary-tree of bisections until everything is single spans.
        inner, calls = self._fake_inner(max_spans_per_request=1)
        bisecter = _BisectingExporter(inner)
        spans = [MagicMock() for _ in range(8)]

        result = bisecter.export(spans)

        assert result == SpanExportResult.SUCCESS
        # All 8 single-span exports plus the failing intermediates.
        single_span_calls = [n for n in calls if n == 1]
        assert len(single_span_calls) == 8

    def test_drops_single_oversized_span(self):
        # Inner rejects everything with 413 (even 1 span is too big).
        inner, calls = self._fake_inner(max_spans_per_request=0)
        bisecter = _BisectingExporter(inner)
        spans = [MagicMock()]

        result = bisecter.export(spans)

        assert result == SpanExportResult.FAILURE
        assert calls == [1]  # tried once, gave up (can't split a single span)

    def test_other_exceptions_propagate(self):
        """Non-413 errors should not be swallowed by the bisecter."""

        class _Inner(SpanExporter):
            def export(self, spans):
                raise RuntimeError("auth broken")

            def shutdown(self) -> None:
                pass

            def force_flush(self, timeout_millis: int = 30000) -> bool:
                return True

        bisecter = _BisectingExporter(_Inner())
        with pytest.raises(RuntimeError, match="auth broken"):
            bisecter.export([MagicMock()])


class TestAiComponent:
    """Tests for the ai_component() context manager and _ComponentAttributeProcessor."""

    def setup_method(self):
        """Set up a fresh TracerProvider with the component processor and an in-memory exporter."""
        _reset_tracer_provider()
        self.exported_spans: list = []
        self.provider = TracerProvider()
        self.provider.add_span_processor(_ComponentAttributeProcessor())
        self.provider.add_span_processor(
            SimpleSpanProcessor(_InMemoryExporter(self.exported_spans))
        )
        trace.set_tracer_provider(self.provider)

    def teardown_method(self):
        _reset_tracer_provider()

    def test_sets_attribute_inside_context(self):
        tracer = trace.get_tracer("test")
        with ai_component("my-agent"), tracer.start_as_current_span("test-span"):
            pass

        assert len(self.exported_spans) == 1
        attrs = dict(self.exported_spans[0].attributes)
        assert attrs[PROMPTIC_COMPONENT_ATTR] == "my-agent"

    def test_no_attribute_outside_context(self):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-span"):
            pass

        assert len(self.exported_spans) == 1
        attrs = dict(self.exported_spans[0].attributes)
        assert PROMPTIC_COMPONENT_ATTR not in attrs

    def test_nested_contexts_use_innermost(self):
        tracer = trace.get_tracer("test")
        with ai_component("outer"):
            with tracer.start_as_current_span("outer-span"):
                pass
            with ai_component("inner"), tracer.start_as_current_span("inner-span"):
                pass
            # After inner context exits, should revert to outer
            with tracer.start_as_current_span("back-to-outer"):
                pass

        assert len(self.exported_spans) == 3
        assert dict(self.exported_spans[0].attributes)[PROMPTIC_COMPONENT_ATTR] == "outer"
        assert dict(self.exported_spans[1].attributes)[PROMPTIC_COMPONENT_ATTR] == "inner"
        assert dict(self.exported_spans[2].attributes)[PROMPTIC_COMPONENT_ATTR] == "outer"

    def test_context_var_reset_after_exit(self):
        with ai_component("temp"):
            assert _current_component.get() == "temp"
        assert _current_component.get() is None


class _InMemoryExporter(SpanExporter):
    """Simple span exporter that collects spans in a list for testing."""

    def __init__(self, spans_list: list):
        self._spans = spans_list

    def export(self, spans):
        self._spans.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True
