"""Tests for the tracing module."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from promptic_sdk.tracing import (
    PROMPTIC_COMPONENT_ATTR,
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
        with patch("promptic_sdk.tracing.OTLPSpanExporter") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init()

        mock_exporter.assert_called_once()
        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://promptic.eu/api/v1/traces"
        assert call_kwargs["headers"]["Authorization"] == "Bearer pk_test_key"

    def test_init_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with patch("promptic_sdk.tracing.OTLPSpanExporter") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init(endpoint="https://custom.example.com")

        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://custom.example.com/api/v1/traces"

    def test_init_sets_tracer_provider(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with patch(
            "promptic_sdk.tracing.OTLPSpanExporter",
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
                "promptic_sdk.tracing.OTLPSpanExporter",
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
                "promptic_sdk.tracing.OTLPSpanExporter",
                return_value=MagicMock(),
            ),
            patch("promptic_sdk.tracing._auto_instrument") as mock_auto,
        ):
            init(auto_instrument=True)

        mock_auto.assert_called_once()

    def test_init_endpoint_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.setenv("PROMPTIC_ENDPOINT", "https://env.example.com")
        with patch("promptic_sdk.tracing.OTLPSpanExporter") as mock_exporter:
            mock_exporter.return_value = MagicMock()
            init()

        call_kwargs = mock_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://env.example.com/api/v1/traces"


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
