"""Tests for the tracing module."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import promptic_sdk.tracing as tracing_module
from promptic_sdk.tracing import (
    PROMPTIC_COMPONENT_ATTR,
    ArtifactReference,
    _ArtifactSanitizer,
    _ArtifactUploader,
    _auto_instrument,
    _BisectingExporter,
    _BodyTooLargeError,
    _ComponentAttributeProcessor,
    _current_component,
    _OTLPSpanExporter413Aware,
    ai_component,
    artifact,
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
        tracing_module._configured_api_key = None  # noqa: SLF001
        tracing_module._configured_endpoint = None  # noqa: SLF001

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

    def test_init_rewrites_artifacts_before_bisecting(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        otlp_exporter = MagicMock(spec=SpanExporter)
        bisecting_exporter = MagicMock(spec=SpanExporter)
        artifact_exporter = MagicMock(spec=SpanExporter)
        logging_exporter = MagicMock(spec=SpanExporter)

        with (
            patch(
                "promptic_sdk.tracing._OTLPSpanExporter413Aware",
                return_value=otlp_exporter,
            ),
            patch(
                "promptic_sdk.tracing._BisectingExporter", return_value=bisecting_exporter
            ) as mock_bisecting,
            patch(
                "promptic_sdk.tracing._ArtifactRewritingExporter",
                return_value=artifact_exporter,
            ) as mock_artifacts,
            patch("promptic_sdk.tracing._LoggingExporter", return_value=logging_exporter),
        ):
            init(auto_instrument=False)

        mock_bisecting.assert_called_once_with(otlp_exporter)
        mock_artifacts.assert_called_once_with(
            bisecting_exporter,
            endpoint="https://promptic.eu",
            api_key="pk_test",
        )

    def test_repeated_init_does_not_replace_artifact_credentials(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_first")
        with patch(
            "promptic_sdk.tracing._OTLPSpanExporter413Aware",
            return_value=MagicMock(),
        ):
            init(endpoint="https://first.example", auto_instrument=False)

        init(api_key="pk_second", endpoint="https://second.example", auto_instrument=False)

        assert tracing_module._configured_api_key == "pk_first"  # noqa: SLF001
        assert tracing_module._configured_endpoint == "https://first.example"  # noqa: SLF001

    def test_init_preserves_artifact_credentials_with_existing_provider(self):
        trace.set_tracer_provider(TracerProvider())

        init(
            api_key="pk_external",
            endpoint="https://external.example",
            auto_instrument=False,
        )

        assert tracing_module._configured_api_key == "pk_external"  # noqa: SLF001
        assert tracing_module._configured_endpoint == "https://external.example"  # noqa: SLF001

    def test_init_auto_instruments_with_existing_provider(self):
        trace.set_tracer_provider(TracerProvider())

        with patch("promptic_sdk.tracing._auto_instrument") as mock_auto:
            init(api_key="pk_external", auto_instrument=True)

        mock_auto.assert_called_once()


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


class TestOTLPSpanExporter413Aware:
    def test_export_supports_parent_export_without_timeout_parameter(self, monkeypatch):
        calls: list[tuple[bytes, object]] = []

        def parent_export(self, serialized_data: bytes):
            calls.append((serialized_data, None))
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(OTLPSpanExporter, "_export", parent_export)

        exporter = _OTLPSpanExporter413Aware(endpoint="http://example.com/v1/traces")
        response = exporter._export(b"payload", timeout_sec=1.0)

        assert response.status_code == 200
        assert calls == [(b"payload", None)]

    def test_export_supports_parent_export_with_timeout_parameter(self, monkeypatch):
        calls: list[tuple[bytes, object]] = []

        def parent_export(self, serialized_data: bytes, timeout_sec: float | None = None):
            calls.append((serialized_data, timeout_sec))
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(OTLPSpanExporter, "_export", parent_export)

        exporter = _OTLPSpanExporter413Aware(endpoint="http://example.com/v1/traces")
        response = exporter._export(b"payload", timeout_sec=1.0)

        assert response.status_code == 200
        assert calls == [(b"payload", 1.0)]

    def test_export_raises_typed_error_on_413(self, monkeypatch):
        def parent_export(self, serialized_data: bytes):
            return SimpleNamespace(status_code=413)

        monkeypatch.setattr(OTLPSpanExporter, "_export", parent_export)

        exporter = _OTLPSpanExporter413Aware(endpoint="http://example.com/v1/traces")
        with pytest.raises(_BodyTooLargeError, match="HTTP 413"):
            exporter._export(b"payload")


class TestAutoInstrument:
    def test_openai_instrumentor_disables_default_image_uploader(self, monkeypatch):
        calls: list[object] = []

        class FakeOpenAIInstrumentor:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def instrument(self):
                calls.append("instrumented")

        monkeypatch.setattr(
            "promptic_sdk.tracing._INSTRUMENTORS",
            [("opentelemetry.instrumentation.openai", "OpenAIInstrumentor")],
        )

        with patch(
            "importlib.import_module",
            return_value=SimpleNamespace(OpenAIInstrumentor=FakeOpenAIInstrumentor),
        ):
            _auto_instrument()

        assert calls == [{"upload_base64_image": None}, "instrumented"]


class TestArtifactSanitizer:
    def test_rewrites_json_data_uri_into_artifact_reference(self):
        uploads: list[tuple[bytes, str, str]] = []

        class FakeUploader:
            def upload(self, content, *, mime_type, source_path="$", preview=None):
                uploads.append((content, mime_type, source_path))
                return ArtifactReference(
                    id="artifact-id",
                    uri="promptic-artifact://artifact-id",
                    mime_type=mime_type,
                    size_bytes=len(content),
                    sha256="abc123",
                )

        sanitizer = _ArtifactSanitizer(FakeUploader())
        encoded = "aGVsbG8gaGVsbG8gaGVsbG8="
        value = (
            '[{"role":"user","content":[{"type":"image_url","image_url":'
            f'{{"url":"data:image/png;base64,{encoded}"}}}}]}}]'
        )

        sanitized = sanitizer.sanitize_attribute("gen_ai.input.messages", value)

        assert uploads == [
            (
                b"hello hello hello",
                "image/png",
                "gen_ai.input.messages[0].content[0].image_url.url",
            )
        ]
        assert encoded not in sanitized
        assert "promptic-artifact://artifact-id" in sanitized

    def test_rewrites_openai_blob_base64_with_sibling_mime_type(self):
        uploads: list[tuple[bytes, str, str]] = []

        class FakeUploader:
            def upload(self, content, *, mime_type, source_path="$", preview=None):
                uploads.append((content, mime_type, source_path))
                return ArtifactReference(
                    id="artifact-id",
                    uri="promptic-artifact://artifact-id",
                    mime_type=mime_type,
                    size_bytes=len(content),
                    sha256="abc123",
                )

        sanitizer = _ArtifactSanitizer(FakeUploader())
        content = b"\x89PNG\r\n\x1a\n" + (b"\0" * 9000)
        encoded = base64.b64encode(content).decode("ascii")
        value = json.dumps(
            [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "content": "Read this image."},
                        {
                            "type": "blob",
                            "modality": "image",
                            "mime_type": "image/png",
                            "content": encoded,
                        },
                    ],
                }
            ]
        )

        sanitized = sanitizer.sanitize_attribute("gen_ai.input.messages", value)

        assert uploads == [
            (
                content,
                "image/png",
                "gen_ai.input.messages[0].parts[1].content",
            )
        ]
        assert encoded not in sanitized
        assert "promptic-artifact://artifact-id" in sanitized


class TestArtifactUploader:
    def test_upload_prefers_direct_storage_upload(self):
        class FakeResponse:
            def __init__(self, status_code=200, data=None):
                self.status_code = status_code
                self._data = data or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                self.calls.append(("POST", url, kwargs))
                if url.endswith("/storage-objects/presign"):
                    return FakeResponse(
                        data={
                            "method": "PUT",
                            "uploadUrl": "https://storage.example/upload",
                            "storageObjectId": "storage-object-id",
                            "headers": {"x-ms-blob-type": "BlockBlob"},
                        }
                    )
                if url.endswith("/api/v1/artifacts"):
                    return FakeResponse(
                        data={
                            "id": "artifact-id",
                            "uri": "promptic-artifact://artifact-id",
                            "mimeType": "image/png",
                            "sizeBytes": 5,
                            "sha256": "hash",
                        }
                    )
                raise AssertionError(url)

            def put(self, url, **kwargs):
                self.calls.append(("PUT", url, kwargs))
                return FakeResponse()

        fake_client = FakeClient()
        with patch("promptic_sdk.tracing.httpx.Client", return_value=fake_client):
            ref = _ArtifactUploader(endpoint="https://api.example", api_key="pk").upload(
                b"hello",
                mime_type="image/png",
                source_path="messages[0].image_url.url",
            )

        assert ref is not None
        assert ref.id == "artifact-id"
        assert [call[0] for call in fake_client.calls] == ["POST", "PUT", "POST"]
        assert fake_client.calls[0][2]["json"]["folder"] == "trace-artifacts"
        assert fake_client.calls[1][1] == "https://storage.example/upload"
        assert fake_client.calls[1][2]["content"] == b"hello"
        assert fake_client.calls[2][2]["json"]["storageObjectId"] == "storage-object-id"
        assert "contentBase64" not in fake_client.calls[2][2]["json"]

    def test_direct_upload_uses_windows_basename_for_source_path(self):
        class FakeResponse:
            def __init__(self, data=None):
                self.status_code = 200
                self._data = data or {}

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                self.calls.append(("POST", url, kwargs))
                if url.endswith("/storage-objects/presign"):
                    return FakeResponse(
                        data={
                            "method": "PUT",
                            "uploadUrl": "https://storage.example/upload",
                            "storageObjectId": "storage-object-id",
                        }
                    )
                return FakeResponse(
                    data={
                        "id": "artifact-id",
                        "uri": "promptic-artifact://artifact-id",
                        "mimeType": "application/pdf",
                        "sizeBytes": 5,
                        "sha256": "hash",
                    }
                )

            def put(self, url, **kwargs):
                self.calls.append(("PUT", url, kwargs))
                return FakeResponse()

        fake_client = FakeClient()
        with patch("promptic_sdk.tracing.httpx.Client", return_value=fake_client):
            _ArtifactUploader(endpoint="https://api.example", api_key="pk").upload(
                b"hello",
                mime_type="application/pdf",
                source_path=r"C:\tmp\report.pdf",
            )

        assert fake_client.calls[0][2]["json"]["filename"] == "report.pdf"

    def test_upload_falls_back_to_server_upload_when_presign_is_missing(self):
        class FakeResponse:
            def __init__(self, status_code=200, data=None):
                self.status_code = status_code
                self._data = data or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                self.calls.append(("POST", url, kwargs))
                if url.endswith("/storage-objects/presign"):
                    return FakeResponse(status_code=404)
                return FakeResponse(
                    data={
                        "id": "artifact-id",
                        "uri": "promptic-artifact://artifact-id",
                        "mimeType": "text/plain",
                        "sizeBytes": 5,
                        "sha256": "hash",
                    }
                )

        fake_client = FakeClient()
        with patch("promptic_sdk.tracing.httpx.Client", return_value=fake_client):
            ref = _ArtifactUploader(endpoint="https://api.example", api_key="pk").upload(
                b"hello",
                mime_type="text/plain",
            )

        assert ref is not None
        assert [call[0] for call in fake_client.calls] == ["POST", "POST"]
        assert fake_client.calls[1][2]["json"]["contentBase64"] == "aGVsbG8="

    def test_upload_falls_back_to_local_size_for_malformed_response_fields(self):
        class FakeResponse:
            def __init__(self, status_code=200, data=None):
                self.status_code = status_code
                self._data = data or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self._data

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                if url.endswith("/storage-objects/presign"):
                    return FakeResponse(status_code=404)
                return FakeResponse(
                    data={
                        "id": "artifact-id",
                        "uri": "promptic-artifact://artifact-id",
                        "mimeType": 123,
                        "sizeBytes": "not-a-number",
                        "sha256": None,
                    }
                )

        with patch("promptic_sdk.tracing.httpx.Client", return_value=FakeClient()):
            ref = _ArtifactUploader(endpoint="https://api.example", api_key="pk").upload(
                b"hello",
                mime_type="text/plain",
            )

        assert ref is not None
        assert ref.mime_type == "text/plain"
        assert ref.size_bytes == 5
        assert ref.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


class TestArtifactHelper:
    def test_missing_path_like_string_raises(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            artifact("/tmp/definitely-missing-report.pdf")

    def test_missing_extensionless_path_like_string_raises(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            artifact("outputs/report")

    def test_missing_windows_path_like_string_raises(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            artifact(r"C:\tmp\definitely-missing-report.pdf")

    def test_plain_text_string_is_uploaded_as_text(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        calls = []

        def fake_upload(self, content, *, mime_type, source_path="$", preview=None):
            calls.append((content, mime_type, source_path, preview))
            return ArtifactReference(
                id="artifact-id",
                uri="promptic-artifact://artifact-id",
                mime_type=mime_type,
                size_bytes=len(content),
                sha256="hash",
            )

        monkeypatch.setattr("promptic_sdk.tracing._ArtifactUploader.upload", fake_upload)

        ref = artifact("plain text content")

        assert ref.id == "artifact-id"
        assert calls == [(b"plain text content", "text/plain", "$", "plain text content")]

    @pytest.mark.parametrize("value", ["Q: answer", "a:1"])
    def test_colon_prefixed_text_is_uploaded_as_text(self, monkeypatch, value):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        calls = []

        def fake_upload(self, content, *, mime_type, source_path="$", preview=None):
            calls.append((content, mime_type, source_path, preview))
            return ArtifactReference(
                id="artifact-id",
                uri="promptic-artifact://artifact-id",
                mime_type=mime_type,
                size_bytes=len(content),
                sha256="hash",
            )

        monkeypatch.setattr("promptic_sdk.tracing._ArtifactUploader.upload", fake_upload)

        ref = artifact(value)

        assert ref.id == "artifact-id"
        assert calls == [(value.encode("utf-8"), "text/plain", "$", value)]

    @pytest.mark.parametrize(
        "value",
        [
            "https://example.com/report.pdf",
            "s3://bucket/report.pdf",
            "file:///tmp/report.pdf",
        ],
    )
    def test_uri_like_string_is_uploaded_as_text(self, monkeypatch, value):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")

        calls = []

        def fake_upload(self, content, *, mime_type, source_path="$", preview=None):
            calls.append((content, mime_type, source_path, preview))
            return ArtifactReference(
                id="artifact-id",
                uri="promptic-artifact://artifact-id",
                mime_type=mime_type,
                size_bytes=len(content),
                sha256="hash",
            )

        monkeypatch.setattr("promptic_sdk.tracing._ArtifactUploader.upload", fake_upload)

        ref = artifact(value)

        assert ref.id == "artifact-id"
        assert calls == [
            (
                value.encode("utf-8"),
                "text/plain",
                "$",
                value,
            )
        ]


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
