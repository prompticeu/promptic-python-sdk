"""Convenience wrapper for configuring OpenTelemetry to send traces to Promptic."""

from __future__ import annotations

import atexit
import base64
import binascii
import contextvars
import hashlib
import inspect
import json
import logging
import mimetypes
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Protocol, TypeAlias
from uuid import UUID

import httpx
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger("promptic_sdk")

_DEFAULT_ENDPOINT = "https://promptic.eu"

PROMPTIC_COMPONENT_ATTR = "promptic.ai_component"
PROMPTIC_DATASET_ID_ATTR = "promptic.dataset.id"
PROMPTIC_RUN_ATTR = "promptic.run"

InstrumentorName: TypeAlias = Literal[
    "openai",
    "anthropic",
    "google_generativeai",
    "vertexai",
    "bedrock",
    "mistralai",
    "cohere",
    "langchain",
    "openai_agents",
    "claude_agent_sdk",
    "google-generativeai",
    "google",
    "gemini",
    "vertex",
    "mistral",
    "openai-agents",
    "openaiagents",
    "claude-agent-sdk",
    "claude-agent",
    "claude_agents",
    "claude",
]
InstrumentorSelection: TypeAlias = InstrumentorName | Iterable[InstrumentorName]

# Context variable that holds the current AI component name.
_current_component: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_ai_component", default=None
)

# Context variable that holds the current dataset UUID.
_current_dataset_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_dataset_id", default=None
)

# Context variable that holds the current run name.
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptic_run", default=None
)

# Instrumentors that we try to auto-detect and enable.
# Each entry: (name, module_path, class_name)
#
# The first three cover direct LLM SDK calls (works across any framework).
# The rest cover framework-level instrumentation; all emit OTel-official
# ``gen_ai.*`` semantic conventions that the backend parser handles uniformly.
#
# Pydantic AI is intentionally absent — it ships its own OTel emitter; users
# opt in by constructing the Agent with ``instrument=True``.
_INSTRUMENTORS: list[tuple[str, str, str]] = [
    # LLM providers (direct SDK calls) — emit gen_ai.* on every chat/completion.
    ("openai", "opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
    ("anthropic", "opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
    (
        "google_generativeai",
        "opentelemetry.instrumentation.google_generativeai",
        "GoogleGenerativeAiInstrumentor",
    ),
    ("vertexai", "opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor"),
    ("bedrock", "opentelemetry.instrumentation.bedrock", "BedrockInstrumentor"),
    ("mistralai", "opentelemetry.instrumentation.mistralai", "MistralAiInstrumentor"),
    ("cohere", "opentelemetry.instrumentation.cohere", "CohereInstrumentor"),
    # Agent frameworks — emit chain/tool/llm spans with the full graph structure.
    ("langchain", "opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
    (
        "openai_agents",
        "opentelemetry.instrumentation.openai_agents",
        "OpenAIAgentsInstrumentor",
    ),
    (
        "claude_agent_sdk",
        "opentelemetry.instrumentation.claude_agent_sdk",
        "ClaudeAgentSdkInstrumentor",
    ),
]

_INSTRUMENTOR_ALIASES = {
    "google-generativeai": "google_generativeai",
    "google": "google_generativeai",
    "gemini": "google_generativeai",
    "vertex": "vertexai",
    "mistral": "mistralai",
    "openai-agents": "openai_agents",
    "openaiagents": "openai_agents",
    "claude-agent-sdk": "claude_agent_sdk",
    "claude-agent": "claude_agent_sdk",
    "claude_agents": "claude_agent_sdk",
    "claude": "claude_agent_sdk",
}

_INSTRUMENTOR_NAMES = {name for name, _, _ in _INSTRUMENTORS}

_OPENAI_INSTRUMENTATION_MODULE = "opentelemetry.instrumentation.openai"
_ARTIFACT_URI_SCHEME = "promptic-artifact://"
_DATA_URI_PREFIX = "data:"
_GENERIC_BASE64_MIN_BYTES = 8 * 1024
_LARGE_TEXT_MIN_BYTES = 256 * 1024
_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
_TEXT_PREVIEW_CHARS = 4096
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_FILENAME_LIKE_RE = re.compile(r"^[^\s/\\]+\.[A-Za-z][A-Za-z0-9]{0,5}$")
_MEDIA_PATH_MARKERS = (
    "image",
    "img",
    "audio",
    "video",
    "file",
    "pdf",
    "blob",
    "bytes",
    "base64",
    "b64",
    "attachment",
)

_configured_api_key: str | None = None
_configured_endpoint: str | None = None


def _split_instrumentor_env(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]
    return names or None


def _normalize_instrumentor_name(name: str) -> str:
    raw = name.strip().lower()
    normalized = raw.replace("-", "_")
    return _INSTRUMENTOR_ALIASES.get(raw, _INSTRUMENTOR_ALIASES.get(normalized, normalized))


def _resolve_instrumentor_names(
    names: InstrumentorSelection | None,
    *,
    env_var: str,
    default: set[str],
) -> set[str]:
    if names is None:
        raw_names = _split_instrumentor_env(os.environ.get(env_var))
    elif isinstance(names, str):
        raw_names = [names]
    else:
        raw_names = list(names)

    if raw_names is None:
        return set(default)

    resolved = {_normalize_instrumentor_name(name) for name in raw_names}
    unknown = sorted(resolved - _INSTRUMENTOR_NAMES)
    if unknown:
        msg = (
            "Unknown Promptic instrumentor(s): "
            f"{', '.join(unknown)}. Valid instrumentors: "
            f"{', '.join(sorted(_INSTRUMENTOR_NAMES))}."
        )
        raise ValueError(msg)
    return resolved


def _selected_instrumentors(
    instrumentors: InstrumentorSelection | None = None,
    *,
    exclude_instrumentors: InstrumentorSelection | None = None,
) -> list[tuple[str, str, str]]:
    selected = _resolve_instrumentor_names(
        instrumentors,
        env_var="PROMPTIC_INSTRUMENTORS",
        default=_INSTRUMENTOR_NAMES,
    )
    excluded = _resolve_instrumentor_names(
        exclude_instrumentors,
        env_var="PROMPTIC_EXCLUDE_INSTRUMENTORS",
        default=set(),
    )
    selected -= excluded
    return [item for item in _INSTRUMENTORS if item[0] in selected]


def _instrumentor_init_kwargs(module_path: str) -> dict[str, object]:
    """Return compatibility kwargs for instrumentors that need SDK defaults."""
    if module_path == _OPENAI_INSTRUMENTATION_MODULE:
        # OpenLLMetry's OpenAI image prompt extraction awaits this optional hook.
        # Its default no-op is sync in some releases, which drops multimodal input
        # capture behind a swallowed TypeError. Passing None keeps inline image
        # references in gen_ai.input.messages until Promptic has media storage.
        return {"upload_base64_image": None}
    return {}


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


@dataclass(frozen=True)
class ArtifactReference:
    """Reference to an artifact uploaded to Promptic."""

    id: str
    uri: str
    mime_type: str
    size_bytes: int
    sha256: str

    @property
    def ref(self) -> str:
        """String reference suitable for OpenTelemetry span attributes."""
        return self.uri

    def to_dict(self) -> dict[str, Any]:
        """Return a structured JSON-safe artifact reference."""
        return {
            "$prompticArtifact": {
                "id": self.id,
                "uri": self.uri,
                "mimeType": self.mime_type,
                "sizeBytes": self.size_bytes,
                "sha256": self.sha256,
            }
        }


class _ArtifactUploader:
    """Synchronous artifact uploader used by exporter and public helper."""

    def __init__(self, *, endpoint: str, api_key: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key

    def _direct_upload(
        self,
        client: httpx.Client,
        content: bytes,
        *,
        mime_type: str,
        source_path: str,
        source_field: str | None,
        preview: str | None,
        sha256: str,
    ) -> dict[str, Any] | None:
        filename = _source_path_name(source_path)
        if "." not in filename:
            filename = f"{filename}{mimetypes.guess_extension(mime_type) or ''}"

        try:
            presign_response = client.post(
                f"{self._endpoint}/api/v1/storage-objects/presign",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "folder": "trace-artifacts",
                    "filename": filename,
                    "contentType": mime_type,
                    "access": "private",
                    "sizeBytes": len(content),
                    "maxSizeBytes": _MAX_ARTIFACT_BYTES,
                },
            )
            if presign_response.status_code == 404:
                return None
            presign_response.raise_for_status()
        except Exception:
            logger.debug("Promptic: storage presign failed; falling back to server upload.")
            return None
        presigned = presign_response.json()

        upload_url = presigned.get("uploadUrl")
        storage_object_id = presigned.get("storageObjectId")
        if not isinstance(upload_url, str) or not upload_url:
            logger.warning("Promptic: storage presign response did not include uploadUrl.")
            return None
        if not isinstance(storage_object_id, str) or not storage_object_id:
            logger.warning("Promptic: storage presign response did not include storageObjectId.")
            return None

        method = str(presigned.get("method") or "PUT").upper()
        if method == "POST":
            files = {
                str(key): (None, str(value))
                for key, value in dict(presigned.get("fields") or {}).items()
            }
            files["file"] = (filename, content, mime_type)
            upload_response = client.post(upload_url, files=files)
        else:
            headers = {
                str(key): str(value) for key, value in dict(presigned.get("headers") or {}).items()
            }
            headers.setdefault("Content-Type", mime_type)
            upload_response = client.put(upload_url, content=content, headers=headers)
        upload_response.raise_for_status()

        register_response = client.post(
            f"{self._endpoint}/api/v1/artifacts",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "storageObjectId": storage_object_id,
                "mimeType": mime_type,
                "sizeBytes": len(content),
                "sha256": sha256,
                "sourcePath": source_path,
                "sourceField": source_field or _artifact_source_field(source_path),
                "source": "direct_upload",
                "preview": preview,
            },
        )
        register_response.raise_for_status()
        return register_response.json()

    def upload(
        self,
        content: bytes,
        *,
        mime_type: str,
        source_path: str = "$",
        source_field: str | None = None,
        preview: str | None = None,
    ) -> ArtifactReference | None:
        if len(content) > _MAX_ARTIFACT_BYTES:
            logger.warning(
                "Promptic: artifact at %s is too large to upload (%d bytes).",
                source_path,
                len(content),
            )
            return None

        sha256 = hashlib.sha256(content).hexdigest()

        try:
            with httpx.Client(timeout=30) as client:
                try:
                    data = self._direct_upload(
                        client,
                        content,
                        mime_type=mime_type,
                        source_path=source_path,
                        source_field=source_field,
                        preview=preview,
                        sha256=sha256,
                    )
                except Exception:
                    logger.warning(
                        "Promptic: direct artifact upload failed; falling back to server upload.",
                        exc_info=True,
                    )
                    data = None
                if data is None:
                    response = client.post(
                        f"{self._endpoint}/api/v1/artifacts",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "contentBase64": base64.b64encode(content).decode("ascii"),
                            "mimeType": mime_type,
                            "sourcePath": source_path,
                            "sourceField": source_field or _artifact_source_field(source_path),
                            "preview": preview,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
        except Exception:
            logger.warning("Promptic: failed to upload trace artifact.", exc_info=True)
            return None

        if not isinstance(data, Mapping):
            logger.warning("Promptic: artifact upload returned a non-object response.")
            return None

        artifact_id = data.get("id")
        uri = data.get("uri") or (f"{_ARTIFACT_URI_SCHEME}{artifact_id}" if artifact_id else None)
        if not artifact_id or not uri:
            logger.warning("Promptic: artifact upload returned an invalid response.")
            return None

        raw_size = data.get("sizeBytes")
        try:
            size_bytes = int(raw_size) if raw_size is not None else len(content)
        except (TypeError, ValueError):
            size_bytes = len(content)
        raw_mime_type = data.get("mimeType")
        raw_sha256 = data.get("sha256")

        return ArtifactReference(
            id=str(artifact_id),
            uri=str(uri),
            mime_type=raw_mime_type if isinstance(raw_mime_type, str) else mime_type,
            size_bytes=size_bytes,
            sha256=raw_sha256 if isinstance(raw_sha256, str) else sha256,
        )


def _is_media_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in _MEDIA_PATH_MARKERS)


def _preview_text(value: str) -> str:
    if len(value) <= _TEXT_PREVIEW_CHARS:
        return value
    return f"{value[:_TEXT_PREVIEW_CHARS]}..."


def _normalise_base64(value: str) -> bytes | None:
    compact = "".join(value.split())
    if len(compact) < 16 or len(compact) % 4 != 0:
        return None
    try:
        content = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not content:
        return None
    if base64.b64encode(content).decode("ascii").rstrip("=") != compact.rstrip("="):
        return None
    return content


def _decode_data_uri(value: str) -> tuple[str, bytes] | None:
    if not value.startswith(_DATA_URI_PREFIX) or ";base64," not in value:
        return None
    header, encoded = value.split(",", 1)
    mime_type = header[len(_DATA_URI_PREFIX) :].split(";", 1)[0] or "application/octet-stream"
    content = _normalise_base64(encoded)
    if content is None:
        return None
    return mime_type, content


def _json_loads_maybe(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _source_path_name(source_path: str) -> str:
    if source_path == "$":
        return "artifact"
    if "\\" in source_path:
        return PureWindowsPath(source_path).name or "artifact"
    return Path(source_path).name or "artifact"


def _artifact_source_field(source_path: str) -> str:
    return "manual" if source_path == "$" else "metadata"


def _looks_like_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped != value or "\n" in value or "\r" in value:
        return False
    if stripped.lower().startswith(("http://", "https://", "data:")):
        return False
    if (
        len(stripped) >= 3
        and stripped[1] == ":"
        and stripped[0].isalpha()
        and stripped[2] in {"/", "\\"}
    ):
        return True
    if _URI_SCHEME_RE.match(stripped):
        return False
    if stripped.startswith(("~/", "~\\", "./", ".\\", "../", "..\\")):
        return True
    if stripped.startswith(("/", "\\")):
        return True
    if "\\" in stripped or "/" in stripped:
        return True
    return bool(_FILENAME_LIKE_RE.match(_source_path_name(stripped)))


def _configure_artifacts_once(*, api_key: str, endpoint: str) -> None:
    global _configured_api_key, _configured_endpoint

    if _configured_api_key is None:
        _configured_api_key = api_key
    if _configured_endpoint is None:
        _configured_endpoint = endpoint


class _ArtifactUploadBackend(Protocol):
    def upload(
        self,
        content: bytes,
        *,
        mime_type: str,
        source_path: str = "$",
        preview: str | None = None,
    ) -> ArtifactReference | None: ...


class _ArtifactSanitizer:
    def __init__(self, uploader: _ArtifactUploadBackend) -> None:
        self._uploader = uploader

    def sanitize_attribute(self, key: str, value: Any) -> Any:
        return self._sanitize(value, key, prefer_json_roundtrip=True)

    def _artifact_ref(self, ref: ArtifactReference) -> dict[str, Any]:
        return ref.to_dict()

    def _upload_string(self, value: str, path: str, *, mime_type_hint: str | None = None) -> Any:
        data_uri = _decode_data_uri(value)
        if data_uri:
            mime_type, content = data_uri
            ref = self._uploader.upload(content, mime_type=mime_type, source_path=path)
            return ref.uri if ref else value

        if not value.startswith(("http://", "https://")) and (
            _is_media_path(path) or mime_type_hint
        ):
            content = _normalise_base64(value)
            if content is not None and len(content) >= _GENERIC_BASE64_MIN_BYTES:
                ref = self._uploader.upload(
                    content,
                    mime_type=mime_type_hint or "application/octet-stream",
                    source_path=path,
                )
                return ref.uri if ref else value

        if len(value.encode("utf-8")) >= _LARGE_TEXT_MIN_BYTES:
            ref = self._uploader.upload(
                value.encode("utf-8"),
                mime_type="text/plain",
                source_path=path,
                preview=_preview_text(value),
            )
            return ref.uri if ref else value

        return value

    def _sanitize(
        self,
        value: Any,
        path: str,
        *,
        prefer_json_roundtrip: bool = False,
        mime_type_hint: str | None = None,
    ) -> Any:
        if isinstance(value, str):
            parsed = _json_loads_maybe(value) if prefer_json_roundtrip else None
            if parsed is not None:
                sanitized = self._sanitize(parsed, path)
                if sanitized != parsed:
                    return json.dumps(sanitized, separators=(",", ":"))
            return self._upload_string(value, path, mime_type_hint=mime_type_hint)

        if isinstance(value, Mapping):
            changed = False
            next_value: dict[str, Any] = {}
            record_mime_type = value.get("mime_type") or value.get("media_type")
            child_mime_type = record_mime_type if isinstance(record_mime_type, str) else None
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                sanitized = self._sanitize(
                    child,
                    child_path,
                    mime_type_hint=(
                        child_mime_type
                        if child_mime_type and key in {"content", "data", "blob"}
                        else None
                    ),
                )
                changed = changed or sanitized != child
                next_value[str(key)] = sanitized
            return next_value if changed else value

        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            changed = False
            next_items: list[Any] = []
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                sanitized = self._sanitize(child, child_path)
                changed = changed or sanitized != child
                next_items.append(sanitized)
            return next_items if changed else value

        return value

    def sanitize_span(self, span: ReadableSpan) -> ReadableSpan:
        attrs = span.attributes
        events = span.events
        sanitized_attrs = (
            {key: self.sanitize_attribute(key, value) for key, value in attrs.items()}
            if attrs
            else attrs
        )
        sanitized_events = []
        events_changed = False
        for event_index, event in enumerate(events):
            event_attrs = event.attributes
            if event_attrs:
                next_attrs = {
                    key: self._sanitize(
                        value, f"events[{event_index}].{key}", prefer_json_roundtrip=True
                    )
                    for key, value in event_attrs.items()
                }
            else:
                next_attrs = event_attrs
            events_changed = events_changed or next_attrs != event_attrs
            sanitized_events.append(
                Event(event.name, attributes=next_attrs, timestamp=event.timestamp)
            )

        if sanitized_attrs == attrs and not events_changed:
            return span

        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=sanitized_attrs,
            events=tuple(sanitized_events),
            links=span.links,
            kind=span.kind,
            instrumentation_info=span.instrumentation_info,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )


class _ArtifactRewritingExporter(SpanExporter):
    """Uploads large inline media/file payloads and replaces them with refs."""

    def __init__(self, inner: SpanExporter, *, endpoint: str, api_key: str) -> None:
        self._inner = inner
        self._sanitizer = _ArtifactSanitizer(_ArtifactUploader(endpoint=endpoint, api_key=api_key))

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        sanitized = [self._sanitizer.sanitize_span(span) for span in spans]
        return self._inner.export(sanitized)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


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
        parent_export = super()._export
        if "timeout_sec" in inspect.signature(parent_export).parameters:
            resp = parent_export(serialized_data, timeout_sec)
        else:
            resp = parent_export(serialized_data)
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
    """Add Promptic ownership attributes to spans created inside SDK contexts."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        name = _current_component.get()
        if name is not None:
            span.set_attribute(PROMPTIC_COMPONENT_ATTR, name)
        dataset_id = _current_dataset_id.get()
        if dataset_id is not None:
            span.set_attribute(PROMPTIC_DATASET_ID_ATTR, dataset_id)
        run = _current_run.get()
        if run is not None:
            span.set_attribute(PROMPTIC_RUN_ATTR, run)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D102
        pass

    def shutdown(self) -> None:  # noqa: D102
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D102
        return True


def artifact(
    value: str | bytes | Path,
    *,
    mime_type: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> ArtifactReference:
    """Upload a local file or bytes and return a trace-safe artifact reference.

    Local files are explicit on purpose: the SDK does not silently read paths
    that happen to appear in span attributes.

    Example::

        ref = promptic_sdk.artifact("report.pdf")
        span.set_attribute("retrieval.input_file", ref.ref)
    """
    resolved_api_key = api_key or _configured_api_key or os.environ.get("PROMPTIC_API_KEY")
    if not resolved_api_key:
        msg = "Promptic API key is required to upload artifacts."
        raise ValueError(msg)

    resolved_endpoint = (
        endpoint or _configured_endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
    )

    if isinstance(value, Path) or (isinstance(value, str) and Path(value).exists()):
        path = Path(value)
        content = path.read_bytes()
        resolved_mime_type = (
            mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        source_path = str(path)
    elif isinstance(value, bytes):
        content = value
        resolved_mime_type = mime_type or "application/octet-stream"
        source_path = "$"
    elif isinstance(value, str):
        if _looks_like_path(value):
            msg = (
                f"Artifact file path does not exist: {value!r}. "
                "Pass bytes/text explicitly for inline artifact content."
            )
            raise FileNotFoundError(msg)
        content = value.encode("utf-8")
        resolved_mime_type = mime_type or "text/plain"
        source_path = "$"
    else:
        msg = "artifact() expects a path, bytes, or text string."
        raise TypeError(msg)

    ref = _ArtifactUploader(endpoint=resolved_endpoint, api_key=resolved_api_key).upload(
        content,
        mime_type=resolved_mime_type,
        source_path=source_path,
        source_field="manual",
        preview=_preview_text(content.decode("utf-8", errors="ignore"))
        if resolved_mime_type.startswith("text/")
        else None,
    )
    if ref is None:
        msg = "Failed to upload artifact to Promptic."
        raise RuntimeError(msg)
    return ref


def init(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    auto_instrument: bool = True,
    instrumentors: InstrumentorSelection | None = None,
    exclude_instrumentors: InstrumentorSelection | None = None,
    service_name: str | None = None,
) -> None:
    """Configure OpenTelemetry to send traces to Promptic.

    Args:
        api_key: Promptic API key. Falls back to ``PROMPTIC_API_KEY`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://promptic.eu``.
        auto_instrument: If True, auto-detect installed LLM client libraries and
            instrument them.
        instrumentors: Optional instrumentor names to enable. Falls back to the
            ``PROMPTIC_INSTRUMENTORS`` env var, then to all known instrumentors.
        exclude_instrumentors: Optional instrumentor names to skip. Falls back to the
            ``PROMPTIC_EXCLUDE_INSTRUMENTORS`` env var.
        service_name: OpenTelemetry ``service.name`` resource attribute.
    """
    global _configured_api_key, _configured_endpoint

    api_key = api_key or os.environ.get("PROMPTIC_API_KEY")
    if not api_key:
        msg = (
            "Promptic API key is required. "
            "Pass api_key= or set the PROMPTIC_API_KEY environment variable."
        )
        raise ValueError(msg)

    endpoint = endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
    selected_instrumentors = None
    if auto_instrument:
        selected_instrumentors = _selected_instrumentors(
            instrumentors,
            exclude_instrumentors=exclude_instrumentors,
        )

    if getattr(trace._TRACER_PROVIDER_SET_ONCE, "_done", False):  # noqa: SLF001
        logger.warning("Promptic tracing is already initialized; ignoring repeated init() call.")
        _configure_artifacts_once(api_key=api_key, endpoint=endpoint)
        if auto_instrument:
            _auto_instrument(
                instrumentors=instrumentors,
                exclude_instrumentors=exclude_instrumentors,
                selected=selected_instrumentors,
            )
        return
    traces_endpoint = f"{endpoint.rstrip('/')}/api/v1/traces"

    # Layered exporter:
    #   _LoggingExporter      → emits a one-time WARNING on the first failure
    #   _ArtifactRewritingExporter → uploads large inline payloads once per batch
    #   _BisectingExporter    → on HTTP 413, halves sanitized spans and retries
    #   _OTLPSpanExporter413Aware → raises _BodyTooLargeError for 413 so the
    #                                bisecter sees it (instead of the parent
    #                                swallowing it as a generic FAILURE)
    #
    # With this stack, oversized batches recover transparently without
    # re-uploading artifacts on each bisected retry. We keep OTel's default
    # `max_export_batch_size` (512) because the bisecter makes overflow free.
    exporter = _LoggingExporter(
        _ArtifactRewritingExporter(
            _BisectingExporter(
                _OTLPSpanExporter413Aware(
                    endpoint=traces_endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
            ),
            endpoint=endpoint,
            api_key=api_key,
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
    _configure_artifacts_once(api_key=api_key, endpoint=endpoint)

    # Ensure all spans are flushed when the process exits.
    atexit.register(provider.shutdown)

    if auto_instrument:
        _auto_instrument(
            instrumentors=instrumentors,
            exclude_instrumentors=exclude_instrumentors,
            selected=selected_instrumentors,
        )


def _langsmith_tracing_context(
    component: str,
    dataset_id: str | None,
    run: str | None,
) -> AbstractContextManager | None:
    """Return a ``langsmith.tracing_context`` if available, else ``None``.

    Injects Promptic attributes into LangSmith run metadata so they appear
    as span attributes when the LangSmith OTel exporter converts runs to
    OTel spans.  Without this, LangSmith-created spans would lack the
    ``promptic.ai_component`` / ``promptic.dataset.id`` / ``promptic.run``
    attributes needed to link traces to AI components.
    """
    if os.environ.get("LANGSMITH_OTEL_ENABLED", "").lower() != "true":
        return None
    try:
        from langsmith import tracing_context
    except ImportError:
        return None

    metadata: dict[str, str] = {PROMPTIC_COMPONENT_ATTR: component}
    if dataset_id:
        metadata[PROMPTIC_DATASET_ID_ATTR] = dataset_id
    if run:
        metadata[PROMPTIC_RUN_ATTR] = run
    return tracing_context(metadata=metadata)


@contextmanager
def ai_component(
    name: str,
    *,
    dataset_id: str | None = None,
    run: str | None = None,
) -> Iterator[None]:
    """Tag all spans created within this context with an AI Component name.

    The server matches the name against AI Components in the workspace
    and links traces accordingly.

    Args:
        name: AI Component name in the workspace.
        dataset_id: Optional dataset UUID. When set, traces are added to that
            existing dataset. Invalid IDs fail before any spans are created.
        run: Optional run name. When set alongside ``dataset_id``, traces are
            grouped into a named run within the dataset. Each unique run name
            creates a separate run, allowing you to compare different
            executions against the same dataset.

    Example::

        with promptic_sdk.ai_component("customer-support-agent"):
            response = openai_client.chat.completions.create(...)

        # With dataset and run tagging:
        with promptic_sdk.ai_component(
            "my-agent",
            dataset_id="550e8400-e29b-41d4-a716-446655440000",
            run="v1-baseline",
        ):
            agent.run(test_input)
    """
    normalized_dataset_id = _normalize_dataset_id(dataset_id) if dataset_id is not None else None
    if run is not None and normalized_dataset_id is None:
        msg = "run requires dataset_id so the trace can be linked to a dataset run."
        raise ValueError(msg)

    comp_token = _current_component.set(name)
    ds_token = _current_dataset_id.set(normalized_dataset_id) if normalized_dataset_id else None
    run_token = _current_run.set(run) if run else None

    # When LangSmith OTel bridge is active, inject Promptic attributes into
    # LangSmith run metadata so they appear as span attributes after export.
    langsmith_ctx = _langsmith_tracing_context(name, normalized_dataset_id, run)

    try:
        if langsmith_ctx is not None:
            langsmith_ctx.__enter__()
        yield
    finally:
        if langsmith_ctx is not None:
            langsmith_ctx.__exit__(None, None, None)
        _current_component.reset(comp_token)
        if ds_token is not None:
            _current_dataset_id.reset(ds_token)
        if run_token is not None:
            _current_run.reset(run_token)


@contextmanager
def dataset(dataset_id: str) -> Iterator[None]:
    """Tag all spans created within this context with a dataset UUID.

    The dataset must already exist. Invalid UUIDs fail before the context is
    entered, while unknown IDs are rejected by trace ingestion.

    Can be nested inside :func:`ai_component` for composability::

        with promptic_sdk.ai_component("my-agent"):
            with promptic_sdk.dataset("550e8400-e29b-41d4-a716-446655440000"):
                agent.run(test_input)
    """
    token = _current_dataset_id.set(_normalize_dataset_id(dataset_id))
    try:
        yield
    finally:
        _current_dataset_id.reset(token)


def _normalize_dataset_id(dataset_id: str) -> str:
    """Validate and normalize a dataset UUID before emitting trace attributes."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        msg = "dataset_id must be a UUID string."
        raise ValueError(msg)
    try:
        return str(UUID(dataset_id))
    except ValueError as error:
        msg = "dataset_id must be a valid UUID string."
        raise ValueError(msg) from error


def _warn_on_instrumentor_conflicts(selected_names: set[str]) -> None:
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    langsmith_otel = os.environ.get("LANGSMITH_OTEL_ENABLED", "").lower() == "true"

    if langsmith_tracing and not langsmith_otel and "langchain" in selected_names:
        logger.warning(
            "Promptic: LANGSMITH_TRACING=true is set without LANGSMITH_OTEL_ENABLED=true. "
            "LangSmith's callback handler can intercept LangChain/LangGraph runs before "
            "OpenLLMetry can instrument them, so ChatOpenAI spans and tool definitions "
            "may be missing from your Promptic traces. Unset LANGSMITH_TRACING, or set "
            "LANGSMITH_OTEL_ENABLED=true and exclude the Promptic langchain instrumentor."
        )

    if langsmith_otel and "langchain" in selected_names:
        logger.warning(
            "Promptic: LANGSMITH_OTEL_ENABLED=true and the Promptic langchain "
            "instrumentor are both active. LangSmith emits OTel spans through the "
            "global Promptic tracer provider, so enabling both can duplicate "
            "LangChain/LangGraph model spans. Disable one path, for example with "
            "PROMPTIC_EXCLUDE_INSTRUMENTORS=langchain."
        )

    if "openai" in selected_names and "langchain" in selected_names:
        logger.warning(
            "Promptic: both openai and langchain instrumentors are enabled. "
            "Frameworks built on LangChain, including LangGraph and DeepAgents, "
            "can emit duplicate model spans when the underlying OpenAI SDK is also "
            "instrumented. For LangChain-style apps, prefer "
            "PROMPTIC_EXCLUDE_INSTRUMENTORS=openai or pass exclude_instrumentors=['openai']."
        )

    if (
        "openai_agents" in selected_names
        and os.environ.get("OPENAI_AGENTS_DISABLE_TRACING", "").lower() != "true"
    ):
        logger.warning(
            "Promptic: openai_agents instrumentation is enabled while native OpenAI "
            "Agents tracing is not disabled. Promptic tracing does not require the "
            "OpenAI Agents native exporter; set OPENAI_AGENTS_DISABLE_TRACING=true "
            "if you do not want a separate export to OpenAI."
        )


def _instrumentor_module_exists(module_path: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_path) is not None
    except (ImportError, ValueError):
        return False


def _auto_instrument(
    instrumentors: InstrumentorSelection | None = None,
    *,
    exclude_instrumentors: InstrumentorSelection | None = None,
    selected: list[tuple[str, str, str]] | None = None,
) -> None:
    """Try to import and enable each known instrumentor.

    OpenLLMetry's instrumentors are the primary path. They emit OTel-official
    ``gen_ai.*`` semantic conventions (including ``gen_ai.tool.definitions``)
    and cover LangGraph / deepagents correctly as of
    ``opentelemetry-instrumentation-langchain>=0.60.0``.

    ``LANGSMITH_OTEL_ENABLED=true`` makes LangSmith emit spans through the same
    global OTel provider that Promptic installs. Keep that path mutually
    exclusive with the Promptic LangChain instrumentor to avoid duplicate model
    spans.
    """
    import importlib

    if selected is None:
        selected = _selected_instrumentors(
            instrumentors,
            exclude_instrumentors=exclude_instrumentors,
        )
    loaded_names: set[str] = set()

    for name, module_path, class_name in selected:
        if not _instrumentor_module_exists(module_path):
            logger.debug(
                "Promptic: skipping optional %s instrumentor (%s could not be imported).",
                name,
                module_path,
            )
            continue

        try:
            mod = importlib.import_module(module_path)
            instrumentor_cls = getattr(mod, class_name)
            instrumentor_cls(**_instrumentor_init_kwargs(module_path)).instrument()
            loaded_names.add(name)
            logger.debug("Promptic: enabled %s.%s", module_path, class_name)
        except ImportError as exc:
            logger.warning(
                "Promptic: failed to import installed %s instrumentor (%s): %s",
                name,
                module_path,
                exc,
                exc_info=True,
            )
        except Exception:
            logger.warning(
                "Promptic: failed to enable %s.%s — the package may be incompatible.",
                module_path,
                class_name,
                exc_info=True,
            )

    _warn_on_instrumentor_conflicts(loaded_names)
