"""Agent Gym external submission clients and session helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias, cast
from uuid import UUID

import httpx

from promptic_sdk.agent_gym_models import (
    BundleIdentity,
    CancelledSubmission,
    CompletedSubmissionArtifact,
    ExecutionRefs,
    ExternalPrediction,
    ExternalSubmissionCreated,
    ExternalSubmissionManifest,
    FinalizedSubmission,
    FinalizeSubmissionRequest,
    ManifestCase,
    ManifestInputFile,
    ManifestPage,
    ReservedSubmissionArtifact,
    SubmissionArtifact,
    SubmissionStatus,
    TraceResolutionList,
)
from promptic_sdk.client import PrompticAPIError

_DEFAULT_ENDPOINT = "https://promptic.eu"
_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
_MAX_TRACE_IDS = 100
_TRACE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_TERMINAL_SUBMISSION_STATUSES = {"succeeded", "failed", "expired", "cancelled"}
_UNSET = object()

RequestParams: TypeAlias = Mapping[str, Any] | Sequence[tuple[str, Any]]


class AgentGymAPIError(PrompticAPIError):
    """Structured error returned by an Agent Gym submission endpoint."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a credential-safe API error."""
        self.status_code = status_code
        self.code = code
        self.message = message or code
        self.details = details
        super().__init__(status_code, f"{code}: {self.message}")
        self.message = message or code


class ArtifactUploadError(Exception):
    """A direct-to-storage upload failed."""

    def __init__(self, status_code: int | None, message: str) -> None:
        """Initialize without retaining the credential-bearing upload URL."""
        self.status_code = status_code
        self.message = message
        prefix = f"[{status_code}] " if status_code is not None else ""
        super().__init__(f"{prefix}{message}")


class ArtifactIntegrityError(Exception):
    """Downloaded or local artifact bytes do not match declared integrity metadata."""


class UnresolvedTraceError(Exception):
    """One or more OTEL trace IDs have not been ingested into Promptic."""

    def __init__(self, trace_ids: Sequence[str]) -> None:
        """Initialize with unresolved raw OTEL trace IDs."""
        self.trace_ids = list(trace_ids)
        super().__init__(f"{len(self.trace_ids)} trace ID(s) were not resolved")


@dataclass(frozen=True)
class MaterializedInputFile:
    """One manifest input downloaded to a safe local path."""

    case_id: str
    artifact_id: str
    logical_path: str
    local_path: Path
    mime_type: str
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True)
class MaterializedManifest:
    """A collected manifest and its locally downloaded input files."""

    root: Path
    manifest_path: Path
    files: tuple[MaterializedInputFile, ...]
    manifest: ExternalSubmissionManifest = field(repr=False)


def _require_uuid(value: str, field_name: str) -> str:
    try:
        UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    return value


def _require_idempotency_key(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        msg = "idempotency_key must contain 1-200 non-whitespace characters"
        raise ValueError(msg)
    return normalized


def _validate_artifact_path(value: str) -> str:
    if not value or len(value) > 500 or value.startswith("/") or "\\" in value or "\0" in value:
        msg = "artifact path must be a relative normalized POSIX path"
        raise ValueError(msg)
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        msg = "artifact path must be a relative normalized POSIX path"
        raise ValueError(msg)
    return "/".join(parts)


def _validate_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        msg = "sha256 must be a lowercase 64-character hexadecimal digest"
        raise ValueError(msg)
    return value


def _validate_utc_datetime(value: str, field_name: str) -> None:
    if not _UTC_DATETIME_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be an ISO 8601 UTC datetime ending in Z")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime") from error


def _bounded_string(
    value: str,
    field_name: str,
    *,
    maximum: int,
    require_non_empty: bool = False,
    trim: bool = False,
) -> str:
    normalized = value.strip() if trim else value
    if len(normalized) > maximum or (require_non_empty and not normalized):
        qualifier = "1-" if require_non_empty else "at most "
        raise ValueError(f"{field_name} must contain {qualifier}{maximum} characters")
    return normalized


def _normalize_trace_ids(trace_ids: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for trace_id in trace_ids:
        lowered = trace_id.lower()
        if not _TRACE_ID_RE.fullmatch(lowered):
            msg = "OTEL trace IDs must contain exactly 32 hexadecimal characters"
            raise ValueError(msg)
        if lowered not in normalized:
            normalized.append(lowered)
    if not normalized or len(normalized) > _MAX_TRACE_IDS:
        raise ValueError(f"trace_ids must contain 1-{_MAX_TRACE_IDS} unique values")
    return normalized


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_execution_refs(refs: ExecutionRefs | None, field_name: str) -> ExecutionRefs | None:
    if refs is None:
        return None
    unknown = set(refs) - {"trace_ids", "trace_artifact_ids"}
    if unknown:
        raise ValueError(f"{field_name} contains unsupported fields: {sorted(unknown)}")
    result: ExecutionRefs = {}
    for key in ("trace_ids", "trace_artifact_ids"):
        values = refs.get(key)
        if values is None:
            continue
        if len(values) > 100:
            raise ValueError(f"{field_name}.{key} cannot contain more than 100 IDs")
        normalized = _unique(values)
        for value in normalized:
            _require_uuid(value, f"{field_name}.{key}")
        result[cast(Any, key)] = normalized
    return result


def _validate_prediction_runtime_fields(result: ExternalPrediction, prefix: str) -> None:
    implementation_reference_id = result.get("implementation_reference_id")
    if implementation_reference_id is not None:
        _require_uuid(implementation_reference_id, f"{prefix}.implementation_reference_id")
    for key in ("executor_id", "executor_version"):
        value = result.get(key)
        if value is not None:
            result[cast(Any, key)] = _bounded_string(
                value,
                f"{prefix}.{key}",
                maximum=200,
                require_non_empty=True,
            )
    token_usage = result.get("token_usage")
    if token_usage is not None:
        for key in ("prompt", "completion", "total"):
            value = token_usage[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{prefix}.token_usage.{key} must be a non-negative integer")
    latency_ms = result.get("latency_ms")
    if latency_ms is not None and (
        isinstance(latency_ms, bool) or not isinstance(latency_ms, int) or latency_ms < 0
    ):
        raise ValueError(f"{prefix}.latency_ms must be a non-negative integer")
    for key in ("started_at", "completed_at"):
        value = result.get(key)
        if value is not None:
            _validate_utc_datetime(value, f"{prefix}.{key}")
    for key, maximum in (
        ("error_code", 200),
        ("error_category", 200),
        ("error", 10_000),
    ):
        value = result.get(key)
        if value is not None:
            _bounded_string(value, f"{prefix}.{key}", maximum=maximum)
    diagnostics = result.get("diagnostics")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError(f"{prefix}.diagnostics must be an object")


def _validate_prediction(prediction: ExternalPrediction, index: int) -> ExternalPrediction:
    result = cast(ExternalPrediction, dict(prediction))
    prefix = f"predictions[{index}]"
    _require_uuid(result["revision_case_id"], f"{prefix}.revision_case_id")
    status = result["status"]
    if status not in {"succeeded", "failed", "skipped", "cancelled"}:
        raise ValueError(f"{prefix}.status is invalid")
    output = result.get("output")
    error_code = result.get("error_code")
    if status == "succeeded" and output is None:
        raise ValueError(f"{prefix}.output is required for succeeded predictions")
    if status != "succeeded" and output is not None:
        raise ValueError(f"{prefix}.output is only valid for succeeded predictions")
    if status != "succeeded" and not error_code:
        raise ValueError(f"{prefix}.error_code is required for non-successful predictions")
    if output is not None and output["kind"] not in {
        "structured",
        "text",
        "artifact",
        "side_effect",
        "none",
        "custom",
    }:
        raise ValueError(f"{prefix}.output.kind is invalid")

    artifact_ids = _unique(result.get("artifact_ids", []))
    if len(artifact_ids) != len(result.get("artifact_ids", [])):
        raise ValueError(f"{prefix}.artifact_ids cannot contain duplicates")
    if len(artifact_ids) > 20:
        raise ValueError(f"{prefix}.artifact_ids cannot contain more than 20 IDs")
    for artifact_id in artifact_ids:
        _require_uuid(artifact_id, f"{prefix}.artifact_ids")
    result["artifact_ids"] = artifact_ids

    refs = _validate_execution_refs(result.get("execution_refs"), f"{prefix}.execution_refs")
    if refs is not None:
        result["execution_refs"] = refs

    _validate_prediction_runtime_fields(result, prefix)
    return result


def _normalize_bundle_identity(value: BundleIdentity) -> BundleIdentity:
    identity = cast(BundleIdentity, dict(value))
    name = identity["name"].strip()
    version = identity["version"].strip()
    if not name or len(name) > 100:
        raise ValueError("bundle_identity.name must contain 1-100 characters")
    if not version or len(version) > 50:
        raise ValueError("bundle_identity.version must contain 1-50 characters")
    identity["name"] = name
    identity["version"] = version
    if "parent_version" in identity:
        identity["parent_version"] = _bounded_string(
            identity["parent_version"],
            "bundle_identity.parent_version",
            maximum=50,
            require_non_empty=True,
            trim=True,
        )
    if "rationale" in identity:
        _bounded_string(identity["rationale"], "bundle_identity.rationale", maximum=2_000)
    if "intent" in identity:
        _bounded_string(identity["intent"], "bundle_identity.intent", maximum=5_000)
    if "architecture_description" in identity:
        description = identity["architecture_description"].strip()
        if not description or len(description) > 20_000:
            raise ValueError(
                "bundle_identity.architecture_description must contain 1-20000 characters"
            )
        identity["architecture_description"] = description
    architecture_tags = identity.get("architecture_tags", [])
    if len(architecture_tags) > 30:
        raise ValueError("bundle_identity.architecture_tags cannot contain more than 30 values")
    if architecture_tags:
        identity["architecture_tags"] = [
            _bounded_string(
                tag,
                "bundle_identity.architecture_tags",
                maximum=100,
                require_non_empty=True,
                trim=True,
            )
            for tag in architecture_tags
        ]
    if "commit_hash" in identity:
        identity["commit_hash"] = _bounded_string(
            identity["commit_hash"],
            "bundle_identity.commit_hash",
            maximum=128,
            require_non_empty=True,
            trim=True,
        )
    return identity


def _normalize_finalize_request(body: FinalizeSubmissionRequest) -> FinalizeSubmissionRequest:
    unknown = set(body) - {
        "bundle_identity",
        "implementation_reference_id",
        "predictions",
        "execution_refs",
        "metadata",
    }
    if unknown:
        raise ValueError(f"finalize request contains unsupported fields: {sorted(unknown)}")

    identity = _normalize_bundle_identity(body["bundle_identity"])

    predictions = [
        _validate_prediction(prediction, index)
        for index, prediction in enumerate(body["predictions"])
    ]
    if not predictions or len(predictions) > 10_000:
        raise ValueError("predictions must contain 1-10000 terminal predictions")
    case_ids = [prediction["revision_case_id"] for prediction in predictions]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("predictions must contain each revision_case_id exactly once")
    artifact_ids = [
        artifact_id for prediction in predictions for artifact_id in prediction["artifact_ids"]
    ]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("each artifact_id can be attached to only one prediction")

    result: FinalizeSubmissionRequest = {
        "bundle_identity": identity,
        "predictions": predictions,
    }
    implementation_reference_id = body.get("implementation_reference_id")
    if implementation_reference_id is not None:
        _require_uuid(implementation_reference_id, "implementation_reference_id")
        result["implementation_reference_id"] = implementation_reference_id
    refs = _validate_execution_refs(body.get("execution_refs"), "execution_refs")
    if refs is not None:
        result["execution_refs"] = refs
    if "metadata" in body:
        if not isinstance(body["metadata"], dict):
            raise ValueError("metadata must be an object")
        result["metadata"] = body["metadata"]
    return result


def _api_error(response: httpx.Response) -> AgentGymAPIError:
    try:
        payload = response.json()
    except Exception:
        return AgentGymAPIError(
            response.status_code,
            "request_failed",
            "Agent Gym API request failed",
        )
    if not isinstance(payload, dict):
        return AgentGymAPIError(
            response.status_code,
            "request_failed",
            "Agent Gym API request failed",
        )
    code = str(payload.get("error") or "request_failed")
    message = payload.get("message")
    details = payload.get("details")
    return AgentGymAPIError(
        response.status_code,
        code,
        str(message) if message is not None else code,
        details if isinstance(details, dict) else None,
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_artifact(path: Path) -> bytes:
    size = path.stat().st_size
    if size > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact files cannot exceed {_MAX_ARTIFACT_BYTES} bytes")
    return path.read_bytes()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _existing_file_matches(path: Path, expected_size: int, expected_sha256: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if expected_sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _materialized_path(root: Path, case: ManifestCase, input_file: ManifestInputFile) -> Path:
    logical_path = _validate_artifact_path(input_file["path"])
    candidate = root / f"case-{case['ordinal']:06d}" / "inputs" / logical_path
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("manifest input path escapes the materialization root")
    return candidate


def _sanitized_materialized_manifest(
    manifest: ExternalSubmissionManifest,
    files: Sequence[MaterializedInputFile],
    root: Path,
) -> dict[str, Any]:
    local_paths = {
        (item.case_id, item.artifact_id, item.logical_path): str(item.local_path.relative_to(root))
        for item in files
    }
    cases: list[dict[str, Any]] = []
    for case in manifest["data"]:
        input_files: list[dict[str, Any]] = []
        for input_file in case["input_files"]:
            key = (case["case_id"], input_file["artifact_id"], input_file["path"])
            input_files.append(
                {
                    "artifact_id": input_file["artifact_id"],
                    "storage_object_id": input_file["storage_object_id"],
                    "path": input_file["path"],
                    "mime_type": input_file["mime_type"],
                    "size_bytes": input_file["size_bytes"],
                    "sha256": input_file["sha256"],
                    "local_path": local_paths[key],
                }
            )
        cases.append(
            {
                "case_id": case["case_id"],
                "ordinal": case["ordinal"],
                "input_payload": case["input_payload"],
                "input_files": input_files,
            }
        )
    return {
        "submission_id": manifest["submission_id"],
        "revision": manifest["revision"],
        "task": manifest["task"],
        "data": cases,
    }


def _validate_manifest_page_identity(first: ManifestPage, page: ManifestPage) -> None:
    if (
        page["submission_id"] != first["submission_id"]
        or page["revision"] != first["revision"]
        or page["task"] != first["task"]
    ):
        raise RuntimeError("manifest identity changed across immutable pages")


@dataclass
class AgentGymClient:
    """Synchronous client authenticated by an ``ags_`` submission token."""

    submission_token: str | None = field(default=None, repr=False)
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.Client = field(init=False, repr=False)
    _direct_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize authenticated API and unauthenticated direct-transfer clients."""
        self.submission_token = self.submission_token or os.environ.get("PROMPTIC_AGENT_GYM_TOKEN")
        if not self.submission_token:
            raise ValueError(
                "Agent Gym submission token required. Pass submission_token= or set "
                "PROMPTIC_AGENT_GYM_TOKEN."
            )
        if not self.submission_token.startswith("ags_"):
            raise ValueError("Agent Gym submission tokens must start with 'ags_'")
        self.endpoint = (
            self.endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
        ).rstrip("/")
        self._client = httpx.Client(
            base_url=f"{self.endpoint}/api/v1",
            headers={"Authorization": f"Bearer {self.submission_token}"},
            timeout=self.timeout,
        )
        self._direct_client = httpx.Client(timeout=self.timeout, follow_redirects=True)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: RequestParams | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = _UNSET,
    ) -> Any:
        kwargs: dict[str, Any] = {"params": params, "headers": headers}
        if json_body is not _UNSET:
            kwargs["json"] = json_body
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise _api_error(response)
        if response.status_code == 204:
            return None
        return response.json()

    def create_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        revision_id: str | None = None,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> ExternalSubmissionCreated:
        """Create or idempotently replay a revision-bound submission."""
        _require_uuid(benchmark_id, "benchmark_id")
        key = _require_idempotency_key(idempotency_key)
        if revision_id is not None:
            _require_uuid(revision_id, "revision_id")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be an integer")
        if not 300 <= ttl_seconds <= 7 * 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 300 and 604800")
        body: dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if revision_id is not None:
            body["revision_id"] = revision_id
        return self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions",
            headers={"Idempotency-Key": key},
            json_body=body,
        )

    def start_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        revision_id: str | None = None,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> ExternalSubmissionSession:
        """Create a submission and return an ergonomic bound session."""
        created = self.create_submission(
            benchmark_id,
            idempotency_key=idempotency_key,
            revision_id=revision_id,
            ttl_seconds=ttl_seconds,
        )
        return ExternalSubmissionSession(
            client=self,
            benchmark_id=benchmark_id,
            submission_id=created["submission_id"],
            revision_id=created["revision"]["id"],
            created_response=created,
        )

    def resume_submission(self, benchmark_id: str, submission_id: str) -> ExternalSubmissionSession:
        """Bind a session wrapper to an existing submission."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return ExternalSubmissionSession(
            client=self,
            benchmark_id=benchmark_id,
            submission_id=submission_id,
        )

    def get_manifest_page(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ManifestPage:
        """Fetch one immutable manifest page."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        return self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/manifest",
            params=params,
        )

    def iter_manifest_cases(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> Iterator[ManifestCase]:
        """Iterate every immutable revision case across cursor pages."""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            page = self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            yield from page["data"]
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen_cursors:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen_cursors.add(cursor)

    def get_manifest(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> ExternalSubmissionManifest:
        """Collect every immutable manifest page into one response."""
        first = self.get_manifest_page(benchmark_id, submission_id, limit=page_size)
        cases = list(first["data"])
        cursor = first["next_cursor"]
        seen_cursors: set[str] = set()
        while cursor is not None:
            if cursor in seen_cursors:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen_cursors.add(cursor)
            page = self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            _validate_manifest_page_identity(first, page)
            cases.extend(page["data"])
            cursor = page["next_cursor"]
        if len(cases) != first["revision"]["case_count"]:
            raise RuntimeError("manifest case count does not match the frozen revision declaration")
        return {
            "submission_id": first["submission_id"],
            "revision": first["revision"],
            "task": first["task"],
            "data": cases,
            "next_cursor": None,
        }

    def _download_manifest_input(
        self,
        input_file: ManifestInputFile,
        destination: Path,
        *,
        overwrite: bool,
    ) -> None:
        if destination.exists() and not overwrite:
            if _existing_file_matches(destination, input_file["size_bytes"], input_file["sha256"]):
                return
            raise FileExistsError(f"refusing to replace existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        observed_size = 0
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as output:
                try:
                    with self._direct_client.stream("GET", input_file["download_url"]) as response:
                        if response.status_code >= 400:
                            raise ArtifactUploadError(
                                response.status_code,
                                "Manifest input download failed",
                            )
                        for chunk in response.iter_bytes():
                            observed_size += len(chunk)
                            if observed_size > input_file["size_bytes"]:
                                raise ArtifactIntegrityError(
                                    "Manifest input exceeds its declared size"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                except httpx.HTTPError:
                    raise ArtifactUploadError(None, "Manifest input download failed") from None
                output.flush()
                os.fsync(output.fileno())
            if observed_size != input_file["size_bytes"]:
                raise ArtifactIntegrityError("Manifest input size does not match its declaration")
            expected_digest = input_file["sha256"]
            if expected_digest is not None and digest.hexdigest() != expected_digest:
                raise ArtifactIntegrityError("Manifest input digest does not match its declaration")
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def materialize_manifest(
        self,
        benchmark_id: str,
        submission_id: str,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download the manifest and all case inputs under a safe local root."""
        manifest = self.get_manifest(benchmark_id, submission_id, page_size=page_size)
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = _materialized_path(root, case, input_file)
                self._download_manifest_input(input_file, local_path, overwrite=overwrite)
                files.append(
                    MaterializedInputFile(
                        case_id=case["case_id"],
                        artifact_id=input_file["artifact_id"],
                        logical_path=input_file["path"],
                        local_path=local_path,
                        mime_type=input_file["mime_type"],
                        size_bytes=input_file["size_bytes"],
                        sha256=input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        sanitized = _sanitized_materialized_manifest(manifest, files, root)
        _atomic_write_bytes(
            manifest_path,
            (json.dumps(sanitized, indent=2, sort_keys=True) + "\n").encode(),
        )
        return MaterializedManifest(
            root=root,
            manifest_path=manifest_path,
            files=tuple(files),
            manifest=manifest,
        )

    def reserve_artifact(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        role: str = "output",
    ) -> ReservedSubmissionArtifact:
        """Reserve a submission-owned output artifact and direct upload target."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        logical_path = _validate_artifact_path(path)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError("size_bytes must be an integer")
        if not 0 <= size_bytes <= _MAX_ARTIFACT_BYTES:
            raise ValueError(f"size_bytes must be between 0 and {_MAX_ARTIFACT_BYTES}")
        if not mime_type.strip() or len(mime_type.strip()) > 200:
            raise ValueError("mime_type must contain 1-200 characters")
        if not role.strip() or len(role.strip()) > 100:
            raise ValueError("role must contain 1-100 characters")
        return self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts",
            json_body={
                "path": logical_path,
                "role": role.strip(),
                "mime_type": mime_type.strip(),
                "size_bytes": size_bytes,
                "sha256": _validate_sha256(sha256),
            },
        )

    def upload_reserved_artifact(
        self,
        reservation: ReservedSubmissionArtifact,
        content: bytes,
        *,
        mime_type: str,
        filename: str | None = None,
    ) -> None:
        """Upload bytes using exactly the reservation's direct upload descriptor."""
        upload = reservation["upload"]
        if len(content) > upload["maxSizeBytes"]:
            raise ValueError("artifact content exceeds the reserved upload size")
        headers = dict(upload.get("headers", {}))
        try:
            if upload["method"] == "PUT":
                response = self._direct_client.put(
                    upload["uploadUrl"], content=content, headers=headers
                )
            else:
                response = self._direct_client.post(
                    upload["uploadUrl"],
                    data=upload.get("fields", {}),
                    files={
                        "file": (
                            filename or Path(reservation["path"]).name,
                            content,
                            mime_type,
                        )
                    },
                    headers=headers,
                )
        except httpx.HTTPError:
            raise ArtifactUploadError(None, "Submission artifact upload failed") from None
        if response.status_code >= 400:
            raise ArtifactUploadError(
                response.status_code,
                "Submission artifact upload failed",
            )

    def complete_artifact(
        self, benchmark_id: str, submission_id: str, artifact_id: str
    ) -> CompletedSubmissionArtifact:
        """Ask the server to verify an upload's size, MIME type, and digest."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        _require_uuid(artifact_id, "artifact_id")
        return self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts/"
            f"{artifact_id}/complete",
        )

    def upload_artifact_bytes(
        self,
        benchmark_id: str,
        submission_id: str,
        content: bytes,
        *,
        path: str,
        mime_type: str,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Reserve, upload, and verify an in-memory output artifact."""
        reservation = self.reserve_artifact(
            benchmark_id,
            submission_id,
            path=path,
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=_sha256_bytes(content),
            role=role,
        )
        self.upload_reserved_artifact(
            reservation,
            content,
            mime_type=mime_type,
            filename=Path(path).name,
        )
        completed = self.complete_artifact(benchmark_id, submission_id, reservation["artifact_id"])
        return {
            "artifact_id": reservation["artifact_id"],
            "storage_object_id": reservation["storage_object_id"],
            "path": reservation["path"],
            "status": completed["status"],
        }

    def upload_artifact_file(
        self,
        benchmark_id: str,
        submission_id: str,
        source: str | os.PathLike[str],
        *,
        path: str | None = None,
        mime_type: str | None = None,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Reserve, upload, and verify a local output file."""
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        content = _read_artifact(source_path)
        logical_path = path or source_path.name
        resolved_mime_type = (
            mime_type or mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        )
        return self.upload_artifact_bytes(
            benchmark_id,
            submission_id,
            content,
            path=logical_path,
            mime_type=resolved_mime_type,
            role=role,
        )

    def resolve_traces(
        self, benchmark_id: str, submission_id: str, trace_ids: Sequence[str]
    ) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs to database UUIDs accepted by finalization."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        normalized = _normalize_trace_ids(trace_ids)
        return self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/traces",
            params=[("trace_id", trace_id) for trace_id in normalized],
        )

    def wait_for_resolved_traces(
        self,
        benchmark_id: str,
        submission_id: str,
        trace_ids: Sequence[str],
        *,
        max_wait: float = 30,
        poll_interval: float = 0.5,
    ) -> list[str]:
        """Poll ingestion until all raw OTEL IDs resolve, then return database UUIDs."""
        normalized = _normalize_trace_ids(trace_ids)
        deadline = time.monotonic() + max_wait
        while True:
            resolution = self.resolve_traces(benchmark_id, submission_id, normalized)
            by_trace_id = {item["trace_id"]: item["trace_db_id"] for item in resolution["data"]}
            unresolved = [trace_id for trace_id in normalized if by_trace_id.get(trace_id) is None]
            if not unresolved:
                return [cast(str, by_trace_id[trace_id]) for trace_id in normalized]
            if time.monotonic() >= deadline:
                raise UnresolvedTraceError(unresolved)
            time.sleep(poll_interval)

    def finalize_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        body: FinalizeSubmissionRequest,
        *,
        idempotency_key: str,
    ) -> FinalizedSubmission:
        """Finalize exactly one terminal prediction per manifest revision case."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        normalized = _normalize_finalize_request(body)
        return self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/finalize",
            headers={"Idempotency-Key": _require_idempotency_key(idempotency_key)},
            json_body=normalized,
        )

    def get_submission_status(self, benchmark_id: str, submission_id: str) -> SubmissionStatus:
        """Fetch submission status and its leaderboard benchmark-run linkage."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}",
        )

    def wait_for_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        max_wait: float = 600,
        poll_interval: float = 2,
    ) -> SubmissionStatus:
        """Poll until submission scoring succeeds or reaches another terminal state."""
        deadline = time.monotonic() + max_wait
        while True:
            status = self.get_submission_status(benchmark_id, submission_id)
            if status["status"] in _TERMINAL_SUBMISSION_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Submission {submission_id} did not complete within {max_wait}s "
                    f"(last status: {status['status']})"
                )
            time.sleep(poll_interval)

    def cancel_submission(self, benchmark_id: str, submission_id: str) -> CancelledSubmission:
        """Cancel a submission that is still in ``created`` or ``uploading`` state."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return self._request(
            "DELETE",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}",
        )

    def close(self) -> None:
        """Close API and direct-transfer HTTP clients."""
        self._client.close()
        self._direct_client.close()

    def __enter__(self) -> AgentGymClient:
        """Support use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close both HTTP clients on context manager exit."""
        self.close()


@dataclass
class ExternalSubmissionSession:
    """Synchronous client bound to one benchmark submission."""

    client: AgentGymClient = field(repr=False)
    benchmark_id: str
    submission_id: str
    revision_id: str | None = None
    created_response: ExternalSubmissionCreated | None = field(default=None, repr=False)

    def get_manifest_page(self, *, cursor: str | None = None, limit: int = 50) -> ManifestPage:
        """Fetch one manifest page."""
        return self.client.get_manifest_page(
            self.benchmark_id,
            self.submission_id,
            cursor=cursor,
            limit=limit,
        )

    def iter_cases(self, *, page_size: int = 100) -> Iterator[ManifestCase]:
        """Iterate every frozen revision case."""
        return self.client.iter_manifest_cases(
            self.benchmark_id, self.submission_id, page_size=page_size
        )

    def get_manifest(self, *, page_size: int = 100) -> ExternalSubmissionManifest:
        """Collect every manifest page."""
        return self.client.get_manifest(self.benchmark_id, self.submission_id, page_size=page_size)

    def materialize_manifest(
        self,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download the manifest and all case input files."""
        return self.client.materialize_manifest(
            self.benchmark_id,
            self.submission_id,
            destination,
            page_size=page_size,
            overwrite=overwrite,
        )

    def upload_artifact_bytes(
        self,
        content: bytes,
        *,
        path: str,
        mime_type: str,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Upload and verify bytes as a case output artifact."""
        return self.client.upload_artifact_bytes(
            self.benchmark_id,
            self.submission_id,
            content,
            path=path,
            mime_type=mime_type,
            role=role,
        )

    def upload_artifact_file(
        self,
        source: str | os.PathLike[str],
        *,
        path: str | None = None,
        mime_type: str | None = None,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Upload and verify a local case output file."""
        return self.client.upload_artifact_file(
            self.benchmark_id,
            self.submission_id,
            source,
            path=path,
            mime_type=mime_type,
            role=role,
        )

    def resolve_traces(self, trace_ids: Sequence[str]) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs without polling."""
        return self.client.resolve_traces(self.benchmark_id, self.submission_id, trace_ids)

    def wait_for_resolved_traces(
        self,
        trace_ids: Sequence[str],
        *,
        max_wait: float = 30,
        poll_interval: float = 0.5,
    ) -> list[str]:
        """Wait for trace ingestion and return database UUIDs."""
        return self.client.wait_for_resolved_traces(
            self.benchmark_id,
            self.submission_id,
            trace_ids,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    def finalize(
        self,
        *,
        bundle_identity: BundleIdentity,
        predictions: list[ExternalPrediction],
        idempotency_key: str,
        implementation_reference_id: str | None = None,
        execution_refs: ExecutionRefs | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FinalizedSubmission:
        """Build and send a terminal finalization request."""
        body: FinalizeSubmissionRequest = {
            "bundle_identity": bundle_identity,
            "predictions": predictions,
        }
        if implementation_reference_id is not None:
            body["implementation_reference_id"] = implementation_reference_id
        if execution_refs is not None:
            body["execution_refs"] = execution_refs
        if metadata is not None:
            body["metadata"] = metadata
        return self.client.finalize_submission(
            self.benchmark_id,
            self.submission_id,
            body,
            idempotency_key=idempotency_key,
        )

    def status(self) -> SubmissionStatus:
        """Fetch submission and leaderboard-run state."""
        return self.client.get_submission_status(self.benchmark_id, self.submission_id)

    def wait(self, *, max_wait: float = 600, poll_interval: float = 2) -> SubmissionStatus:
        """Wait for a terminal submission state."""
        return self.client.wait_for_submission(
            self.benchmark_id,
            self.submission_id,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    def cancel(self) -> CancelledSubmission:
        """Cancel this submission before finalization."""
        return self.client.cancel_submission(self.benchmark_id, self.submission_id)


@dataclass
class AsyncAgentGymClient:
    """Asynchronous client authenticated by an ``ags_`` submission token."""

    submission_token: str | None = field(default=None, repr=False)
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _direct_client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize authenticated API and unauthenticated direct-transfer clients."""
        self.submission_token = self.submission_token or os.environ.get("PROMPTIC_AGENT_GYM_TOKEN")
        if not self.submission_token:
            raise ValueError(
                "Agent Gym submission token required. Pass submission_token= or set "
                "PROMPTIC_AGENT_GYM_TOKEN."
            )
        if not self.submission_token.startswith("ags_"):
            raise ValueError("Agent Gym submission tokens must start with 'ags_'")
        self.endpoint = (
            self.endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
        ).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=f"{self.endpoint}/api/v1",
            headers={"Authorization": f"Bearer {self.submission_token}"},
            timeout=self.timeout,
        )
        self._direct_client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: RequestParams | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = _UNSET,
    ) -> Any:
        kwargs: dict[str, Any] = {"params": params, "headers": headers}
        if json_body is not _UNSET:
            kwargs["json"] = json_body
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise _api_error(response)
        if response.status_code == 204:
            return None
        return response.json()

    async def create_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        revision_id: str | None = None,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> ExternalSubmissionCreated:
        """Create or idempotently replay a revision-bound submission."""
        _require_uuid(benchmark_id, "benchmark_id")
        key = _require_idempotency_key(idempotency_key)
        if revision_id is not None:
            _require_uuid(revision_id, "revision_id")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be an integer")
        if not 300 <= ttl_seconds <= 7 * 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 300 and 604800")
        body: dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if revision_id is not None:
            body["revision_id"] = revision_id
        return await self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions",
            headers={"Idempotency-Key": key},
            json_body=body,
        )

    async def start_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        revision_id: str | None = None,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> AsyncExternalSubmissionSession:
        """Create a submission and return an ergonomic bound session."""
        created = await self.create_submission(
            benchmark_id,
            idempotency_key=idempotency_key,
            revision_id=revision_id,
            ttl_seconds=ttl_seconds,
        )
        return AsyncExternalSubmissionSession(
            client=self,
            benchmark_id=benchmark_id,
            submission_id=created["submission_id"],
            revision_id=created["revision"]["id"],
            created_response=created,
        )

    def resume_submission(
        self, benchmark_id: str, submission_id: str
    ) -> AsyncExternalSubmissionSession:
        """Bind a session wrapper to an existing submission."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return AsyncExternalSubmissionSession(
            client=self,
            benchmark_id=benchmark_id,
            submission_id=submission_id,
        )

    async def get_manifest_page(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ManifestPage:
        """Fetch one immutable manifest page."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        return await self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/manifest",
            params=params,
        )

    async def iter_manifest_cases(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> AsyncIterator[ManifestCase]:
        """Iterate every immutable revision case across cursor pages."""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            page = await self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            for case in page["data"]:
                yield case
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen_cursors:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen_cursors.add(cursor)

    async def get_manifest(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> ExternalSubmissionManifest:
        """Collect every immutable manifest page into one response."""
        first = await self.get_manifest_page(benchmark_id, submission_id, limit=page_size)
        cases = list(first["data"])
        cursor = first["next_cursor"]
        seen_cursors: set[str] = set()
        while cursor is not None:
            if cursor in seen_cursors:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen_cursors.add(cursor)
            page = await self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            _validate_manifest_page_identity(first, page)
            cases.extend(page["data"])
            cursor = page["next_cursor"]
        if len(cases) != first["revision"]["case_count"]:
            raise RuntimeError("manifest case count does not match the frozen revision declaration")
        return {
            "submission_id": first["submission_id"],
            "revision": first["revision"],
            "task": first["task"],
            "data": cases,
            "next_cursor": None,
        }

    async def _download_manifest_input(
        self,
        input_file: ManifestInputFile,
        destination: Path,
        *,
        overwrite: bool,
    ) -> None:
        destination_exists = await asyncio.to_thread(destination.exists)
        if destination_exists and not overwrite:
            matches = await asyncio.to_thread(
                _existing_file_matches,
                destination,
                input_file["size_bytes"],
                input_file["sha256"],
            )
            if matches:
                return
            raise FileExistsError(f"refusing to replace existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        observed_size = 0
        digest = hashlib.sha256()
        output = cast(BinaryIO, await asyncio.to_thread(temporary_path.open, "wb"))
        try:
            try:
                async with self._direct_client.stream(
                    "GET", input_file["download_url"]
                ) as response:
                    if response.status_code >= 400:
                        raise ArtifactUploadError(
                            response.status_code,
                            "Manifest input download failed",
                        )
                    async for chunk in response.aiter_bytes():
                        observed_size += len(chunk)
                        if observed_size > input_file["size_bytes"]:
                            raise ArtifactIntegrityError("Manifest input exceeds its declared size")
                        digest.update(chunk)
                        await asyncio.to_thread(output.write, chunk)
            except httpx.HTTPError:
                raise ArtifactUploadError(None, "Manifest input download failed") from None
            await asyncio.to_thread(output.flush)
            await asyncio.to_thread(os.fsync, output.fileno())
            await asyncio.to_thread(output.close)
            if observed_size != input_file["size_bytes"]:
                raise ArtifactIntegrityError("Manifest input size does not match its declaration")
            expected_digest = input_file["sha256"]
            if expected_digest is not None and digest.hexdigest() != expected_digest:
                raise ArtifactIntegrityError("Manifest input digest does not match its declaration")
            await asyncio.to_thread(temporary_path.replace, destination)
        finally:
            if not output.closed:
                await asyncio.to_thread(output.close)
            await asyncio.to_thread(temporary_path.unlink, missing_ok=True)

    async def materialize_manifest(
        self,
        benchmark_id: str,
        submission_id: str,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download the manifest and all case inputs under a safe local root."""
        manifest = await self.get_manifest(benchmark_id, submission_id, page_size=page_size)
        root = Path(destination)
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = _materialized_path(root, case, input_file)
                await self._download_manifest_input(input_file, local_path, overwrite=overwrite)
                files.append(
                    MaterializedInputFile(
                        case_id=case["case_id"],
                        artifact_id=input_file["artifact_id"],
                        logical_path=input_file["path"],
                        local_path=local_path,
                        mime_type=input_file["mime_type"],
                        size_bytes=input_file["size_bytes"],
                        sha256=input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        sanitized = _sanitized_materialized_manifest(manifest, files, root)
        content = (json.dumps(sanitized, indent=2, sort_keys=True) + "\n").encode()
        await asyncio.to_thread(_atomic_write_bytes, manifest_path, content)
        return MaterializedManifest(
            root=root,
            manifest_path=manifest_path,
            files=tuple(files),
            manifest=manifest,
        )

    async def reserve_artifact(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        role: str = "output",
    ) -> ReservedSubmissionArtifact:
        """Reserve a submission-owned output artifact and direct upload target."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        logical_path = _validate_artifact_path(path)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError("size_bytes must be an integer")
        if not 0 <= size_bytes <= _MAX_ARTIFACT_BYTES:
            raise ValueError(f"size_bytes must be between 0 and {_MAX_ARTIFACT_BYTES}")
        if not mime_type.strip() or len(mime_type.strip()) > 200:
            raise ValueError("mime_type must contain 1-200 characters")
        if not role.strip() or len(role.strip()) > 100:
            raise ValueError("role must contain 1-100 characters")
        return await self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts",
            json_body={
                "path": logical_path,
                "role": role.strip(),
                "mime_type": mime_type.strip(),
                "size_bytes": size_bytes,
                "sha256": _validate_sha256(sha256),
            },
        )

    async def upload_reserved_artifact(
        self,
        reservation: ReservedSubmissionArtifact,
        content: bytes,
        *,
        mime_type: str,
        filename: str | None = None,
    ) -> None:
        """Upload bytes using exactly the reservation's direct upload descriptor."""
        upload = reservation["upload"]
        if len(content) > upload["maxSizeBytes"]:
            raise ValueError("artifact content exceeds the reserved upload size")
        headers = dict(upload.get("headers", {}))
        try:
            if upload["method"] == "PUT":
                response = await self._direct_client.put(
                    upload["uploadUrl"], content=content, headers=headers
                )
            else:
                response = await self._direct_client.post(
                    upload["uploadUrl"],
                    data=upload.get("fields", {}),
                    files={
                        "file": (
                            filename or Path(reservation["path"]).name,
                            content,
                            mime_type,
                        )
                    },
                    headers=headers,
                )
        except httpx.HTTPError:
            raise ArtifactUploadError(None, "Submission artifact upload failed") from None
        if response.status_code >= 400:
            raise ArtifactUploadError(
                response.status_code,
                "Submission artifact upload failed",
            )

    async def complete_artifact(
        self, benchmark_id: str, submission_id: str, artifact_id: str
    ) -> CompletedSubmissionArtifact:
        """Ask the server to verify an upload's size, MIME type, and digest."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        _require_uuid(artifact_id, "artifact_id")
        return await self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts/"
            f"{artifact_id}/complete",
        )

    async def upload_artifact_bytes(
        self,
        benchmark_id: str,
        submission_id: str,
        content: bytes,
        *,
        path: str,
        mime_type: str,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Reserve, upload, and verify an in-memory output artifact."""
        reservation = await self.reserve_artifact(
            benchmark_id,
            submission_id,
            path=path,
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=_sha256_bytes(content),
            role=role,
        )
        await self.upload_reserved_artifact(
            reservation,
            content,
            mime_type=mime_type,
            filename=Path(path).name,
        )
        completed = await self.complete_artifact(
            benchmark_id, submission_id, reservation["artifact_id"]
        )
        return {
            "artifact_id": reservation["artifact_id"],
            "storage_object_id": reservation["storage_object_id"],
            "path": reservation["path"],
            "status": completed["status"],
        }

    async def upload_artifact_file(
        self,
        benchmark_id: str,
        submission_id: str,
        source: str | os.PathLike[str],
        *,
        path: str | None = None,
        mime_type: str | None = None,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Reserve, upload, and verify a local output file."""
        source_path = Path(source)
        if not await asyncio.to_thread(source_path.is_file):
            raise FileNotFoundError(source_path)
        content = await asyncio.to_thread(_read_artifact, source_path)
        logical_path = path or source_path.name
        resolved_mime_type = (
            mime_type or mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        )
        return await self.upload_artifact_bytes(
            benchmark_id,
            submission_id,
            content,
            path=logical_path,
            mime_type=resolved_mime_type,
            role=role,
        )

    async def resolve_traces(
        self, benchmark_id: str, submission_id: str, trace_ids: Sequence[str]
    ) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs to database UUIDs accepted by finalization."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        normalized = _normalize_trace_ids(trace_ids)
        return await self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/traces",
            params=[("trace_id", trace_id) for trace_id in normalized],
        )

    async def wait_for_resolved_traces(
        self,
        benchmark_id: str,
        submission_id: str,
        trace_ids: Sequence[str],
        *,
        max_wait: float = 30,
        poll_interval: float = 0.5,
    ) -> list[str]:
        """Poll ingestion until all raw OTEL IDs resolve, then return database UUIDs."""
        normalized = _normalize_trace_ids(trace_ids)
        deadline = time.monotonic() + max_wait
        while True:
            resolution = await self.resolve_traces(benchmark_id, submission_id, normalized)
            by_trace_id = {item["trace_id"]: item["trace_db_id"] for item in resolution["data"]}
            unresolved = [trace_id for trace_id in normalized if by_trace_id.get(trace_id) is None]
            if not unresolved:
                return [cast(str, by_trace_id[trace_id]) for trace_id in normalized]
            if time.monotonic() >= deadline:
                raise UnresolvedTraceError(unresolved)
            await asyncio.sleep(poll_interval)

    async def finalize_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        body: FinalizeSubmissionRequest,
        *,
        idempotency_key: str,
    ) -> FinalizedSubmission:
        """Finalize exactly one terminal prediction per manifest revision case."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        normalized = _normalize_finalize_request(body)
        return await self._request(
            "POST",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}/finalize",
            headers={"Idempotency-Key": _require_idempotency_key(idempotency_key)},
            json_body=normalized,
        )

    async def get_submission_status(
        self, benchmark_id: str, submission_id: str
    ) -> SubmissionStatus:
        """Fetch submission status and its leaderboard benchmark-run linkage."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return await self._request(
            "GET",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}",
        )

    async def wait_for_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        max_wait: float = 600,
        poll_interval: float = 2,
    ) -> SubmissionStatus:
        """Poll until submission scoring succeeds or reaches another terminal state."""
        deadline = time.monotonic() + max_wait
        while True:
            status = await self.get_submission_status(benchmark_id, submission_id)
            if status["status"] in _TERMINAL_SUBMISSION_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Submission {submission_id} did not complete within {max_wait}s "
                    f"(last status: {status['status']})"
                )
            await asyncio.sleep(poll_interval)

    async def cancel_submission(self, benchmark_id: str, submission_id: str) -> CancelledSubmission:
        """Cancel a submission that is still in ``created`` or ``uploading`` state."""
        _require_uuid(benchmark_id, "benchmark_id")
        _require_uuid(submission_id, "submission_id")
        return await self._request(
            "DELETE",
            f"/benchmarks/{benchmark_id}/submissions/{submission_id}",
        )

    async def close(self) -> None:
        """Close API and direct-transfer HTTP clients."""
        await self._client.aclose()
        await self._direct_client.aclose()

    async def __aenter__(self) -> AsyncAgentGymClient:
        """Support use as an asynchronous context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close both HTTP clients on asynchronous context manager exit."""
        await self.close()


@dataclass
class AsyncExternalSubmissionSession:
    """Asynchronous client bound to one benchmark submission."""

    client: AsyncAgentGymClient = field(repr=False)
    benchmark_id: str
    submission_id: str
    revision_id: str | None = None
    created_response: ExternalSubmissionCreated | None = field(default=None, repr=False)

    async def get_manifest_page(
        self, *, cursor: str | None = None, limit: int = 50
    ) -> ManifestPage:
        """Fetch one manifest page."""
        return await self.client.get_manifest_page(
            self.benchmark_id,
            self.submission_id,
            cursor=cursor,
            limit=limit,
        )

    def iter_cases(self, *, page_size: int = 100) -> AsyncIterator[ManifestCase]:
        """Iterate every frozen revision case."""
        return self.client.iter_manifest_cases(
            self.benchmark_id, self.submission_id, page_size=page_size
        )

    async def get_manifest(self, *, page_size: int = 100) -> ExternalSubmissionManifest:
        """Collect every manifest page."""
        return await self.client.get_manifest(
            self.benchmark_id, self.submission_id, page_size=page_size
        )

    async def materialize_manifest(
        self,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download the manifest and all case input files."""
        return await self.client.materialize_manifest(
            self.benchmark_id,
            self.submission_id,
            destination,
            page_size=page_size,
            overwrite=overwrite,
        )

    async def upload_artifact_bytes(
        self,
        content: bytes,
        *,
        path: str,
        mime_type: str,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Upload and verify bytes as a case output artifact."""
        return await self.client.upload_artifact_bytes(
            self.benchmark_id,
            self.submission_id,
            content,
            path=path,
            mime_type=mime_type,
            role=role,
        )

    async def upload_artifact_file(
        self,
        source: str | os.PathLike[str],
        *,
        path: str | None = None,
        mime_type: str | None = None,
        role: str = "output",
    ) -> SubmissionArtifact:
        """Upload and verify a local case output file."""
        return await self.client.upload_artifact_file(
            self.benchmark_id,
            self.submission_id,
            source,
            path=path,
            mime_type=mime_type,
            role=role,
        )

    async def resolve_traces(self, trace_ids: Sequence[str]) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs without polling."""
        return await self.client.resolve_traces(self.benchmark_id, self.submission_id, trace_ids)

    async def wait_for_resolved_traces(
        self,
        trace_ids: Sequence[str],
        *,
        max_wait: float = 30,
        poll_interval: float = 0.5,
    ) -> list[str]:
        """Wait for trace ingestion and return database UUIDs."""
        return await self.client.wait_for_resolved_traces(
            self.benchmark_id,
            self.submission_id,
            trace_ids,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    async def finalize(
        self,
        *,
        bundle_identity: BundleIdentity,
        predictions: list[ExternalPrediction],
        idempotency_key: str,
        implementation_reference_id: str | None = None,
        execution_refs: ExecutionRefs | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FinalizedSubmission:
        """Build and send a terminal finalization request."""
        body: FinalizeSubmissionRequest = {
            "bundle_identity": bundle_identity,
            "predictions": predictions,
        }
        if implementation_reference_id is not None:
            body["implementation_reference_id"] = implementation_reference_id
        if execution_refs is not None:
            body["execution_refs"] = execution_refs
        if metadata is not None:
            body["metadata"] = metadata
        return await self.client.finalize_submission(
            self.benchmark_id,
            self.submission_id,
            body,
            idempotency_key=idempotency_key,
        )

    async def status(self) -> SubmissionStatus:
        """Fetch submission and leaderboard-run state."""
        return await self.client.get_submission_status(self.benchmark_id, self.submission_id)

    async def wait(self, *, max_wait: float = 600, poll_interval: float = 2) -> SubmissionStatus:
        """Wait for a terminal submission state."""
        return await self.client.wait_for_submission(
            self.benchmark_id,
            self.submission_id,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    async def cancel(self) -> CancelledSubmission:
        """Cancel this submission before finalization."""
        return await self.client.cancel_submission(self.benchmark_id, self.submission_id)
