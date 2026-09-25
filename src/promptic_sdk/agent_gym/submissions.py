"""Shared Agent Gym submission validation and request construction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlsplit
from uuid import UUID

from promptic_sdk.agent_gym._http import RequestSpec
from promptic_sdk.agent_gym.models import (
    ExecutionRefs,
    ExternalPrediction,
    ExternalSubmissionManifest,
    ManifestCase,
    ManifestInputFile,
    ManifestPage,
    RevisionManifest,
    RevisionManifestPage,
    SubmitSubmissionRequest,
    VariantIdentity,
)

MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
MAX_TRACE_IDS = 100
MAX_PREDICTION_BATCH_ITEMS = 500
MAX_PREDICTION_BATCH_BYTES = 1024 * 1024
TERMINAL_SUBMISSION_STATUSES = {
    "succeeded",
    "failed",
    "dispatch_failed",
    "expired",
    "cancelled",
}
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TracePolicy: TypeAlias = Literal["required", "best_effort", "disabled"]


class ArtifactIntegrityError(Exception):
    """Artifact bytes do not match declared integrity metadata."""


class UnresolvedTraceError(Exception):
    """One or more OTEL trace IDs were not found before a deadline."""

    def __init__(
        self,
        trace_ids: Sequence[str],
        *,
        resolved: Mapping[str, str] | None = None,
        code: str = "trace_resolution_timeout",
        phase: str = "resolution",
        case_ids: Sequence[int] = (),
    ) -> None:
        """Initialize with unresolved raw OTEL trace IDs."""
        self.trace_ids = list(trace_ids)
        self.resolved = dict(resolved or {})
        self.code = code
        self.phase = phase
        self.case_ids = list(case_ids)
        if self.case_ids:
            detail = (
                f"{len(self.case_ids)} successful case(s) have no trace evidence: {self.case_ids}"
            )
        else:
            detail = f"{len(self.trace_ids)} trace ID(s) were not resolved"
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class MaterializedInputFile:
    """One manifest input downloaded to a safe local path."""

    dataset_case_id: int
    artifact_id: str
    field_path: str
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
    manifest: ExternalSubmissionManifest | RevisionManifest = field(repr=False)


@dataclass(frozen=True)
class MaterializedDatasetCase:
    """One public benchmark input with files materialized in schema position."""

    id: int
    ordinal: int
    input: dict[str, Any]


@dataclass(frozen=True)
class DownloadedBenchmarkDataset:
    """Persistent public-input package for architecture development."""

    root: Path
    manifest_path: Path
    revision: dict[str, Any]
    task: dict[str, Any]
    cases: tuple[MaterializedDatasetCase, ...]


def materialized_case_input(value: Any, files: Sequence[MaterializedInputFile]) -> dict[str, Any]:
    """Replace public artifact URIs with their verified local materializations."""
    artifacts = {f"promptic-artifact://{item.artifact_id}": item for item in files}

    def replace_value(item: Any) -> Any:
        if isinstance(item, str) and item in artifacts:
            return artifacts[item]
        if isinstance(item, Mapping):
            return {str(key): replace_value(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace_value(child) for child in item]
        return item

    resolved = replace_value(value)
    if not isinstance(resolved, dict):
        raise TypeError("case input payload must be an object")
    return resolved


def downloaded_dataset(materialized: MaterializedManifest) -> DownloadedBenchmarkDataset:
    """Build the public architecture-discovery resource from a safe manifest."""
    grouped: dict[int, list[MaterializedInputFile]] = {}
    for item in materialized.files:
        grouped.setdefault(item.dataset_case_id, []).append(item)
    cases = tuple(
        MaterializedDatasetCase(
            id=case["dataset_case_id"],
            ordinal=case["ordinal"],
            input=materialized_case_input(
                case["input_payload"], grouped.get(case["dataset_case_id"], [])
            ),
        )
        for case in materialized.manifest["data"]
    )
    return DownloadedBenchmarkDataset(
        root=materialized.root,
        manifest_path=materialized.manifest_path,
        revision=dict(cast(Mapping[str, Any], materialized.manifest["revision"])),
        task=dict(cast(Mapping[str, Any], materialized.manifest["task"])),
        cases=cases,
    )


def require_uuid(value: str, field_name: str) -> str:
    """Require a UUID-shaped API identifier."""
    try:
        UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    return value


def require_idempotency_key(value: str) -> str:
    """Validate and normalize an idempotency key."""
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise ValueError("idempotency_key must contain 1-200 non-whitespace characters")
    return normalized


def validate_artifact_path(value: str) -> str:
    """Require a normalized relative POSIX artifact path."""
    if not value or len(value) > 500 or value.startswith("/") or "\\" in value or "\0" in value:
        raise ValueError("artifact path must be a relative normalized POSIX path")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("artifact path must be a relative normalized POSIX path")
    return "/".join(parts)


def validate_sha256(value: str) -> str:
    """Require a lowercase SHA-256 digest."""
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
    return value


def normalize_trace_ids(trace_ids: Sequence[str]) -> list[str]:
    """Validate, lowercase, and deduplicate raw OTEL trace IDs."""
    normalized = list(dict.fromkeys(trace_id.lower() for trace_id in trace_ids))
    if not normalized or len(normalized) > MAX_TRACE_IDS:
        raise ValueError(f"trace_ids must contain 1-{MAX_TRACE_IDS} unique values")
    if any(not _TRACE_ID_RE.fullmatch(trace_id) for trace_id in normalized):
        raise ValueError("OTEL trace IDs must contain exactly 32 hexadecimal characters")
    return normalized


def normalize_trace_policy(value: str) -> TracePolicy:
    """Validate the submission-wide trace-linking policy."""
    if value not in {"required", "best_effort", "disabled"}:
        raise ValueError("trace_policy must be 'required', 'best_effort', or 'disabled'")
    return cast(TracePolicy, value)


def _bounded_string(
    value: str,
    field_name: str,
    maximum: int,
    *,
    non_empty: bool = False,
    trim: bool = False,
) -> str:
    normalized = value.strip() if trim else value
    if len(normalized) > maximum or (non_empty and not normalized):
        raise ValueError(
            f"{field_name} must contain {'1-' if non_empty else 'at most '}{maximum} characters"
        )
    return normalized


def _repository_url(value: str) -> str:
    """Validate a bounded HTTPS repository URL without embedded credentials."""
    normalized = _bounded_string(
        value,
        "variant_identity.repository_url",
        2_000,
        non_empty=True,
        trim=True,
    )
    try:
        parsed = urlsplit(normalized)
        # Accessing port performs urlsplit's deferred syntax and range validation.
        parsed.port
        valid = bool(
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not any(character.isspace() for character in normalized)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "variant_identity.repository_url must be a valid HTTPS URL without credentials"
        )
    return normalized


def _validate_refs(refs: ExecutionRefs | None, field_name: str) -> ExecutionRefs | None:
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
        unique = list(dict.fromkeys(values))
        if len(unique) > 100:
            raise ValueError(f"{field_name}.{key} cannot contain more than 100 IDs")
        for value in unique:
            require_uuid(value, f"{field_name}.{key}")
        result[key] = unique
    return result


def normalize_variant_identity(value: VariantIdentity) -> VariantIdentity:
    """Validate a candidate variant identity against the platform contract."""
    identity = cast(VariantIdentity, dict(value))
    identity["name"] = _bounded_string(
        identity["name"], "variant_identity.name", 100, non_empty=True, trim=True
    )
    identity["version"] = _bounded_string(
        identity["version"], "variant_identity.version", 50, non_empty=True, trim=True
    )
    if "parent_name" in identity:
        identity["parent_name"] = _bounded_string(
            identity["parent_name"],
            "variant_identity.parent_name",
            100,
            non_empty=True,
            trim=True,
        )
        if "parent_version" not in identity:
            raise ValueError("variant_identity.parent_version is required with parent_name")
    if "parent_version" in identity:
        identity["parent_version"] = _bounded_string(
            identity["parent_version"],
            "variant_identity.parent_version",
            50,
            non_empty=True,
            trim=True,
        )
    for key, maximum in (("rationale", 2_000), ("intent", 5_000)):
        if key in identity:
            _bounded_string(identity[key], f"variant_identity.{key}", maximum)
    if "architecture_description" in identity:
        identity["architecture_description"] = _bounded_string(
            identity["architecture_description"],
            "variant_identity.architecture_description",
            20_000,
            non_empty=True,
            trim=True,
        )
    tags = identity.get("architecture_tags", [])
    if len(tags) > 30:
        raise ValueError("variant_identity.architecture_tags cannot contain more than 30 values")
    if tags:
        identity["architecture_tags"] = [
            _bounded_string(
                tag,
                "variant_identity.architecture_tags",
                100,
                non_empty=True,
                trim=True,
            )
            for tag in tags
        ]
    if "repository_url" in identity:
        identity["repository_url"] = _repository_url(identity["repository_url"])
    if "commit_hash" in identity:
        identity["commit_hash"] = _bounded_string(
            identity["commit_hash"],
            "variant_identity.commit_hash",
            128,
            non_empty=True,
            trim=True,
        )
    return identity


def normalize_prediction(prediction: ExternalPrediction, index: int = 0) -> ExternalPrediction:
    """Validate and copy one terminal prediction."""
    result = cast(ExternalPrediction, dict(prediction))
    prefix = f"predictions[{index}]"
    case_id = result["dataset_case_id"]
    if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
        raise ValueError(f"{prefix}.dataset_case_id must be a positive integer")
    status = result["status"]
    if status not in {"succeeded", "failed", "skipped", "cancelled"}:
        raise ValueError(f"{prefix}.status is invalid")
    output = result.get("output")
    if status == "succeeded" and output is None:
        raise ValueError(f"{prefix}.output is required for succeeded predictions")
    if status != "succeeded" and output is not None:
        raise ValueError(f"{prefix}.output is only valid for succeeded predictions")
    if status != "succeeded" and not result.get("error_code"):
        raise ValueError(f"{prefix}.error_code is required for non-successful predictions")
    artifacts = result.get("artifact_ids", [])
    if len(artifacts) != len(set(artifacts)) or len(artifacts) > 20:
        raise ValueError(f"{prefix}.artifact_ids must contain at most 20 unique IDs")
    for artifact_id in artifacts:
        require_uuid(artifact_id, f"{prefix}.artifact_ids")
    result["artifact_ids"] = artifacts
    refs = _validate_refs(result.get("execution_refs"), f"{prefix}.execution_refs")
    if refs is not None:
        result["execution_refs"] = refs
    reference_id = result.get("implementation_reference_id")
    if reference_id is not None:
        require_uuid(reference_id, f"{prefix}.implementation_reference_id")
    for key in ("executor_id", "executor_version"):
        if key in result:
            _bounded_string(result[key], f"{prefix}.{key}", 200, non_empty=True)
    usage = result.get("token_usage")
    if usage is not None:
        for key in ("prompt", "completion", "total"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{prefix}.token_usage.{key} must be a non-negative integer")
    latency = result.get("latency_ms")
    if latency is not None and (
        isinstance(latency, bool) or not isinstance(latency, int) or latency < 0
    ):
        raise ValueError(f"{prefix}.latency_ms must be a non-negative integer")
    return result


def normalize_prediction_batch(
    predictions: Sequence[ExternalPrediction],
) -> list[ExternalPrediction]:
    """Validate one bounded idempotent prediction-upload batch."""
    normalized = [
        normalize_prediction(prediction, index) for index, prediction in enumerate(predictions)
    ]
    if not 1 <= len(normalized) <= MAX_PREDICTION_BATCH_ITEMS:
        raise ValueError(
            f"prediction batches must contain 1-{MAX_PREDICTION_BATCH_ITEMS} terminal predictions"
        )
    case_ids = [prediction["dataset_case_id"] for prediction in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("a prediction batch can contain each dataset_case_id only once")
    artifact_ids = [
        artifact_id for prediction in normalized for artifact_id in prediction["artifact_ids"]
    ]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("each artifact_id can be attached to only one prediction in a batch")
    if _prediction_batch_size(normalized) > MAX_PREDICTION_BATCH_BYTES:
        raise ValueError(
            "prediction batch exceeds 1 MiB; upload fewer predictions or store large outputs "
            "as artifacts"
        )
    return normalized


def _prediction_batch_size(predictions: Sequence[ExternalPrediction]) -> int:
    return len(
        json.dumps(
            {"predictions": list(predictions)},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def prediction_upload_batches(
    predictions: Sequence[ExternalPrediction],
) -> list[list[ExternalPrediction]]:
    """Split predictions by both API item count and serialized request size."""
    if not predictions:
        return []
    normalized = [
        normalize_prediction(prediction, index) for index, prediction in enumerate(predictions)
    ]
    case_ids = [prediction["dataset_case_id"] for prediction in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("predictions can contain each dataset_case_id only once")
    artifact_ids = [
        artifact_id for prediction in normalized for artifact_id in prediction["artifact_ids"]
    ]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("each artifact_id can be attached to only one prediction")

    batches: list[list[ExternalPrediction]] = []
    current: list[ExternalPrediction] = []
    for prediction in normalized:
        if _prediction_batch_size([prediction]) > MAX_PREDICTION_BATCH_BYTES:
            raise ValueError(
                "one prediction exceeds 1 MiB; store large output content as an artifact"
            )
        candidate = [*current, prediction]
        if current and (
            len(candidate) > MAX_PREDICTION_BATCH_ITEMS
            or _prediction_batch_size(candidate) > MAX_PREDICTION_BATCH_BYTES
        ):
            batches.append(current)
            current = [prediction]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def normalize_submit_request(body: SubmitSubmissionRequest) -> SubmitSubmissionRequest:
    """Validate optional evidence metadata used when requesting scoring."""
    unknown = set(body) - {
        "variant_identity",
        "implementation_reference_id",
        "execution_refs",
        "metadata",
    }
    if unknown:
        raise ValueError(f"submit request contains unsupported fields: {sorted(unknown)}")
    result: SubmitSubmissionRequest = {}
    if "variant_identity" in body:
        result["variant_identity"] = normalize_variant_identity(body["variant_identity"])
    reference_id = body.get("implementation_reference_id")
    if reference_id is not None:
        require_uuid(reference_id, "implementation_reference_id")
        result["implementation_reference_id"] = reference_id
    refs = _validate_refs(body.get("execution_refs"), "execution_refs")
    if refs is not None:
        result["execution_refs"] = refs
    if "metadata" in body:
        if not isinstance(body["metadata"], dict):
            raise ValueError("metadata must be an object")
        result["metadata"] = body["metadata"]
    return result


class SubmissionPredictionBuilder:
    """Shared state and protocol validation for sync and async session builders."""

    def __init__(self) -> None:
        """Initialize an empty resumable client-side builder."""
        self.manifest: ExternalSubmissionManifest | None = None
        self.predictions: dict[int, ExternalPrediction] = {}
        self.artifact_ids: set[str] = set()
        self.raw_trace_ids: dict[int, tuple[str, ...]] = {}

    def set_manifest(self, manifest: ExternalSubmissionManifest) -> None:
        """Cache the immutable manifest used to validate prediction uploads."""
        self.manifest = manifest

    def validate_case(self, case_id: int) -> None:
        """Require a manifest-owned case without an existing prediction."""
        if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
            raise ValueError("case_id must be a positive integer")
        if self.manifest is None:
            raise RuntimeError("the submission manifest has not been loaded")
        if case_id not in {case["dataset_case_id"] for case in self.manifest["data"]}:
            raise ValueError("case_id does not belong to this submission's manifest")
        if case_id in self.predictions:
            raise ValueError("case_id already has a local prediction")

    def stage(
        self,
        case_id: int,
        prediction: ExternalPrediction,
        *,
        raw_trace_ids: Sequence[str] = (),
    ) -> ExternalPrediction:
        """Validate and stage one prediction after its evidence is uploaded."""
        self.validate_case(case_id)
        artifact_ids = prediction.get("artifact_ids", [])
        if self.artifact_ids.intersection(artifact_ids):
            raise ValueError("an artifact can be attached to only one prediction")
        normalized = normalize_prediction(prediction, len(self.predictions))
        self.predictions[case_id] = normalized
        self.artifact_ids.update(artifact_ids)
        self.raw_trace_ids[case_id] = (
            tuple(normalize_trace_ids(raw_trace_ids)) if raw_trace_ids else ()
        )
        return normalized

    def pending_trace_ids(self) -> list[str]:
        """Return unique raw trace IDs in immutable manifest order."""
        if self.manifest is None:
            raise RuntimeError("the submission manifest has not been loaded")
        return list(
            dict.fromkeys(
                trace_id
                for case in self.manifest["data"]
                for trace_id in self.raw_trace_ids.get(case["dataset_case_id"], ())
            )
        )

    def validate_trace_coverage(self, policy: TracePolicy) -> None:
        """An explicit required policy needs a trace for each successful case.

        This is a caller opt-in, not an inference from evaluator weights. The
        server remains authoritative for revision-specific evidence requirements.
        """
        if policy != "required":
            return
        missing = [
            case_id
            for case_id, prediction in self.predictions.items()
            if prediction["status"] == "succeeded"
            and not self.raw_trace_ids.get(case_id)
            and not (prediction.get("execution_refs") or {}).get("trace_ids")
        ]
        if missing:
            raise UnresolvedTraceError(
                [], code="required_trace_missing", phase="validation", case_ids=missing
            )

    def validate_coverage(self) -> list[int]:
        """Return manifest case IDs after validating exact prediction coverage."""
        if self.manifest is None:
            raise RuntimeError("the submission manifest has not been loaded")
        ordered_ids = [case["dataset_case_id"] for case in self.manifest["data"]]
        expected_ids = set(ordered_ids)
        missing = [case_id for case_id in ordered_ids if case_id not in self.predictions]
        extra = [case_id for case_id in self.predictions if case_id not in expected_ids]
        if missing or extra:
            raise ValueError(
                "predictions must cover every manifest case exactly once "
                f"(missing={missing}, extra={extra})"
            )
        return ordered_ids

    def submission_payloads(
        self,
        identity: VariantIdentity | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        trace_policy: TracePolicy = "best_effort",
        resolved_trace_ids: Mapping[str, str] | None = None,
    ) -> tuple[list[ExternalPrediction], SubmitSubmissionRequest]:
        """Compile exact manifest coverage into prediction batches and a scoring request."""
        ordered_ids = self.validate_coverage()
        policy = normalize_trace_policy(trace_policy)
        self.validate_trace_coverage(policy)
        resolved = {key.lower(): value for key, value in (resolved_trace_ids or {}).items()}
        unresolved = [trace_id for trace_id in self.pending_trace_ids() if trace_id not in resolved]
        if policy == "required" and unresolved:
            raise UnresolvedTraceError(unresolved, resolved=resolved)

        predictions: list[ExternalPrediction] = []
        for case_id in ordered_ids:
            prediction = cast(ExternalPrediction, dict(self.predictions[case_id]))
            if policy == "disabled":
                prediction.pop("execution_refs", None)
            else:
                refs = cast(ExecutionRefs, dict(prediction.get("execution_refs") or {}))
                linked = [
                    resolved[trace_id]
                    for trace_id in self.raw_trace_ids.get(case_id, ())
                    if trace_id in resolved
                ]
                if linked:
                    refs["trace_ids"] = list(dict.fromkeys([*refs.get("trace_ids", []), *linked]))
                if refs:
                    prediction["execution_refs"] = refs
                else:
                    prediction.pop("execution_refs", None)
            predictions.append(prediction)

        body: SubmitSubmissionRequest = {}
        if identity is not None:
            body["variant_identity"] = identity
        if metadata is not None:
            body["metadata"] = metadata
        return normalize_prediction_batch(predictions), normalize_submit_request(body)


def create_submission_request(
    benchmark_id: str,
    *,
    idempotency_key: str,
    variant_identity: VariantIdentity,
    revision_id: str | None,
    ttl_seconds: int,
) -> RequestSpec:
    """Build a create-submission request."""
    require_uuid(benchmark_id, "benchmark_id")
    if revision_id is not None:
        require_uuid(revision_id, "revision_id")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise ValueError("ttl_seconds must be an integer")
    if not 300 <= ttl_seconds <= 604_800:
        raise ValueError("ttl_seconds must be between 300 and 604800")
    body: dict[str, Any] = {
        "ttl_seconds": ttl_seconds,
        "variant_identity": normalize_variant_identity(variant_identity),
    }
    if revision_id is not None:
        body["revision_id"] = revision_id
    return RequestSpec(
        "POST",
        f"/benchmarks/{benchmark_id}/submissions",
        json=body,
        headers={"Idempotency-Key": require_idempotency_key(idempotency_key)},
    )


def manifest_page_request(
    benchmark_id: str,
    submission_id: str,
    *,
    cursor: str | None,
    limit: int,
) -> RequestSpec:
    """Build one manifest-page request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    params: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        params["cursor"] = cursor
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/manifest",
        params=params,
    )


def revision_manifest_page_request(
    benchmark_id: str,
    revision_id: str | None,
    *,
    cursor: str | None,
    limit: int,
) -> RequestSpec:
    """Build one read-only immutable revision input request."""
    require_uuid(benchmark_id, "benchmark_id")
    selected_revision = (
        "current" if revision_id is None else require_uuid(revision_id, "revision_id")
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    params: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        params["cursor"] = cursor
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/revisions/{selected_revision}/manifest",
        params=params,
    )


def reserve_artifact_request(
    benchmark_id: str,
    submission_id: str,
    *,
    path: str,
    mime_type: str,
    size_bytes: int,
    sha256: str,
    role: str,
) -> RequestSpec:
    """Build an output-artifact reservation request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise ValueError("size_bytes must be an integer")
    if not 0 <= size_bytes <= MAX_ARTIFACT_BYTES:
        raise ValueError(f"size_bytes must be between 0 and {MAX_ARTIFACT_BYTES}")
    mime_type = _bounded_string(mime_type, "mime_type", 200, non_empty=True, trim=True)
    role = _bounded_string(role, "role", 100, non_empty=True, trim=True)
    return RequestSpec(
        "POST",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts",
        json={
            "path": validate_artifact_path(path),
            "role": role,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "sha256": validate_sha256(sha256),
        },
    )


def complete_artifact_request(
    benchmark_id: str, submission_id: str, artifact_id: str
) -> RequestSpec:
    """Build a server-side artifact verification request."""
    for value, name in (
        (benchmark_id, "benchmark_id"),
        (submission_id, "submission_id"),
        (artifact_id, "artifact_id"),
    ):
        require_uuid(value, name)
    return RequestSpec(
        "POST",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/artifacts/{artifact_id}/complete",
    )


def resolve_traces_request(
    benchmark_id: str, submission_id: str, trace_ids: Sequence[str]
) -> RequestSpec:
    """Build a submission-scoped trace resolver request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/traces",
        params=[("trace_id", trace_id) for trace_id in normalize_trace_ids(trace_ids)],
    )


def upload_prediction_batch_request(
    benchmark_id: str,
    submission_id: str,
    predictions: Sequence[ExternalPrediction],
) -> RequestSpec:
    """Build an idempotent canonical-prediction batch request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    return RequestSpec(
        "PUT",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/predictions",
        json={"predictions": normalize_prediction_batch(predictions)},
    )


def submit_submission_request(
    benchmark_id: str,
    submission_id: str,
    body: SubmitSubmissionRequest,
    *,
    idempotency_key: str,
) -> RequestSpec:
    """Build an idempotent request to freeze predictions and start scoring."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    return RequestSpec(
        "POST",
        f"/benchmarks/{benchmark_id}/submissions/{submission_id}/submit",
        json=normalize_submit_request(body),
        headers={"Idempotency-Key": require_idempotency_key(idempotency_key)},
    )


def submission_status_request(benchmark_id: str, submission_id: str) -> RequestSpec:
    """Build a submission status request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(submission_id, "submission_id")
    return RequestSpec("GET", f"/benchmarks/{benchmark_id}/submissions/{submission_id}")


def cancel_submission_request(benchmark_id: str, submission_id: str) -> RequestSpec:
    """Build a submission cancellation request."""
    spec = submission_status_request(benchmark_id, submission_id)
    return RequestSpec("DELETE", spec.path)


def collect_manifest(
    first: ManifestPage, pages: Sequence[ManifestPage]
) -> ExternalSubmissionManifest:
    """Merge validated immutable manifest pages."""
    cases = list(first["data"])
    for page in pages:
        if (
            page["submission_id"] != first["submission_id"]
            or page["revision"] != first["revision"]
            or page["task"] != first["task"]
        ):
            raise RuntimeError("manifest identity changed across immutable pages")
        cases.extend(page["data"])
    if len(cases) != first["revision"]["case_count"]:
        raise RuntimeError("manifest case count does not match the frozen revision declaration")
    return {
        "submission_id": first["submission_id"],
        "revision": first["revision"],
        "task": first["task"],
        "data": cases,
        "next_cursor": None,
    }


def collect_revision_manifest(
    first: RevisionManifestPage, pages: Sequence[RevisionManifestPage]
) -> RevisionManifest:
    """Merge validated immutable revision input pages."""
    cases = list(first["data"])
    for page in pages:
        if page["revision"] != first["revision"] or page["task"] != first["task"]:
            raise RuntimeError("revision manifest identity changed across immutable pages")
        cases.extend(page["data"])
    if len(cases) != first["revision"]["case_count"]:
        raise RuntimeError("manifest case count does not match the frozen revision declaration")
    return {
        "revision": first["revision"],
        "task": first["task"],
        "data": cases,
        "next_cursor": None,
    }


def materialized_path(root: Path, case: ManifestCase, input_file: ManifestInputFile) -> Path:
    """Resolve an untrusted logical input path beneath the materialization root."""
    logical_path = validate_artifact_path(input_file["path"])
    candidate = root / f"case-{case['ordinal']:06d}" / "inputs" / logical_path
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("manifest input path escapes the materialization root")
    return candidate


def sanitized_manifest_json(
    manifest: ExternalSubmissionManifest | RevisionManifest,
    files: Sequence[MaterializedInputFile],
    root: Path,
) -> bytes:
    """Serialize a materialized manifest without short-lived signed URLs."""
    paths = {
        (item.dataset_case_id, item.artifact_id, item.logical_path): str(
            item.local_path.relative_to(root)
        )
        for item in files
    }
    cases: list[dict[str, Any]] = []
    for case in manifest["data"]:
        serialized_files = []
        for item in case["input_files"]:
            key = (case["dataset_case_id"], item["artifact_id"], item["path"])
            serialized_files.append(
                {
                    "artifact_id": item["artifact_id"],
                    "storage_object_id": item["storage_object_id"],
                    "path": item["path"],
                    "field_path": item["field_path"],
                    "mime_type": item["mime_type"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "local_path": paths[key],
                }
            )
        cases.append({**case, "input_files": serialized_files})
    value = {"revision": manifest["revision"], "task": manifest["task"], "data": cases}
    manifest_values = cast(Mapping[str, Any], manifest)
    if "submission_id" in manifest_values:
        value["submission_id"] = manifest_values["submission_id"]
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
