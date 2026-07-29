"""Typed models for Agent Gym external benchmark submissions."""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

ExternalSubmissionStatus = Literal[
    "created",
    "uploading",
    "finalizing",
    "queued",
    "scoring",
    "succeeded",
    "failed",
    "expired",
    "cancelled",
]
ExternalSubmissionArtifactStatus = Literal["reserved", "verified", "attached", "deleted"]
ExternalPredictionStatus = Literal["succeeded", "failed", "skipped", "cancelled"]
ExternalOutputKind = Literal[
    "structured",
    "text",
    "artifact",
    "side_effect",
    "none",
    "custom",
]
BenchmarkRunStatus = Literal["queued", "running", "succeeded", "failed"]
BenchmarkScoringStatus = Literal["pending", "running", "succeeded", "failed"]
BenchmarkEligibilityStatus = Literal["not_applicable", "pending", "eligible", "ineligible"]


class ExternalTaskSnapshot(TypedDict):
    """Frozen task contract returned with a submission and its manifest."""

    taskId: str | None
    name: str
    description: str | None
    inputContract: dict[str, Any]
    outputContract: dict[str, Any]
    publicSuccessCriteria: dict[str, Any] | None


class ExternalSubmissionRevision(TypedDict):
    """Frozen benchmark revision selected for a submission."""

    id: str
    version: int
    fingerprint: str
    case_count: int
    scorer_contract_version: str


class ExternalSubmissionLinks(TypedDict):
    """Relative API links returned when a submission is created."""

    manifest: str
    artifacts: str
    finalize: str
    status: str


class ExternalSubmissionCreated(TypedDict):
    """Response from creating or replaying an external submission."""

    submission_id: str
    revision: ExternalSubmissionRevision
    status: ExternalSubmissionStatus
    expires_at: str
    task: ExternalTaskSnapshot
    links: ExternalSubmissionLinks
    created: bool


class ManifestRevision(TypedDict):
    """Revision identity repeated on each manifest page."""

    id: str
    version: int
    fingerprint: str
    case_count: int


class ManifestInputFile(TypedDict):
    """Input file with a short-lived direct download URL."""

    artifact_id: str
    storage_object_id: str
    path: str
    mime_type: str
    size_bytes: int
    sha256: str | None
    download_url: str
    expires_at: str


class ManifestCase(TypedDict):
    """One immutable revision case.

    ``case_id`` is the revision case UUID required as
    ``revision_case_id`` when finalizing a prediction.
    """

    case_id: str
    ordinal: int
    input_payload: dict[str, Any]
    input_files: list[ManifestInputFile]


class ManifestPage(TypedDict):
    """One cursor-paginated immutable manifest page."""

    submission_id: str
    revision: ManifestRevision
    task: ExternalTaskSnapshot
    data: list[ManifestCase]
    next_cursor: str | None


class ExternalSubmissionManifest(TypedDict):
    """Fully collected immutable manifest."""

    submission_id: str
    revision: ManifestRevision
    task: ExternalTaskSnapshot
    data: list[ManifestCase]
    next_cursor: None


class PresignedUploadRequired(TypedDict):
    """Required fields in a submission artifact upload descriptor."""

    strategy: Literal["url"]
    provider: Literal["s3", "azure"]
    uploadUrl: str
    finalUrl: str
    method: Literal["PUT", "POST"]
    maxSizeBytes: int
    expiresAt: str


class PresignedUpload(PresignedUploadRequired, total=False):
    """Direct upload target returned by the active storage provider."""

    headers: dict[str, str]
    fields: dict[str, str]


class ReservedSubmissionArtifact(TypedDict):
    """Artifact reservation owned by one external submission."""

    artifact_id: str
    storage_object_id: str
    path: str
    status: ExternalSubmissionArtifactStatus
    upload: PresignedUpload


class CompletedSubmissionArtifact(TypedDict):
    """Response after server-side artifact verification."""

    artifact_id: str
    status: ExternalSubmissionArtifactStatus


class SubmissionArtifact(TypedDict):
    """Verified submission output artifact ready to attach to one prediction."""

    artifact_id: str
    storage_object_id: str
    path: str
    status: ExternalSubmissionArtifactStatus


class TraceResolution(TypedDict):
    """Mapping from a raw 32-hex OTEL trace ID to its database UUID."""

    trace_id: str
    trace_db_id: str | None


class TraceResolutionList(TypedDict):
    """Trace mappings returned by the submission-scoped resolver."""

    data: list[TraceResolution]


class ExecutionRefs(TypedDict, total=False):
    """Database evidence IDs accepted by finalization.

    ``trace_ids`` contains resolved trace database UUIDs, not raw OTEL IDs.
    """

    trace_ids: list[str]
    trace_artifact_ids: list[str]


class RunOutputRequired(TypedDict):
    """Required portion of a terminal successful prediction output."""

    kind: ExternalOutputKind


class RunOutput(RunOutputRequired, total=False):
    """Successful prediction output."""

    value: Any


class TokenUsage(TypedDict):
    """Token usage reported by an external executor."""

    prompt: int
    completion: int
    total: int


class BundleIdentityRequired(TypedDict):
    """Required identity for the externally implemented candidate bundle."""

    name: str
    version: str


class BundleIdentity(BundleIdentityRequired, total=False):
    """Candidate bundle identity and architecture metadata."""

    parent_version: str
    rationale: str
    intent: str
    architecture_description: str
    architecture_tags: list[str]
    commit_hash: str


class ExternalPredictionRequired(TypedDict):
    """Required fields for one terminal revision-case prediction."""

    revision_case_id: str
    status: ExternalPredictionStatus


class ExternalPrediction(ExternalPredictionRequired, total=False):
    """One terminal prediction submitted for an immutable manifest case."""

    output: RunOutput
    artifact_ids: list[str]
    execution_refs: ExecutionRefs
    implementation_reference_id: str
    executor_id: str
    executor_version: str
    token_usage: TokenUsage
    latency_ms: int
    started_at: str
    completed_at: str
    error_code: str
    error_category: str
    retryable: bool
    error: str
    diagnostics: dict[str, Any]


class FinalizeSubmissionRequestRequired(TypedDict):
    """Required fields for finalizing an external submission."""

    bundle_identity: BundleIdentity
    predictions: list[ExternalPrediction]


class FinalizeSubmissionRequest(FinalizeSubmissionRequestRequired, total=False):
    """Complete external submission finalization request."""

    implementation_reference_id: str
    execution_refs: ExecutionRefs
    metadata: dict[str, Any]


class FinalizedSubmission(TypedDict):
    """Response after idempotent finalization and scoring dispatch."""

    submission_id: str
    run_id: str
    bundle_id: str
    status: Literal["queued"]
    dispatch_status: Literal["dispatched", "pending"]
    created: bool


class BenchmarkRunLink(TypedDict):
    """Leaderboard run linked to a finalized external submission."""

    id: str
    status: BenchmarkRunStatus
    scoring_status: BenchmarkScoringStatus
    eligibility_status: BenchmarkEligibilityStatus
    eligibility_reasons: list[Any]
    scored_at: str | None
    error: str | None


class SubmissionStatus(TypedDict):
    """Current submission and linked benchmark-run state."""

    submission_id: str
    revision_id: str
    status: ExternalSubmissionStatus
    expires_at: str
    finalized_at: str | None
    queued_at: str | None
    completed_at: str | None
    validation_error: dict[str, Any] | None
    run: BenchmarkRunLink | None


class CancelledSubmission(TypedDict):
    """Response after cancelling an unfinalized submission."""

    submission_id: str
    status: Literal["cancelled"]
