"""Typed wire models for Agent Gym submissions and result inspection."""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from typing_extensions import TypedDict

ExternalSubmissionStatus = Literal[
    "created",
    "uploading",
    "finalizing",
    "queued",
    "dispatch_failed",
    "scoring",
    "succeeded",
    "failed",
    "expired",
    "cancelled",
]
ScoringDispatchStatus = Literal["pending", "dispatching", "dispatched", "acknowledged", "failed"]
ExternalSubmissionArtifactStatus = Literal["reserved", "verified", "attached", "deleted"]
ExternalPredictionStatus = Literal["succeeded", "failed", "skipped", "cancelled"]
BenchmarkRunStatus = Literal["queued", "running", "succeeded", "failed"]
BenchmarkScoringStatus = Literal["pending", "running", "succeeded", "failed"]
BenchmarkEligibilityStatus = Literal["not_applicable", "pending", "eligible", "ineligible"]
CaseResultSort = Literal["score", "score_desc", "latency", "case"]
RunComparisonClassification = Literal["improved", "regressed", "unchanged", "incomparable"]


class ExternalTaskSnapshot(TypedDict):
    """Frozen public task contract returned with a submission manifest."""

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
    predictions: str
    submit: str
    status: str


class ExternalSubmissionCreated(TypedDict):
    """Response from creating or replaying an external submission."""

    submission_id: str
    run_id: str
    variant_id: str
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
    field_path: str
    mime_type: str
    size_bytes: int
    sha256: str | None
    download_url: str
    expires_at: str


class ManifestCase(TypedDict):
    """One immutable revision case."""

    dataset_case_id: int
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


class RevisionManifestPage(TypedDict):
    """One cursor-paginated immutable revision input page."""

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


class RevisionManifest(TypedDict):
    """Fully collected immutable revision inputs without submission state."""

    revision: ManifestRevision
    task: ExternalTaskSnapshot
    data: list[ManifestCase]
    next_cursor: None


class PresignedUploadRequired(TypedDict):
    """Required fields in a direct-upload descriptor."""

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
    """Verified submission artifact ready to attach to a prediction."""

    artifact_id: str
    storage_object_id: str
    path: str
    status: ExternalSubmissionArtifactStatus


class TraceResolution(TypedDict):
    """Mapping from a raw 32-hex OTEL trace ID to its database UUID."""

    trace_id: str
    trace_db_id: str | None


class TraceResolutionList(TypedDict):
    """Submission-scoped trace mappings."""

    data: list[TraceResolution]


class ExecutionRefs(TypedDict, total=False):
    """Database evidence IDs accepted when predictions are submitted."""

    trace_ids: list[str]
    trace_artifact_ids: list[str]


class TokenUsage(TypedDict):
    """Token usage reported by an external executor."""

    prompt: int
    completion: int
    total: int


class VariantIdentityRequired(TypedDict):
    """Required candidate variant identity."""

    name: str
    version: str


class VariantIdentity(VariantIdentityRequired, total=False):
    """Candidate identity and architecture metadata."""

    parent_name: str
    parent_version: str
    rationale: str
    intent: str
    architecture_description: str
    architecture_tags: list[str]
    repository_url: str
    commit_hash: str


class ExternalPredictionRequired(TypedDict):
    """Required fields for one terminal prediction."""

    dataset_case_id: int
    status: ExternalPredictionStatus


class ExternalPrediction(ExternalPredictionRequired, total=False):
    """One terminal prediction for an immutable manifest case."""

    output: Any
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


class SubmitSubmissionRequest(TypedDict, total=False):
    """Optional evidence metadata applied when prediction uploads are frozen."""

    variant_identity: VariantIdentity
    implementation_reference_id: str
    execution_refs: ExecutionRefs
    metadata: dict[str, Any]


class StagedPredictionBatch(TypedDict):
    """Response after uploading an idempotent prediction batch."""

    accepted: int
    stored: int


class SubmittedSubmission(TypedDict):
    """Response after canonical predictions are closed and queued for scoring."""

    submission_id: str
    run_id: str
    variant_id: str
    status: Literal["queued", "running", "succeeded", "failed", "dispatch_failed"]
    dispatch_status: Literal["dispatched", "failed"]
    dispatch_attempts: int
    created: bool


class BenchmarkRunLink(TypedDict):
    """Benchmark run linked to a submitted external prediction set."""

    id: str
    status: BenchmarkRunStatus
    scoring_status: BenchmarkScoringStatus
    eligibility_status: BenchmarkEligibilityStatus
    eligibility_reasons: list[Any]
    scored_at: str | None
    error: str | None


class SubmissionDispatch(TypedDict):
    """Recoverable scoring-dispatch evidence for a submitted prediction set."""

    status: ScoringDispatchStatus
    attempts: int
    retry_available_at: str
    last_error: str | None


class SubmissionStatus(TypedDict):
    """Current submission and linked benchmark-run state."""

    submission_id: str
    revision_id: str
    status: ExternalSubmissionStatus
    expires_at: str
    prediction_count: int
    expected_prediction_count: int
    submitted_at: str | None
    queued_at: str | None
    completed_at: str | None
    validation_error: dict[str, Any] | None
    submit_metadata: dict[str, Any] | None
    dispatch: SubmissionDispatch | None
    run: BenchmarkRunLink | None


class RetriedBenchmarkScoring(TypedDict):
    """Response from restoring scoring delivery for the same benchmark run."""

    run_id: str
    scoring_status: Literal["pending"]
    dispatch_status: Literal["dispatched"]
    dispatch_attempts: NotRequired[int]


class CancelledSubmission(TypedDict):
    """Response after cancelling a submission before scoring is requested."""

    submission_id: str
    status: Literal["cancelled"]


class QueuedBenchmarkReevaluation(TypedDict):
    """A benchmark run accepted for scoring under the active evaluator revision."""

    run_id: str
    evaluation_run_id: str
    status: Literal["queued"]


class BenchmarkEligibility(TypedDict):
    """Official leaderboard eligibility."""

    status: BenchmarkEligibilityStatus
    reasons: list[str]


class BenchmarkArchitectureParent(TypedDict):
    """Parent architecture reference."""

    name: str
    version: str | None


class BenchmarkArchitecture(TypedDict):
    """Architecture represented in a benchmark run."""

    id: str
    name: str
    version: str
    architecture_description: str | None
    rationale: str | None
    intent: str | None
    commit_hash: str | None
    parent: BenchmarkArchitectureParent | None
    external: bool


class ArchitectureReference(TypedDict):
    """Minimal architecture identity in results."""

    name: str
    version: str


class BenchmarkAggregate(TypedDict):
    """Aggregate metrics for one architecture."""

    architecture: ArchitectureReference
    mean_score: float | None
    mean_per_field_scores: dict[str, float] | None
    mean_per_field_scores_basis: Literal["succeeded_only"]
    success_rate: float | None
    mean_latency_ms: float | None
    total_token_usage: dict[str, int] | None
    case_count: int


class BenchmarkProgress(TypedDict):
    """Terminal prediction counts for a run."""

    terminal: int
    total: int
    succeeded: int
    failed: int


class BenchmarkScoreStatusCounts(TypedDict, total=False):
    """Case-score outcome counts across the run."""

    succeeded: int
    failed: int
    skipped: int
    insufficient_evidence: int


class BenchmarkTimestamps(TypedDict):
    """Benchmark run lifecycle timestamps."""

    created_at: str
    started_at: str | None
    finished_at: str | None
    scored_at: str | None


class BenchmarkInsight(TypedDict):
    """Generated run or case insight."""

    source: str
    severity: str
    code: str
    summary: str
    details: dict[str, Any]
    citations: dict[str, Any]


class BenchmarkRunInsight(BenchmarkInsight):
    """Run-level insight with optional case linkage."""

    prediction_id: str | None
    case_id: int | None


class BenchmarkRunResults(TypedDict):
    """Aggregate authenticated benchmark-run results."""

    benchmark_id: str
    run_id: str
    revision_id: str | None
    revision_fingerprint: str | None
    scorer_contract_version: str | None
    evaluator_fingerprint: str | None
    runtime_location: str
    status: BenchmarkRunStatus
    scoring_status: BenchmarkScoringStatus
    evaluation: BenchmarkEvaluation | None
    eligibility: BenchmarkEligibility
    architectures: list[BenchmarkArchitecture]
    aggregates: list[BenchmarkAggregate]
    progress: BenchmarkProgress
    score_status_counts: BenchmarkScoreStatusCounts
    timestamps: BenchmarkTimestamps
    error: str | None
    insights: list[BenchmarkRunInsight]
    links: dict[str, str]


class BenchmarkScore(TypedDict):
    """Official score for one case field."""

    key: str
    evaluator_id: str | None
    evaluator_type: str | None
    evaluator_name: str | None
    evaluator_weight: float
    score: float | None
    strategy: str | None
    status: str
    details: dict[str, Any] | None
    error: str | None


class BenchmarkEvaluatorResult(TypedDict):
    """Aggregate result for one independently executed evaluator."""

    id: str
    source_evaluator_id: str
    metric_key: str | None
    type: str
    name: str
    weight: float
    architecture: ArchitectureReference
    score: float | None
    mean_per_field_scores: dict[str, float]
    mean_per_field_scores_basis: Literal["succeeded_only"]
    case_count: int
    error: str | None


class BenchmarkEvaluation(TypedDict):
    """Latest successful evaluation revision attached to a run."""

    id: str
    trigger: str
    status: str
    composite_score: float | None
    evaluators: list[BenchmarkEvaluatorResult]
    created_at: str
    completed_at: str | None


class BenchmarkJudgement(TypedDict):
    """Normalized official or advisory judgement."""

    key: str
    name: str
    subject: Literal["output", "trajectory", "annotation"]
    backend: Literal["deterministic", "llm_judge", "agent_judge"]
    score: float | None
    rating: str | None
    rationale: str | None
    evidence: list[Any]
    analysis: dict[str, Any] | None
    status: str
    error: str | None
    official: bool
    source: Literal["leaderboard_score", "advisory_judge"]


class _PredictionArtifactRequired(TypedDict):
    """Required prediction artifact response fields."""

    storage_object_id: str
    path: str | None
    mime_type: str | None
    size_bytes: int | None
    role: Literal["output"]
    content_url: str
    download_url: str


class PredictionArtifact(_PredictionArtifactRequired, total=False):
    """Output artifact attached to a prediction.

    ``sha256`` is accepted for forward compatibility and verified by the SDK
    when present; the current API always provides ``size_bytes``.
    """

    sha256: str | None


class BenchmarkTrace(TypedDict):
    """Execution trace attached to a prediction."""

    trace_db_id: str
    trace_id: str
    name: str | None
    status: str
    duration_ms: float | None


class BenchmarkMetrics(TypedDict):
    """Execution metrics for one prediction."""

    latency_ms: int | None
    token_usage: dict[str, int] | None


class BenchmarkExecution(TypedDict):
    """Execution provenance for one prediction."""

    runtime_location: str
    implementation_reference_id: str | None
    executor_id: str | None
    executor_version: str | None


class BenchmarkCaseError(TypedDict):
    """Terminal prediction failure details."""

    code: str | None
    category: str | None
    message: str | None
    retryable: bool | None


class BenchmarkCaseResult(TypedDict):
    """Complete result and evidence for one immutable case."""

    case_id: int
    prediction_id: str
    architecture: ArchitectureReference
    status: str
    output: Any
    overall_score: float | None
    scores: list[BenchmarkScore]
    judgements: list[BenchmarkJudgement]
    artifacts: list[PredictionArtifact]
    traces: list[BenchmarkTrace]
    trace_artifact_ids: list[str]
    metrics: BenchmarkMetrics
    execution: BenchmarkExecution
    error: BenchmarkCaseError | None
    diagnostics: dict[str, Any] | None
    insights: list[BenchmarkInsight]


class BenchmarkCaseResultPage(TypedDict):
    """Cursor-paginated case results."""

    run_id: str
    data: list[BenchmarkCaseResult]
    total: int
    next_cursor: str | None


class ComparableContext(TypedDict):
    """Context checks required for paired comparison."""

    same_revision: bool
    same_scorer_contract: bool
    same_evaluator: bool
    same_case_set: bool


class ComparedPrediction(TypedDict):
    """Compact prediction side of one paired comparison."""

    prediction_id: str
    status: str
    score: float | None
    latency_ms: int | None


class ComparedRun(TypedDict):
    """Run identity and aggregate in a paired comparison."""

    run_id: str
    architecture: BenchmarkArchitecture | None
    aggregate: BenchmarkAggregate | None


class RunComparisonSummary(TypedDict):
    """Aggregate deltas and case classifications."""

    mean_score_delta: float | None
    success_rate_delta: float | None
    mean_latency_delta_ms: float | None
    improved_cases: int
    regressed_cases: int
    unchanged_cases: int
    incomparable_cases: int


class RunComparisonCase(TypedDict):
    """One paired immutable case comparison."""

    case_id: int
    classification: RunComparisonClassification
    score_delta: float | None
    latency_delta_ms: float | None
    parent: ComparedPrediction | None
    candidate: ComparedPrediction | None


class BenchmarkRunComparison(TypedDict):
    """Paired comparison for two compatible benchmark runs."""

    benchmark_id: str
    comparable_context: ComparableContext
    parent: ComparedRun
    candidate: ComparedRun
    summary: RunComparisonSummary
    cases: list[RunComparisonCase]
