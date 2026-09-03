"""Typed response models for the Promptic API."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

# ── Enums as Literals ────────────────────────────────────────────────

ExperimentStatus = Literal["pending", "scheduled", "running", "completed", "failed"]
ModelProvider = Literal["openai", "openrouter", "custom", "google"]
OptimizerType = Literal["promptic", "prompticV2", "miproV2", "bootstrapFewShot", "gepa"]
TaskType = Literal["classification", "textGeneration", "structuredOutput", "toolSelection"]
PromptExperimentTaskType = Literal["classification", "textGeneration", "structuredOutput"]
ToolSource = Literal["mcp", "manual"]
EvaluatorType = Literal[
    "f1",
    "referenceJudge",
    "comparisonJudge",
    "generalJudge",
    "similarity",
    "structuredOutput",
    "toolSelection",
]
SplitType = Literal["train", "eval"]
TraceStatus = Literal["ok", "error"]
PromptFormat = Literal["single", "multi_message"]
PromptMessageRole = Literal["system", "user", "assistant"]


# ── AI Application ────────────────────────────────────────────────────


class AIApplication(TypedDict):
    """AI Application info returned by the API."""

    id: str
    name: str
    description: str | None
    createdAt: str
    updatedAt: str


# Deprecated alias — kept for backward compatibility. ``AIApplication`` is the
# customer-facing name for the same object.
Workspace = AIApplication


# ── Components ───────────────────────────────────────────────────────


class Component(TypedDict):
    """AI component record."""

    id: str
    name: str
    description: str | None
    costAnalysisConfig: dict[str, Any] | None
    aiApplicationId: str
    createdAt: str
    updatedAt: str


class ComponentList(TypedDict):
    """Paginated list of AI components."""

    data: list[Component]


class ComponentCreated(TypedDict):
    """Response after creating an AI component."""

    id: str


# ── Experiments ──────────────────────────────────────────────────────


class Hyperparameters(TypedDict, total=False):
    """Experiment hyperparameters (all optional)."""

    epochs: int
    trainSplitRatio: float
    numFewShots: int
    enableCot: bool


class _ExperimentRequired(TypedDict):
    """Fields present on every experiment record."""

    id: str
    datasetId: str
    name: str | None
    description: str | None
    targetModel: str
    provider: ModelProvider
    aiComponentId: str
    customProviderId: str | None
    createdByUser: str | None
    experimentStatus: ExperimentStatus
    taskType: TaskType
    optimizer: OptimizerType
    tokensUsed: float
    promptFormat: PromptFormat
    initialPromptMessages: list[PromptMessage]
    initialPromptTokens: int | None
    initialPredictionModelSchema: Any
    runNumber: int | None
    hyperparameters: Hyperparameters
    startedAt: str | None
    endedAt: str | None
    retries: int
    errorCode: str | None
    errorMessage: str | None
    createdAt: str
    updatedAt: str


class Experiment(_ExperimentRequired, total=False):
    """Experiment record.

    The three ``*SystemPrompt`` fields only ship with the tool-selection
    optimizer; older API responses (that predate those columns) omit them, so
    they are optional. They live on this ``total=False`` subclass — rather than
    as ``NotRequired`` markers on a single TypedDict — because this module uses
    ``from __future__ import annotations``: under postponed evaluation the
    ``NotRequired`` wrapper is stored as a string and the runtime metadata
    (``__required_keys__`` / ``__optional_keys__``) would wrongly treat the
    fields as required. The base/extension split keeps the optionality correct
    at runtime for anyone introspecting the model.
    """

    # Tool-selection experiments only. ``systemPrompt`` is the fixed system
    # prompt used as context during evaluation. ``optimizeSystemPrompt`` is
    # the toggle that asks the optimizer to also rewrite the system prompt;
    # when on, the best variant is persisted as ``optimizedSystemPrompt``.
    systemPrompt: str | None
    optimizeSystemPrompt: bool
    optimizedSystemPrompt: str | None


class ExperimentList(TypedDict):
    """Paginated list of experiments."""

    data: list[Experiment]


class ExperimentStarted(TypedDict):
    """Response after starting an experiment."""

    messageId: str
    status: str


class _ToolSelectionToolRequired(TypedDict):
    """Tool definition used by tool-selection optimization."""

    name: str
    description: str


class ToolSelectionTool(_ToolSelectionToolRequired, total=False):
    """Tool definition with optional snake- or camel-case input schema."""

    input_schema: dict[str, Any]
    inputSchema: dict[str, Any]


class _SnakeCaseToolSelectionTestCase(TypedDict):
    query: str
    expected_tool: str


class _CamelCaseToolSelectionTestCase(TypedDict):
    query: str
    expectedTool: str


ToolSelectionTestCase: TypeAlias = _SnakeCaseToolSelectionTestCase | _CamelCaseToolSelectionTestCase


# ── Evaluators ───────────────────────────────────────────────────────


class Evaluator(TypedDict):
    """Evaluator configuration record."""

    id: str
    experimentId: str
    name: str
    description: str | None
    type: EvaluatorType
    scaleMin: float
    scaleMax: float
    weight: float
    config: dict[str, Any] | None
    createdAt: str
    updatedAt: str


class EvaluatorList(TypedDict):
    """Paginated list of evaluators."""

    data: list[Evaluator]


# ── Iterations ───────────────────────────────────────────────────────


class _IterationOptional(TypedDict, total=False):
    """Optional iteration keys.

    Carved into a ``total=False`` base so the optionality is recorded in
    ``__required_keys__`` / ``__optional_keys__`` at runtime — using
    ``NotRequired`` directly on the merged class is invisible to TypedDict
    metaclass introspection under ``from __future__ import annotations``
    (the annotation is stored as a forward-reference string).
    """

    # Mean target-model prediction latency for this iteration (ms),
    # averaged across train + eval predictions. Excludes retries,
    # rate-limit backoff, and failed attempts. The API omits this key
    # on iterations completed before per-prediction latency tracking
    # shipped.
    avgPredictionLatencyMs: int | None

    # Tool-selection only: the optimized description for each tool in this
    # iteration, keyed by tool name. The API returns null or omits the field
    # for other task types and historical iterations.
    toolDescriptions: dict[str, str] | None

    # Tool-selection only: the system prompt used for selection in this
    # iteration after evaluation-only guardrails have been removed. It is null
    # or omitted unless system-prompt optimization is enabled.
    selectionSystemPrompt: str | None


class Iteration(_IterationOptional):
    """Experiment iteration record.

    All keys below are populated by every iteration response. The
    Task-specific and historically unavailable keys are inherited from
    ``_IterationOptional`` so callers can safely handle payloads that omit
    them.
    """

    id: int
    experimentId: str
    iterationNumber: int
    prompt: str
    promptTokens: int | None
    overallNormalizedScore: float
    evalNormalizedScore: float | None
    schemaSnapshot: Any
    createdAt: str
    updatedAt: str


class IterationScore(TypedDict):
    """Per-evaluator score for an iteration."""

    iterationId: int
    score: float
    rawScore: float
    evaluatorId: str
    evaluatorName: str
    evaluatorDescription: str | None
    evaluatorType: EvaluatorType


class IterationWithScores(Iteration, total=False):
    """Iteration with optional evaluator score breakdown."""

    scores: list[IterationScore]


class IterationList(TypedDict):
    """Paginated list of iterations."""

    data: list[Iteration]


# ── Deployments ──────────────────────────────────────────────────────


class Deployment(TypedDict):
    """Active deployment linking a component to an experiment."""

    aiComponentId: str
    experimentId: str
    experiment: Experiment


class PromptMessage(TypedDict):
    """A single message in a prompt template."""

    role: PromptMessageRole
    content: str


class DeployedPrompt(TypedDict):
    """Deployed prompt with metadata from the best iteration."""

    prompt: str
    promptMessages: list[PromptMessage]
    promptFormat: PromptFormat
    model: str
    provider: ModelProvider
    componentId: str
    componentName: str | None
    experimentId: str
    iterationId: int
    score: float
    schemaSnapshot: Any


class DeploymentCreated(TypedDict):
    """Response after deploying an experiment."""

    aiComponentId: str
    experimentId: str


# ── Traces ───────────────────────────────────────────────────────────


class TraceListItem(TypedDict):
    """Summary trace item in list responses."""

    id: str
    traceId: str
    name: str | None
    status: str
    durationMs: int | None
    totalTokens: int | None
    totalCostUsd: float | None
    startTime: str
    componentName: str | None


class TraceList(TypedDict):
    """Paginated list of traces."""

    traces: list[TraceListItem]
    total: int


class SpanEvent(TypedDict):
    """OTel span event (message, tool call, etc.)."""

    id: int
    spanDbId: str
    name: str
    timestamp: str
    body: Any


class Span(TypedDict):
    """Trace span representing a single operation."""

    id: str
    spanId: str
    traceId: str
    traceDbId: str
    parentSpanId: str | None
    name: str
    kind: str
    startTime: str
    endTime: str | None
    durationMs: int | None
    status: str
    statusMessage: str | None
    input: str | None
    output: str | None
    metadata: Any
    model: str | None
    provider: str | None
    inputTokens: int | None
    outputTokens: int | None
    totalTokens: int | None
    costUsd: float | None
    modelParameters: Any
    createdAt: str
    events: list[SpanEvent]


class Trace(TypedDict):
    """Full trace with all spans and events."""

    id: str
    traceId: str
    workspaceId: str
    aiComponentId: str | None
    name: str | None
    sessionId: str | None
    userId: str | None
    metadata: Any
    startTime: str
    endTime: str | None
    durationMs: int | None
    totalTokens: int | None
    totalCostUsd: float | None
    status: str
    tags: list[str]
    createdAt: str
    spans: list[Span]


class TracingStats(TypedDict):
    """Aggregated tracing statistics."""

    totalTraces: int
    totalTokens: int
    totalCostUsd: float
    errorRate: float


class TraceArtifact(TypedDict):
    """Artifact referenced from a trace or uploaded explicitly."""

    id: str
    aiApplicationId: str
    traceDbId: str | None
    spanDbId: str | None
    traceId: str | None
    spanId: str | None
    sourcePath: str
    sourceField: str
    source: str
    name: NotRequired[str | None]
    mimeType: str
    sizeBytes: int
    sha256: str
    storageProvider: str
    storageKey: str | None
    preview: str | None
    createdAt: str
    uri: str


class TraceArtifactList(TypedDict):
    """List response for trace artifacts."""

    data: list[TraceArtifact]


# ── Datasets ────────────────────────────────────────────────────────

AgentEvaluationStatus = Literal["pending", "running", "completed", "failed"]
JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


class DatasetCase(TypedDict):
    """Canonical JSON case whose payloads may contain Promptic resource URIs."""

    id: int
    datasetId: str
    idx: int | None
    inputPayload: dict[str, JSONValue]
    expectedPayload: JSONValue
    split: SplitType | None
    metadata: dict[str, JSONValue]
    createdAt: str
    updatedAt: str


class DatasetCaseCreateRequired(TypedDict):
    """Required fields for a canonical dataset case."""

    inputPayload: dict[str, JSONValue]


class DatasetCaseCreate(DatasetCaseCreateRequired, total=False):
    """Fields accepted when creating a canonical dataset case."""

    idx: int | None
    expectedPayload: JSONValue
    split: SplitType | None
    metadata: dict[str, JSONValue]


class DatasetCaseUpdate(TypedDict, total=False):
    """Fields accepted when updating a canonical dataset case."""

    idx: int | None
    inputPayload: dict[str, JSONValue]
    expectedPayload: JSONValue
    split: SplitType | None
    metadata: dict[str, JSONValue]


class DatasetCaseList(TypedDict):
    """List of canonical dataset cases."""

    data: list[DatasetCase]


class Dataset(TypedDict):
    """AI component-owned dataset record."""

    id: str
    name: str
    description: str | None
    aiComponentId: str
    aiApplicationId: str
    caseCount: int
    createdAt: str
    updatedAt: str


class DatasetWithCases(Dataset):
    """Dataset with its canonical cases."""

    cases: list[DatasetCase]


class DatasetList(TypedDict):
    """List of datasets."""

    data: list[Dataset]


# ── Runs ────────────────────────────────────────────────────────────

AnnotationRating = Literal["positive", "negative"]


class Run(TypedDict):
    """Agent run record — traces grouped for a dataset."""

    id: str
    name: str | None
    datasetId: str
    aiComponentId: str
    workspaceId: str
    status: str
    traceCount: int
    createdAt: str
    updatedAt: str


class RunWithTraces(Run):
    """Run with its linked traces."""

    traces: list[TraceListItem]


class RunList(TypedDict):
    """List of runs."""

    data: list[Run]


# ── Annotations ─────────────────────────────────────────────────────


class Annotation(TypedDict):
    """Annotation record — per-trace human feedback within a run."""

    id: str
    runId: str
    traceDbId: str
    userId: str
    rating: str | None
    comment: str | None
    createdAt: str
    updatedAt: str


class AnnotationList(TypedDict):
    """List of annotations."""

    data: list[Annotation]


# ── Agent Evaluations ───────────────────────────────────────────────


class InsightDetail(TypedDict, total=False):
    """Detail fields for an insight (varies by type)."""

    toolName: str
    errorRate: float
    tokensWasted: int
    stepIndex: int
    costPercentage: float
    usageRate: float


class Insight(TypedDict):
    """A single evaluation insight."""

    type: str
    severity: str
    title: str
    description: str
    frequency: float
    affectedRunIds: list[str]
    details: dict[str, Any]
    suggestedFix: str | None


class InsightResultMeta(TypedDict):
    """Metadata for an insight result."""

    totalRuns: int
    totalTokens: int
    totalCostUsd: float
    averageDurationMs: float
    errorRate: float
    analyzedAt: str


class InsightResult(TypedDict):
    """Full insight result from an evaluation."""

    insights: list[Insight]
    meta: InsightResultMeta


EvaluationSubject = Literal["output", "trajectory", "annotation"]
EvaluationTargetType = Literal["trace", "run", "dataset"]
JudgeBackendType = Literal["deterministic", "llm_judge", "agent_judge"]


class AgentEvaluation(TypedDict):
    """Agent evaluation record.

    Evaluations are scoped by ``subject`` (what the judge looks at: output,
    trajectory, or annotation) and anchored to a durable resource by
    ``targetType`` (trace, run, or dataset). The anchor IDs follow the
    target type: ``trace`` evaluations populate only ``traceDbId``;
    ``dataset`` evaluations populate only ``datasetId``; ``run`` evaluations
    populate both ``runId`` and ``datasetId`` (since a run belongs to a
    dataset).

    The four scoping fields (``subject``, ``targetType``, ``traceDbId``,
    ``budget``) are wrapped in ``NotRequired`` so evaluations created before
    the scoped-evaluation migration still type-check — every other field
    remains required because the API always returns it.
    """

    id: str
    name: str | None
    aiComponentId: str
    datasetId: str | None
    runId: str | None
    status: AgentEvaluationStatus
    results: InsightResult | None
    startedAt: str | None
    completedAt: str | None
    createdAt: str
    updatedAt: str
    subject: NotRequired[EvaluationSubject]
    targetType: NotRequired[EvaluationTargetType]
    traceDbId: NotRequired[str | None]
    budget: NotRequired[dict[str, Any] | None]


class AgentEvaluationList(TypedDict):
    """List of agent evaluations."""

    data: list[AgentEvaluation]


class JudgeResult(TypedDict):
    """Canonical per-(target, judge) row produced by an evaluation.

    Each row identifies its judge (``judgeKey`` / ``judgeName`` / ``backend``),
    the concrete thing it scored via exactly one of ``datasetCaseId``,
    ``traceDbId``, ``annotationId``, or ``runId``, the verdict
    (``score`` / ``rating`` / ``rationale`` / ``evidence`` / ``analysisPayload``),
    and the immutable ``judgeSnapshot`` used to produce it. Re-runs of the
    same ``(evaluation, target, judgeKey)`` are versioned via ``version``.
    """

    id: str
    evaluationId: str
    subject: EvaluationSubject
    backend: JudgeBackendType
    judgeKey: str
    judgeName: str
    traceDbId: str | None
    traceOtlpId: str | None
    datasetCaseId: int | None
    annotationId: str | None
    runId: str | None
    score: float | None
    rating: str | None
    rationale: str | None
    evidence: Any
    analysisPayload: Any
    judgeSnapshot: dict[str, Any]
    version: int
    createdAt: str


class JudgeResultList(TypedDict):
    """Response wrapper for ``list_judge_results``."""

    rows: list[JudgeResult]
