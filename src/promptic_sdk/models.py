"""Typed response models for the Promptic API."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from typing_extensions import TypedDict

# ── Enums as Literals ────────────────────────────────────────────────

ExperimentStatus = Literal["pending", "scheduled", "running", "completed", "failed"]
ModelProvider = Literal["openai", "openrouter", "custom", "google"]
OptimizerType = Literal["promptic", "prompticV2", "miproV2", "bootstrapFewShot", "gepa"]
TaskType = Literal["classification", "textGeneration", "structuredOutput"]
EvaluatorType = Literal[
    "f1",
    "referenceJudge",
    "comparisonJudge",
    "generalJudge",
    "similarity",
    "structuredOutput",
]
SplitType = Literal["train", "eval"]
TraceStatus = Literal["ok", "error"]
PromptFormat = Literal["single", "multi_message"]
PromptMessageRole = Literal["system", "user", "assistant"]


# ── Workspace ────────────────────────────────────────────────────────


class Workspace(TypedDict):
    """Workspace info returned by the API."""

    id: str
    name: str
    description: str | None
    createdAt: str
    updatedAt: str


# ── Components ───────────────────────────────────────────────────────


class Component(TypedDict):
    """AI component record."""

    id: str
    name: str
    description: str | None
    costAnalysisConfig: dict[str, Any] | None
    workspaceId: str
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


class Experiment(TypedDict):
    """Experiment record."""

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


class ExperimentList(TypedDict):
    """Paginated list of experiments."""

    data: list[Experiment]


class ExperimentStarted(TypedDict):
    """Response after starting an experiment."""

    messageId: str
    status: str


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


class Iteration(_IterationOptional):
    """Experiment iteration record.

    All keys below are populated by every iteration response. The
    optional ``avgPredictionLatencyMs`` key (inherited from
    ``_IterationOptional``) is absent on iterations completed before
    per-prediction latency tracking shipped.
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
    workspaceId: str
    traceDbId: str | None
    spanDbId: str | None
    traceId: str | None
    spanId: str | None
    sourcePath: str
    sourceField: str
    source: str
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
