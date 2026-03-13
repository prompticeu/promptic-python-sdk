"""Typed response models for the Promptic API."""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

# ── Enums as Literals ────────────────────────────────────────────────

ExperimentStatus = Literal["pending", "scheduled", "running", "completed", "failed"]
ModelProvider = Literal["openai", "openrouter", "custom", "google"]
OptimizerType = Literal["promptic", "prompticV2", "miproV2", "bootstrapFewShot", "gepa"]
TaskType = Literal["classification", "textGeneration", "structuredOutput"]
EvaluatorType = Literal["f1", "judge", "similarity", "structuredOutput"]
SplitType = Literal["train", "eval"]
TraceStatus = Literal["ok", "error"]


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
    initialPrompt: str | None
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


# ── Observations ─────────────────────────────────────────────────────


class Observation(TypedDict):
    """Observation (input/expected pair) record."""

    id: int
    experimentId: str
    idx: int
    input: str
    expected: str
    variables: Any
    split: SplitType
    createdAt: str
    updatedAt: str


class ObservationList(TypedDict):
    """Paginated list of observations."""

    data: list[Observation]


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


class Iteration(TypedDict):
    """Experiment iteration record."""

    id: int
    experimentId: str
    iterationNumber: int
    prompt: str
    promptTokens: int | None
    overallNormalizedScore: float
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


class DeployedPrompt(TypedDict):
    """Deployed prompt with metadata from the best iteration."""

    prompt: str
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


# ── Datasets ────────────────────────────────────────────────────────

AgentEvaluationStatus = Literal["pending", "running", "completed", "failed"]


class DatasetItem(TypedDict):
    """Item in an agent dataset linking to a trace."""

    id: int
    datasetId: str
    traceDbId: str
    input: str | None
    output: str | None
    createdAt: str


class Dataset(TypedDict):
    """Agent dataset record."""

    id: str
    name: str
    aiComponentId: str
    workspaceId: str
    description: str | None
    itemCount: int
    traceCount: int
    createdAt: str
    updatedAt: str


class DatasetWithItems(Dataset):
    """Dataset with its items."""

    items: list[DatasetItem]


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


class AgentEvaluation(TypedDict):
    """Agent evaluation record."""

    id: str
    name: str | None
    aiComponentId: str
    datasetId: str
    runId: str | None
    status: AgentEvaluationStatus
    results: InsightResult | None
    startedAt: str | None
    completedAt: str | None
    createdAt: str
    updatedAt: str


class AgentEvaluationList(TypedDict):
    """List of agent evaluations."""

    data: list[AgentEvaluation]
