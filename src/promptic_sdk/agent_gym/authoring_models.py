"""Typed public models for Agent Gym benchmark authoring."""

from __future__ import annotations

import dataclasses
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict, cast

from promptic_sdk._evaluator_fields import validate_target_field
from promptic_sdk.agent_gym._validation import EvaluatorConfigurationError

FieldMethod = Literal[
    "exact", "embedding", "contains", "judge", "array_exact", "array_similarity", "array_judge"
]

_SCALAR_METHODS = {"exact", "embedding", "contains", "judge"}
_ARRAY_METHODS = {"array_exact", "array_similarity", "array_judge"}


@dataclass(frozen=True)
class VerifierMetric:
    """One verifier criterion and its contribution to benchmark scoring."""

    key: str
    name: str
    instructions: str
    weight: float = 1.0
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate stable metric identity and authoring text."""
        if not all(isinstance(value, str) for value in (self.key, self.name, self.instructions)):
            raise ValueError("verifier metric key, name, and instructions must be strings")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.key):
            raise ValueError("verifier metric key must be a snake_case identifier")
        if not self.name or not self.instructions:
            raise ValueError("verifier metric name and instructions cannot be empty")
        MetricBinding(weight=self.weight, threshold=self.threshold)

    def as_request(self) -> dict[str, str]:
        """Serialize the fixed metric definition."""
        return {"key": self.key, "name": self.name, "instructions": self.instructions}

    def scoring_binding(self) -> MetricBinding | None:
        """Return non-default aggregation settings for the wire-level binding map."""
        if self.weight == 1.0 and self.threshold is None:
            return None
        return MetricBinding(weight=self.weight, threshold=self.threshold)


@dataclass(frozen=True)
class MetricBinding:
    """Aggregation settings for one verifier metric."""

    weight: float = 1.0
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the public metric-binding contract."""
        if not isinstance(self.weight, int | float) or isinstance(self.weight, bool):
            raise ValueError("metric binding weight must be a number")
        if not 0 < self.weight <= 10:
            raise ValueError("metric binding weight must be greater than 0 and at most 10")
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("metric binding threshold must be between 0 and 1")

    def as_request(self) -> dict[str, float | None]:
        """Serialize aggregation settings without verifier-runtime details."""
        return {
            "weight": float(self.weight),
            **({"threshold": self.threshold} if self.threshold is not None else {}),
        }


class EvidenceKind(StrEnum):
    """Semantic evidence available to an Agent verifier."""

    CASE_INPUT = "case_input"
    SUBMITTED_OUTPUT = "submitted_output"
    EXPECTED_BEHAVIOR = "expected_behavior"
    EXPECTED_OUTPUT = "expected_output"
    EXECUTION_TRACE = "execution_trace"


_DEFAULT_VERIFIER_EVIDENCE = (
    EvidenceKind.CASE_INPUT,
    EvidenceKind.SUBMITTED_OUTPUT,
    EvidenceKind.EXPECTED_BEHAVIOR,
    EvidenceKind.EXPECTED_OUTPUT,
)
_EVIDENCE_REQUEST_KEYS = {
    EvidenceKind.CASE_INPUT: "caseInputs",
    EvidenceKind.SUBMITTED_OUTPUT: "submittedOutput",
    EvidenceKind.EXPECTED_BEHAVIOR: "expectedBehavior",
    EvidenceKind.EXPECTED_OUTPUT: "expectedOutput",
    EvidenceKind.EXECUTION_TRACE: "executionTrace",
}


@dataclass(frozen=True)
class EvidencePolicy:
    """Select the five public evidence kinds available to a verifier."""

    selected: Collection[EvidenceKind] = _DEFAULT_VERIFIER_EVIDENCE

    def __post_init__(self) -> None:
        """Normalize and validate semantic evidence selections."""
        selected = frozenset(self.selected)
        if any(not isinstance(value, EvidenceKind) for value in selected):
            raise ValueError("evidence selected must contain EvidenceKind values")
        object.__setattr__(self, "selected", selected)

    def as_request(self) -> dict[str, Any]:
        """Return the platform evidence configuration."""
        return {key: kind in self.selected for kind, key in _EVIDENCE_REQUEST_KEYS.items()}


@dataclass(frozen=True)
class InvestigationBudget:
    """Semantic investigation limit for an Agent verifier."""

    max_steps: int = 20

    def __post_init__(self) -> None:
        """Validate the platform-supported investigation range."""
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("investigation max_steps must be at least 1")

    def as_request(self) -> dict[str, int]:
        """Return the platform investigation-budget configuration."""
        return {"maxSteps": self.max_steps}


@dataclass(frozen=True)
class FieldScoring:
    """Scoring behavior for one structured output field."""

    method: FieldMethod = "exact"
    include: bool = True
    judge_instructions: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate field scoring before any request is made."""
        if self.method not in _SCALAR_METHODS | _ARRAY_METHODS:
            raise ValueError("unsupported field scoring method")
        if self.weight <= 0:
            raise ValueError("field scoring weight must be positive")
        if self.method not in {"judge", "array_judge"} and self.judge_instructions is not None:
            raise ValueError("judge_instructions requires method='judge' or 'array_judge'")

    def as_request(self) -> dict[str, Any]:
        """Return the platform field configuration."""
        value: dict[str, Any] = {
            "include": self.include,
            "method": self.method,
            "weight": self.weight,
        }
        if self.judge_instructions is not None:
            value["judge_instructions"] = self.judge_instructions
        return value


EvaluatorKind = Literal[
    "f1",
    "field_level_judge",
    "verifier_agent",
    "expected_behavior_judge",
    "exact_match",
    "contains_text",
    "semantic_similarity",
    "ai_judge",
    "code_function",
]
TARGETED_EVALUATOR_KINDS: frozenset[str] = frozenset(
    {"exact_match", "contains_text", "semantic_similarity", "ai_judge"}
)
"""Kinds that score one output field, or the whole response, with one method."""

MAX_TARGETED_EVALUATORS = 64
"""Most targeted evaluators one evaluation plan accepts."""

MAX_CODE_FUNCTION_EVALUATORS = 16
"""Most Code Function evaluators one evaluation plan accepts."""

MAX_CODE_FUNCTION_SOURCE_BYTES = 64 * 1024
"""Largest Code Function source, in UTF-8 bytes."""

_NAMED_REPEATABLE_EVALUATOR_KINDS = TARGETED_EVALUATOR_KINDS | {"code_function"}
_REPEATABLE_EVALUATOR_KINDS = _NAMED_REPEATABLE_EVALUATOR_KINDS | {"field_level_judge"}

FIELD_LEVEL_JUDGE_DEPRECATION = (
    "FieldLevelJudge (field_level_judge, shown as Schema match) is deprecated and can no "
    "longer be configured. Use one ExactMatch, ContainsText, SemanticSimilarity or AIJudge "
    "per output field instead. Agents that already use it keep running."
)


class AgentEvaluator(ABC):
    """Typed base for one independently configured Agent evaluator."""

    kind: ClassVar[EvaluatorKind]
    id: str | None = None
    enabled: bool = True
    name: str | None
    description: str | None
    weight: float
    required: bool
    threshold: float | None
    supports_evaluator_weight: ClassVar[bool] = True

    def _validate_common(self) -> None:
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise ValueError("evaluator id must be a non-empty string when provided")
        if not isinstance(self.enabled, bool):
            raise ValueError("evaluator enabled must be a boolean")
        if self.weight <= 0:
            raise ValueError("evaluator weight must be positive")
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("evaluator threshold must be between 0 and 1")

    @abstractmethod
    def _specific_request(self) -> dict[str, Any]:
        """Return fields specific to this evaluator kind."""

    def as_request(self) -> dict[str, Any]:
        """Serialize to the canonical Agent component authoring API."""
        return {
            "kind": self.kind,
            **({"id": self.id} if self.id is not None else {}),
            # Omitted when enabled so older deployments receive the same payload as before.
            **({"enabled": False} if not self.enabled else {}),
            **({"weight": self.weight} if self.supports_evaluator_weight else {}),
            "required": self.required,
            **({"name": self.name} if self.name else {}),
            **({"description": self.description} if self.description is not None else {}),
            **({"threshold": self.threshold} if self.threshold is not None else {}),
            **self._specific_request(),
        }


@dataclass(frozen=True)
class ClassificationF1(AgentEvaluator):
    """Classification quality for one or more enum output fields."""

    field_paths: tuple[str, ...]
    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    threshold: float | None = None
    id: str | None = field(default=None, kw_only=True)
    enabled: bool = field(default=True, kw_only=True)
    kind: ClassVar[Literal["f1"]] = "f1"

    def __post_init__(self) -> None:
        """Validate F1 configuration."""
        self._validate_common()
        if not self.field_paths:
            raise ValueError("ClassificationF1 requires at least one field path")

    def _specific_request(self) -> dict[str, Any]:
        return {"fieldPaths": list(self.field_paths)}


@dataclass(frozen=True)
class FieldLevelJudge(AgentEvaluator):
    """Deprecated field-by-field scoring for structured outputs.

    The platform no longer accepts this evaluator: creating one warns, and sending it
    raises ``EvaluatorConfigurationError``. Score each field with its own ``ExactMatch``,
    ``ContainsText``, ``SemanticSimilarity`` or ``AIJudge`` instead. The class remains so
    existing code keeps importing.
    """

    fields: Mapping[str, FieldScoring | FieldMethod] = field(default_factory=dict)
    model: str | None = None
    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    threshold: float | None = None
    id: str | None = field(default=None, kw_only=True)
    enabled: bool = field(default=True, kw_only=True)
    kind: ClassVar[Literal["field_level_judge"]] = "field_level_judge"

    def __post_init__(self) -> None:
        """Warn about the deprecation and validate the configuration."""
        warnings.warn(FIELD_LEVEL_JUDGE_DEPRECATION, DeprecationWarning, stacklevel=3)
        self._validate_common()

    def _specific_request(self) -> dict[str, Any]:
        return {
            **({"model": self.model} if self.model else {}),
            **(
                {
                    "fields": {
                        path: value.as_request()
                        if isinstance(value, FieldScoring)
                        else FieldScoring(method=value).as_request()
                        for path, value in self.fields.items()
                    }
                }
                if self.fields
                else {}
            ),
        }


@dataclass(frozen=True)
class VerifierAgent(AgentEvaluator):
    """Investigation agent that scores a result from selected evidence."""

    instructions: str | None = None
    model: str | None = None
    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    threshold: float | None = None
    evidence: EvidencePolicy = field(default_factory=EvidencePolicy)
    metrics: tuple[VerifierMetric, ...] = ()
    budget: InvestigationBudget = field(default_factory=InvestigationBudget)
    metric_bindings: Mapping[str, MetricBinding] = field(default_factory=dict)
    rubric: str | None = field(default=None, kw_only=True, repr=False)
    id: str | None = field(default=None, kw_only=True)
    enabled: bool = field(default=True, kw_only=True)
    kind: ClassVar[Literal["verifier_agent"]] = "verifier_agent"
    supports_evaluator_weight: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate verifier configuration."""
        self._validate_common()
        if self.weight != 1.0:
            raise ValueError(
                "verifier evaluator weight is unsupported; configure metric_bindings instead"
            )
        if self.rubric is not None:
            warnings.warn(
                "VerifierAgent(rubric=...) is deprecated; use instructions=... and metrics=...",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.instructions is not None and self.rubric is not None:
            raise ValueError("pass either instructions or rubric, not both")
        resolved_instructions = self.instructions or self.rubric
        if not isinstance(resolved_instructions, str) or not resolved_instructions:
            raise ValueError("verifier instructions cannot be empty")
        object.__setattr__(self, "instructions", resolved_instructions)
        metrics = tuple(self.metrics)
        if not 1 <= len(metrics) <= 8:
            raise ValueError("verifier agent requires between 1 and 8 metrics")
        if any(not isinstance(metric, VerifierMetric) for metric in metrics):
            raise ValueError("verifier metrics must contain VerifierMetric values")
        keys = [metric.key for metric in metrics]
        if len(keys) != len(set(keys)):
            raise ValueError("verifier metric keys must be unique")
        object.__setattr__(self, "metrics", metrics)
        bindings = dict(self.metric_bindings)
        if any(not isinstance(value, MetricBinding) for value in bindings.values()):
            raise ValueError("metric_bindings values must be MetricBinding instances")
        if unknown := set(bindings) - set(keys):
            raise ValueError(
                "metric_bindings reference unknown verifier metric(s): "
                + ", ".join(sorted(unknown))
            )
        inline_bindings = {
            metric.key: binding
            for metric in metrics
            if (binding := metric.scoring_binding()) is not None
        }
        if duplicates := set(bindings) & set(inline_bindings):
            raise ValueError(
                "configure verifier metric scoring inline or through metric_bindings, not both: "
                + ", ".join(sorted(duplicates))
            )
        bindings = {**bindings, **inline_bindings}
        object.__setattr__(self, "metric_bindings", bindings)

    def _specific_request(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            **({"model": self.model} if self.model else {}),
            "evidence": self.evidence.as_request(),
            "metrics": [metric.as_request() for metric in self.metrics],
            **(
                {
                    "metricBindings": {
                        key: binding.as_request() for key, binding in self.metric_bindings.items()
                    }
                }
                if self.metric_bindings
                else {}
            ),
            **(
                {"budget": self.budget.as_request()} if self.budget != InvestigationBudget() else {}
            ),
        }


@dataclass(frozen=True)
class ExpectedBehaviorJudge(AgentEvaluator):
    """Locked platform preset for judging case-specific Expected Behavior."""

    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    threshold: float | None = None
    model: str | None = None
    metric_bindings: Mapping[str, MetricBinding] = field(default_factory=dict)
    id: str | None = field(default=None, kw_only=True)
    enabled: bool = field(default=True, kw_only=True)
    kind: ClassVar[Literal["expected_behavior_judge"]] = "expected_behavior_judge"
    supports_evaluator_weight: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate evaluator-level metadata."""
        self._validate_common()
        if self.weight != 1.0:
            raise ValueError(
                "expected-behavior evaluator weight is unsupported; configure the "
                "behavior_compliance metric binding instead"
            )
        bindings = dict(self.metric_bindings)
        if any(not isinstance(value, MetricBinding) for value in bindings.values()):
            raise ValueError("metric_bindings values must be MetricBinding instances")
        if unknown := set(bindings) - {"behavior_compliance"}:
            raise ValueError(
                "expected-behavior metric_bindings support only behavior_compliance, got: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "metric_bindings", bindings)

    def _specific_request(self) -> dict[str, Any]:
        return {
            **({"model": self.model} if self.model else {}),
            **(
                {
                    "metricBindings": {
                        key: binding.as_request() for key, binding in self.metric_bindings.items()
                    }
                }
                if self.metric_bindings
                else {}
            ),
        }


@dataclass(frozen=True)
class _TargetedEvaluator(AgentEvaluator):
    """Shared configuration of the evaluators that score one field with one method."""

    field: str | None = None
    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    threshold: float | None = None
    # ``field`` names the target path in this class body, so use the qualified helper.
    id: str | None = dataclasses.field(default=None, kw_only=True)
    enabled: bool = dataclasses.field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the common settings and normalize the target field."""
        self._validate_common()
        object.__setattr__(self, "field", validate_target_field(self.field))

    def _specific_request(self) -> dict[str, Any]:
        return {"field": self.field} if self.field is not None else {}


@dataclass(frozen=True)
class ExactMatch(_TargetedEvaluator):
    """Score 1 when one output field, or the whole response, equals the expected value.

    Numbers use a small tolerance and lists are compared by their exactly equal items in
    any order. Without a ``field`` the whole response is compared, ignoring surrounding
    whitespace. Makes no model calls.
    """

    kind: ClassVar[Literal["exact_match"]] = "exact_match"


@dataclass(frozen=True)
class ContainsText(_TargetedEvaluator):
    """Score 1 when the expected text appears in one text field, or the whole response.

    Ignores case and surrounding whitespace. Makes no model calls.
    """

    kind: ClassVar[Literal["contains_text"]] = "contains_text"


@dataclass(frozen=True)
class SemanticSimilarity(_TargetedEvaluator):
    """Embedding similarity of one field, or the whole response, to the expected value.

    Shown as "Embedding Similarity" in the dashboard. Fast and cheap, but approximate: a
    negation or a wrong number in an otherwise identical text still scores high. Use
    ``AIJudge`` when correctness matters.
    """

    kind: ClassVar[Literal["semantic_similarity"]] = "semantic_similarity"


@dataclass(frozen=True)
class AIJudge(_TargetedEvaluator):
    """A model judges whether one field, or the whole response, says the same as expected.

    The judge applies a built-in equivalence rubric; ``instructions`` adds domain rules on
    top, for example "accept paraphrases, but the amount and currency must match".
    ``model`` selects a judge-eligible model; the platform default is used when omitted.
    """

    instructions: str | None = field(default=None, kw_only=True)
    model: str | None = field(default=None, kw_only=True)
    kind: ClassVar[Literal["ai_judge"]] = "ai_judge"

    def __post_init__(self) -> None:
        """Validate the judge guidance and model."""
        super().__post_init__()
        if self.instructions is not None and (
            not isinstance(self.instructions, str) or not self.instructions.strip()
        ):
            raise ValueError("AI judge instructions cannot be empty; omit them instead")
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise ValueError("AI judge model must be a non-empty string when provided")

    def _specific_request(self) -> dict[str, Any]:
        return {
            **super()._specific_request(),
            **({"instructions": self.instructions} if self.instructions is not None else {}),
            **({"model": self.model} if self.model else {}),
        }


@dataclass(frozen=True)
class CodeFunction(AgentEvaluator):
    """Score each case with your own deterministic ``score(row)`` function.

    The function runs in the platform sandbox, written in Python or JavaScript, and
    returns a finite number or ``{"score": number, "reason": str}``. Each row is a JSON
    object with ``id``, ``inputs`` (the case inputs), ``output`` (the submitted agent
    output), ``expected_output`` and ``expected_behavior`` (``None`` when absent).
    Scores are normalized to 0-1 using ``scale_min``/``scale_max``. Source is limited
    to 64 KiB; imports, network, filesystem, subprocesses and provider credentials are
    unavailable. Generated files appear in ``output`` as artifact references, and
    execution traces are not passed. A runtime error or invalid return value fails only
    that case. Sandbox compute is billed per batch of up to 100 cases.
    """

    source: str
    language: Literal["python", "javascript"] = "python"
    name: str | None = None
    description: str | None = None
    weight: float = 1.0
    required: bool = False
    scale_min: float = 0.0
    scale_max: float = 1.0
    id: str | None = field(default=None, kw_only=True)
    enabled: bool = field(default=True, kw_only=True)
    threshold: None = field(default=None, init=False)
    kind: ClassVar[Literal["code_function"]] = "code_function"

    def __post_init__(self) -> None:
        """Validate the language, source and score range."""
        self._validate_common()
        if self.language not in ("python", "javascript"):
            raise ValueError("code function language must be 'python' or 'javascript'")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("code function source cannot be empty")
        if len(self.source.encode("utf-8")) > MAX_CODE_FUNCTION_SOURCE_BYTES:
            raise ValueError("code function source must be at most 64 KiB")
        if self.scale_min == self.scale_max:
            raise ValueError("code function scale_min and scale_max must differ")

    def _specific_request(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            **(
                {"scaleMin": self.scale_min, "scaleMax": self.scale_max}
                if (self.scale_min, self.scale_max) != (0.0, 1.0)
                else {}
            ),
        }


@dataclass(frozen=True)
class BenchmarkFile:
    """A local input or private reference file for a benchmark case."""

    source: Path
    path: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        """Normalize the local source path."""
        object.__setattr__(self, "source", Path(self.source))


@dataclass(frozen=True)
class BenchmarkCase:
    """One in-memory case submitted to an editable benchmark."""

    title: str | None = None
    input: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] | None = None
    expected_behavior: str | None = None

    def __post_init__(self) -> None:
        """Normalize canonical input and expected-output mappings."""
        object.__setattr__(self, "input", dict(self.input))
        if self.output is not None:
            object.__setattr__(self, "output", dict(self.output))
        if self.title is not None and not self.title.strip():
            raise ValueError("case title cannot be empty")
        if self.expected_behavior is not None and not self.expected_behavior.strip():
            raise ValueError("expected_behavior cannot be empty")


class BenchmarkDefinition(TypedDict, total=False):
    """Editable Agent Gym benchmark definition."""

    id: str
    workspaceId: str
    aiComponentId: str
    optimizationTaskId: str
    activeRevisionId: str | None
    name: str
    description: str | None
    inputSchema: dict[str, Any] | None
    targetSchema: dict[str, Any] | None
    outputContract: dict[str, Any]
    fieldStrategyOverrides: dict[str, Any] | None
    judgeConfig: dict[str, Any] | None
    evaluatorGraph: dict[str, Any]
    templateKey: str | None
    createdBy: str | None
    createdAt: str
    updatedAt: str
    configuration: dict[str, Any]


class BenchmarkCaseFile(TypedDict, total=False):
    """Stored private file attached to a draft benchmark case."""

    artifact: str
    artifactId: str
    storageObjectId: str
    fieldPath: str
    path: str
    mime: str
    size: int
    sha256: str | None


class CreatedBenchmarkCase(TypedDict):
    """Persisted canonical benchmark case returned by the authoring API."""

    id: int
    datasetId: str
    inputFiles: list[BenchmarkCaseFile]
    inputPayload: dict[str, Any]
    expectedOutput: dict[str, Any] | None
    expectedBehavior: str | None
    outputFiles: list[BenchmarkCaseFile]
    createdAt: str
    updatedAt: str


class BulkCaseResult(TypedDict, total=False):
    """Per-case outcome from bulk authoring."""

    label: str
    status: Literal["created", "failed"]
    datasetCaseId: int
    hasExpectedOutput: bool
    hasExpectedBehavior: bool
    error: str
    issues: list[dict[str, str]]


class BulkCaseResponse(TypedDict):
    """Aggregated result from one SDK-managed bulk authoring operation."""

    total: int
    processed: int
    created: int
    failed: int
    batchesCompleted: int
    draftCaseCount: int | None
    published: Literal[False]
    results: list[BulkCaseResult]


class BulkCaseProgress(TypedDict):
    """Confirmed progress emitted after one bulk-case batch response."""

    batchIndex: int
    totalBatches: int
    total: int
    processed: int
    created: int
    failed: int
    draftCaseCount: int | None


class BenchmarkRevision(TypedDict, total=False):
    """Immutable published benchmark snapshot."""

    id: str
    datasetId: str
    workspaceId: str
    optimizationTaskId: str | None
    version: int
    status: Literal["published"]
    fingerprint: str
    caseFingerprint: str
    executionFingerprint: str
    evaluationFingerprint: str
    caseDatasetId: str
    taskSnapshot: dict[str, Any]
    evaluatorSnapshot: dict[str, Any]
    targetSchema: dict[str, Any] | None
    fieldStrategyOverrides: dict[str, Any] | None
    judgeConfig: dict[str, Any] | None
    scorerContractVersion: str
    caseCount: int
    publishedBy: str | None
    publishedAt: str
    createdAt: str


class BenchmarkDraftStatus(TypedDict, total=False):
    """Relationship between the editable draft and active revision."""

    activeRevision: BenchmarkRevision | None
    persistedDraft: BenchmarkRevisionDraft | None
    draftChanged: bool
    casesChanged: bool
    currentFingerprint: str
    currentCaseFingerprint: str
    currentExecutionFingerprint: str
    currentEvaluationFingerprint: str
    caseCount: int


class BenchmarkSchemaConflict(TypedDict):
    """One incompatible schema field grouped across affected cases."""

    direction: Literal["input", "output"]
    fieldPath: str
    issue: str
    affectedCaseIds: list[int]
    affectedCaseCount: int


class BenchmarkRevisionDraft(TypedDict, total=False):
    """Persisted benchmark proposal awaiting schema-conflict resolution."""

    id: str
    datasetId: str
    baseRevisionId: str
    workingDatasetId: str
    status: Literal["conflicted", "ready"]
    targetSchema: dict[str, Any] | None
    taskSnapshot: dict[str, Any]
    evaluatorSnapshot: dict[str, Any]
    conflicts: list[BenchmarkSchemaConflict]
    resolutions: dict[str, dict[str, Any]]
    lockVersion: int
    createdAt: str
    updatedAt: str


class BenchmarkRevisionDraftResponse(TypedDict):
    """Draft envelope returned by the revision-draft endpoints."""

    draft: BenchmarkRevisionDraft | None


class StagedBenchmarkUpdate(TypedDict):
    """Response returned when a published benchmark change is staged."""

    staged: Literal[True]
    message: str
    draft: BenchmarkRevisionDraft
    conflicts: list[BenchmarkSchemaConflict]


class BenchmarkRevisionList(TypedDict):
    """Published revisions and current draft state."""

    data: list[BenchmarkRevision]
    status: BenchmarkDraftStatus | None


@dataclass(frozen=True)
class BenchmarkRevisionResource(Mapping[str, Any]):
    """Immutable published revision exposed as a typed SDK resource."""

    data: BenchmarkRevision = field(repr=False)

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> BenchmarkRevisionResource:
        """Build a revision resource from an API response."""
        return cls(cast(BenchmarkRevision, dict(value)))

    @property
    def id(self) -> str:
        """Return the immutable revision identifier."""
        return self.data["id"]

    @property
    def version(self) -> int:
        """Return the monotonically increasing revision version."""
        return self.data["version"]

    @property
    def status(self) -> Literal["published"]:
        """Return the published revision status."""
        return self.data["status"]

    @property
    def case_count(self) -> int:
        """Return the number of frozen cases."""
        return self.data["caseCount"]

    def __getitem__(self, key: str) -> Any:
        """Return a raw response field."""
        return self.data[key]  # type: ignore[literal-required]

    def __iter__(self):
        """Iterate over raw response field names."""
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of raw response fields."""
        return len(self.data)


@dataclass(frozen=True)
class PublishedBenchmarkRevision:
    """Result of idempotently publishing the current draft fingerprint."""

    revision: BenchmarkRevisionResource
    active: bool
    created: bool

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> PublishedBenchmarkRevision:
        """Build a typed publish result from an API response."""
        revision = value.get("revision")
        if not isinstance(revision, Mapping):
            raise ValueError("publish response is missing revision")
        return cls(
            BenchmarkRevisionResource.from_response(revision),
            bool(value.get("active")),
            bool(value.get("created")),
        )


def validate_evaluators(evaluators: Sequence[AgentEvaluator]) -> None:
    """Check the evaluator collection against the server's uniqueness rules before sending it.

    Explicit IDs are unique across all evaluators. ``f1``, ``verifier_agent`` and
    ``expected_behavior_judge`` allow one evaluator each, disabled ones included.
    Targeted evaluators (``ExactMatch``, ``ContainsText``, ``SemanticSimilarity``,
    ``AIJudge``) are repeatable; they need unique names and are limited to
    ``MAX_TARGETED_EVALUATORS``. ``CodeFunction`` evaluators are repeatable, need unique
    names and are limited to ``MAX_CODE_FUNCTION_EVALUATORS``. The deprecated
    ``FieldLevelJudge`` is rejected.
    """
    seen_ids: dict[str, int] = {}
    seen_kinds: dict[str, int] = {}
    seen_names: dict[str, int] = {}
    issues: list[dict[str, Any]] = []
    targeted_count = sum(
        1 for evaluator in evaluators if evaluator.kind in TARGETED_EVALUATOR_KINDS
    )
    if targeted_count > MAX_TARGETED_EVALUATORS:
        issues.append(
            {
                "code": "too_many_targeted_evaluators",
                "path": ["evaluators"],
                "message": (
                    f"At most {MAX_TARGETED_EVALUATORS} targeted evaluators are supported; "
                    f"got {targeted_count}."
                ),
            }
        )
    code_function_count = sum(1 for evaluator in evaluators if evaluator.kind == "code_function")
    if code_function_count > MAX_CODE_FUNCTION_EVALUATORS:
        issues.append(
            {
                "code": "too_many_code_function_evaluators",
                "path": ["evaluators"],
                "message": (
                    f"At most {MAX_CODE_FUNCTION_EVALUATORS} Code Function evaluators are "
                    f"supported; got {code_function_count}."
                ),
            }
        )
    for index, evaluator in enumerate(evaluators):
        if evaluator.kind == "field_level_judge":
            issues.append(
                {
                    "code": "deprecated_evaluator_kind",
                    "path": ["evaluators", index, "kind"],
                    "kind": evaluator.kind,
                    "message": FIELD_LEVEL_JUDGE_DEPRECATION,
                }
            )
        if evaluator.id is not None:
            if evaluator.id in seen_ids:
                first_index = seen_ids[evaluator.id]
                issues.append(
                    {
                        "code": "duplicate_evaluator_id",
                        "path": ["evaluators", index, "id"],
                        "firstIndex": first_index,
                        "message": (
                            f"Evaluator ID '{evaluator.id}' must be unique; first configured at "
                            f"evaluators[{first_index}].id. Choose a distinct id or omit it "
                            "to let the server generate one."
                        ),
                    }
                )
            else:
                seen_ids[evaluator.id] = index
        kind = evaluator.kind
        name = evaluator.name.strip() if isinstance(evaluator.name, str) else ""
        if kind in _NAMED_REPEATABLE_EVALUATOR_KINDS and name:
            if name in seen_names:
                first_index = seen_names[name]
                issues.append(
                    {
                        "code": "duplicate_evaluator_name",
                        "path": ["evaluators", index, "name"],
                        "kind": kind,
                        "firstIndex": first_index,
                        "message": (
                            f"Evaluator name '{name}' is used more than once; first configured "
                            f"at evaluators[{first_index}].name. Give each targeted evaluator "
                            "and code function a unique name."
                        ),
                    }
                )
            else:
                seen_names[name] = index
        if kind in _REPEATABLE_EVALUATOR_KINDS:
            continue
        if kind not in seen_kinds:
            seen_kinds[kind] = index
            continue
        issues.append(
            {
                "code": "duplicate_evaluator_kind",
                "path": ["evaluators", index, "kind"],
                "kind": kind,
                "firstIndex": seen_kinds[kind],
                "message": (
                    f"Duplicate {kind}; first configured at evaluators[{seen_kinds[kind]}].kind. "
                    f"Only one evaluator of kind '{kind}' is supported. "
                    "Combine its configuration into one evaluator."
                ),
            }
        )
    if issues:
        raise EvaluatorConfigurationError(issues)
