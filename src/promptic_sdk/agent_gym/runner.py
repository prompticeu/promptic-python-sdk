"""Trusted callback runner for Agent Gym external submissions."""

from __future__ import annotations

import inspect
import tempfile
import time
import types
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import MISSING, asdict, dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    NotRequired,
    Required,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)
from uuid import uuid4

from promptic_sdk.agent_gym.models import (
    ExecutionRefs,
    ExternalPrediction,
    ExternalPredictionStatus,
    ExternalTaskSnapshot,
    SubmissionStatus,
    TokenUsage,
    VariantIdentity,
)
from promptic_sdk.agent_gym.submissions import (
    MaterializedInputFile,
    TracePolicy,
    materialized_case_input,
)


@dataclass(frozen=True)
class AgentGymOutputArtifact:
    """A local file referenced from a canonical output-schema field."""

    source: Path
    field_path: str
    path: str | None = None
    mime_type: str | None = None
    role: str = "output"

    def __post_init__(self) -> None:
        """Normalize the source to a Path."""
        object.__setattr__(self, "source", Path(self.source))
        if not self.field_path.strip():
            raise ValueError("field_path must identify an Output schema field")


CaseInputT = TypeVar("CaseInputT")
StructuredOutputT = TypeVar("StructuredOutputT")


@dataclass(frozen=True)
class AgentGymCase(Generic[CaseInputT]):
    """One immutable benchmark case with materialized local input files."""

    id: int
    ordinal: int
    input: CaseInputT
    task: ExternalTaskSnapshot

    @property
    def instructions(self) -> str:
        """Return the task's public instructions."""
        return str(self.task.get("description") or "")


@dataclass
class AgentGymCaseResult(Generic[StructuredOutputT]):
    """Candidate result for one benchmark case."""

    status: ExternalPredictionStatus = "succeeded"
    output: Any = None
    output_artifacts: tuple[AgentGymOutputArtifact, ...] = ()
    raw_trace_ids: tuple[str, ...] = ()
    trace_artifact_ids: tuple[str, ...] = ()
    implementation_reference_id: str | None = None
    executor_id: str | None = None
    executor_version: str | None = None
    token_usage: TokenUsage | None = None
    latency_ms: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_code: str | None = None
    error_category: str | None = None
    retryable: bool | None = None
    error: str | None = None
    diagnostics: dict[str, Any] | None = None

    @classmethod
    def succeeded(
        cls,
        output: StructuredOutputT,
        *,
        artifacts: Sequence[AgentGymOutputArtifact] = (),
        raw_trace_ids: Sequence[str] = (),
    ) -> AgentGymCaseResult[StructuredOutputT]:
        """Create a successful result matching the benchmark's Output schema."""
        return cls(
            output=_structured_output(output),
            output_artifacts=tuple(artifacts),
            raw_trace_ids=tuple(raw_trace_ids),
        )

    @classmethod
    def artifact(
        cls,
        *artifacts: AgentGymOutputArtifact,
        output: Mapping[str, Any] | None = None,
        raw_trace_ids: Sequence[str] = (),
    ) -> AgentGymCaseResult:
        """Create a result whose files are inserted at declared Output schema paths."""
        return cls(
            output=dict(output or {}),
            output_artifacts=tuple(artifacts),
            raw_trace_ids=tuple(raw_trace_ids),
        )

    @classmethod
    def failed(
        cls,
        *,
        error_code: str,
        error: str,
        error_category: str | None = None,
        retryable: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> AgentGymCaseResult:
        """Create a terminal failed result."""
        return cls(
            status="failed",
            error_code=error_code,
            error=error,
            error_category=error_category,
            retryable=retryable,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class AgentGymRunResult:
    """Finalized external benchmark run and current scoring state."""

    submission_id: str
    revision_id: str
    run_id: str
    variant_id: str
    status: SubmissionStatus


CandidateReturn: TypeAlias = AgentGymCaseResult[Any] | Mapping[str, Any] | str
Candidate: TypeAlias = Callable[[AgentGymCase[CaseInputT]], CandidateReturn]
AsyncCandidate: TypeAlias = Callable[
    [AgentGymCase[CaseInputT]], CandidateReturn | Awaitable[CandidateReturn]
]


def _structured_output(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _schema_node(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Required, NotRequired}:
        return _schema_node(args[0])
    if annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is MaterializedInputFile:
        return {"type": "string", "x-promptic-type": "file"}
    if origin is Literal:
        values = list(args)
        node: dict[str, Any] = {"enum": values}
        if values and all(isinstance(item, str) for item in values):
            node["type"] = "string"
        return node
    if origin in {list, tuple, Sequence}:
        item = args[0] if args else Any
        if item is MaterializedInputFile:
            return {"type": "array", "x-promptic-type": "file"}
        return {"type": "array", "items": _schema_node(item)}
    if origin in {dict, Mapping}:
        return {"type": "object", "additionalProperties": True}
    if origin in {Union, types.UnionType}:
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1:
            return _schema_node(non_null[0])
        return {"anyOf": [_schema_node(item) for item in non_null]}
    if is_typeddict(annotation):
        hints = get_type_hints(annotation, include_extras=True)
        required = set(getattr(annotation, "__required_keys__", ()))
        return {
            "type": "object",
            "properties": {name: _schema_node(value) for name, value in hints.items()},
            "required": [name for name in hints if name in required],
            "additionalProperties": False,
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        hints = get_type_hints(annotation, include_extras=True)
        return {
            "type": "object",
            "properties": {name: _schema_node(value) for name, value in hints.items()},
            "required": [
                item.name
                for item in dataclass_fields(annotation)
                if item.default is MISSING and item.default_factory is MISSING
            ],
            "additionalProperties": False,
        }
    model_schema = getattr(annotation, "model_json_schema", None)
    if callable(model_schema):
        return cast(dict[str, Any], model_schema())
    raise TypeError(f"unsupported schema annotation: {annotation!r}")


def schema_for_model(model: type[Any]) -> dict[str, Any]:
    """Derive JSON Schema from a dataclass, TypedDict, or Pydantic model."""
    schema = _schema_node(model)
    if schema.get("type") != "object":
        raise TypeError("agent case models must describe an object")
    return schema


def _bind_input_model(value: dict[str, Any], model: type[CaseInputT] | None) -> Any:
    if model is None or is_typeddict(model):
        return value
    model_validate = getattr(model, "model_validate", None)
    if callable(model_validate):
        return model_validate(value)
    if is_dataclass(model):
        return model(**value)
    return model(**value)


def _candidate_input_model(candidate: Callable[..., Any]) -> type[Any] | None:
    """Infer the case input model from a callback's first parameter annotation."""
    target: Any = candidate
    if not inspect.isfunction(target) and not inspect.ismethod(target):
        target = getattr(target, "__call__", target)
    try:
        signature = inspect.signature(target)
        hints = get_type_hints(target, include_extras=True)
    except (NameError, TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if not parameters:
        return None
    annotation = hints.get(parameters[0].name)
    if get_origin(annotation) is not AgentGymCase:
        return None
    arguments = get_args(annotation)
    model = arguments[0] if len(arguments) == 1 else None
    return model if isinstance(model, type) else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_result(value: CandidateReturn) -> AgentGymCaseResult:
    if isinstance(value, AgentGymCaseResult):
        return value
    if isinstance(value, (str, Mapping)):
        return AgentGymCaseResult.succeeded(value)
    raise TypeError("candidate must return AgentGymCaseResult, a mapping, or a string")


def _exception_result(error: Exception) -> AgentGymCaseResult:
    return AgentGymCaseResult.failed(
        error_code="candidate_exception",
        error=(str(error) or error.__class__.__name__)[:10_000],
        error_category=error.__class__.__name__[:200],
    )


def _work_root(
    workdir: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, None
    temporary = tempfile.TemporaryDirectory(prefix="promptic-agent-gym-")
    return Path(temporary.name), temporary


def _files_by_case(
    files: Sequence[MaterializedInputFile],
) -> dict[int, list[MaterializedInputFile]]:
    grouped: dict[int, list[MaterializedInputFile]] = {}
    for item in files:
        grouped.setdefault(item.dataset_case_id, []).append(item)
    return grouped


def _variant_identity(
    name: str,
    version: str,
    description: str,
    value: VariantIdentity | None,
    repository_url: str | None,
    commit_hash: str | None,
) -> VariantIdentity:
    identity = cast(VariantIdentity, dict(value or {}))
    identity["name"] = name
    identity["version"] = version
    identity["architecture_description"] = description
    if repository_url is not None:
        identity["repository_url"] = repository_url
    if commit_hash is not None:
        identity["commit_hash"] = commit_hash
    return identity


def _idempotency_prefix(value: str | None) -> str:
    prefix = (value or f"python-sdk-{uuid4().hex}").strip()
    if not prefix or len(prefix) > 180:
        raise ValueError("idempotency_key must contain 1-180 non-whitespace characters")
    return prefix


def _flush_traces() -> None:
    try:
        from opentelemetry import trace

        force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception:  # noqa: BLE001
        return


@contextmanager
def _case_trace(
    enabled: bool,
    *,
    benchmark_id: str,
    submission_id: str,
    revision_id: str,
    case: AgentGymCase,
) -> Iterator[str | None]:
    """Create an independent Promptic case trace when tracing is configured."""
    from promptic_sdk.tracing import is_tracing_configured

    if not enabled or not is_tracing_configured():
        yield None
        return

    from opentelemetry import trace
    from opentelemetry.context import Context

    tracer = trace.get_tracer("promptic_sdk.agent_gym")
    with tracer.start_as_current_span(
        "agent_gym.case",
        context=Context(),
        attributes={
            "promptic.agent_gym.benchmark_id": benchmark_id,
            "promptic.agent_gym.submission_id": submission_id,
            "promptic.agent_gym.revision_id": revision_id,
            "promptic.agent_gym.case_id": case.id,
            "promptic.agent_gym.case_ordinal": case.ordinal,
        },
    ) as span:
        span_context = span.get_span_context()
        raw_trace_id = (
            f"{span_context.trace_id:032x}"
            if span.is_recording() and span_context.is_valid and span_context.trace_id != 0
            else None
        )
        yield raw_trace_id


def _merge_generated_trace(
    result: AgentGymCaseResult, generated_trace_id: str | None
) -> AgentGymCaseResult:
    if generated_trace_id is not None:
        result.raw_trace_ids = tuple(dict.fromkeys((generated_trace_id, *result.raw_trace_ids)))
    return result


def _set_output_path(output: dict[str, Any], field_path: str, value: Any) -> None:
    segments = [segment for segment in field_path.split(".") if segment]
    if not segments:
        raise ValueError("artifact field_path must identify an Output schema field")
    current = output
    for segment in segments[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child
    current[segments[-1]] = value


def _output_with_artifacts(output: Any, artifact_references: Sequence[tuple[str, str]]) -> Any:
    if not artifact_references:
        return output
    if output is None:
        canonical: dict[str, Any] = {}
    elif isinstance(output, Mapping):
        canonical = dict(output)
    else:
        raise ValueError("file outputs require an object matching the Output schema")
    grouped: dict[str, list[str]] = {}
    for field_path, artifact_id in artifact_references:
        grouped.setdefault(field_path, []).append(f"promptic-artifact://{artifact_id}")
    for field_path, references in grouped.items():
        _set_output_path(canonical, field_path, references)
    return canonical


def prediction_from_result(
    case_id: int,
    result: AgentGymCaseResult,
    artifact_ids: list[str],
    artifact_references: list[tuple[str, str]],
    *,
    measured_latency_ms: int | None = None,
    measured_started_at: str | None = None,
    measured_completed_at: str | None = None,
) -> ExternalPrediction:
    """Compile a callback result and uploaded artifacts into a prediction."""
    prediction: ExternalPrediction = {
        "dataset_case_id": case_id,
        "status": result.status,
        "artifact_ids": artifact_ids,
    }
    latency_ms = result.latency_ms if result.latency_ms is not None else measured_latency_ms
    if latency_ms is not None:
        prediction["latency_ms"] = latency_ms
    started_at = result.started_at or measured_started_at
    if started_at is not None:
        prediction["started_at"] = started_at
    completed_at = result.completed_at or measured_completed_at
    if completed_at is not None:
        prediction["completed_at"] = completed_at
    if result.status == "succeeded":
        prediction["output"] = _output_with_artifacts(result.output, artifact_references)
    else:
        prediction["error_code"] = result.error_code or "candidate_failed"
        if result.error is not None:
            prediction["error"] = result.error
        if result.error_category is not None:
            prediction["error_category"] = result.error_category
        if result.retryable is not None:
            prediction["retryable"] = result.retryable
    for key in ("implementation_reference_id", "executor_id", "executor_version"):
        value = getattr(result, key)
        if value is not None:
            prediction[key] = value
    if result.token_usage is not None:
        prediction["token_usage"] = result.token_usage
    if result.diagnostics is not None:
        prediction["diagnostics"] = result.diagnostics
    refs: ExecutionRefs = {}
    if result.trace_artifact_ids:
        refs["trace_artifact_ids"] = list(result.trace_artifact_ids)
    if refs:
        prediction["execution_refs"] = refs
    return prediction


def submit_benchmark(
    client: Any,
    benchmark_id: str,
    candidate: Candidate[CaseInputT],
    *,
    name: str,
    version: str,
    architecture_description: str,
    repository_url: str | None,
    commit_hash: str | None,
    revision_id: str | None,
    variant_identity: VariantIdentity | None,
    metadata: dict[str, Any] | None,
    workdir: Path | None,
    idempotency_key: str | None,
    capture_exceptions: bool,
    wait: bool,
    max_wait: float,
    poll_interval: float,
    trace_max_wait: float,
    trace_poll_interval: float,
    trace_cases: bool,
    trace_policy: TracePolicy,
    input_model: type[CaseInputT] | None = None,
) -> AgentGymRunResult:
    """Execute a trusted synchronous callback over a frozen manifest."""
    resolved_input_model = input_model or _candidate_input_model(candidate)
    prefix = _idempotency_prefix(idempotency_key)
    identity = _variant_identity(
        name,
        version,
        architecture_description,
        variant_identity,
        repository_url,
        commit_hash,
    )
    session = client.start_submission(
        benchmark_id,
        idempotency_key=f"{prefix}-session",
        variant_identity=identity,
        revision_id=revision_id,
    )
    root, temporary = _work_root(workdir)
    try:
        materialized = session.materialize_manifest(root / "manifest")
        grouped = _files_by_case(materialized.files)
        for manifest_case in materialized.manifest["data"]:
            case_files = tuple(grouped.get(manifest_case["dataset_case_id"], []))
            case = AgentGymCase(
                manifest_case["dataset_case_id"],
                manifest_case["ordinal"],
                _bind_input_model(
                    materialized_case_input(manifest_case["input_payload"], case_files),
                    resolved_input_model,
                ),
                materialized.manifest["task"],
            )
            started_at = _utc_now()
            started = time.monotonic()
            generated_trace_id: str | None = None
            try:
                with _case_trace(
                    trace_cases,
                    benchmark_id=benchmark_id,
                    submission_id=session.submission_id,
                    revision_id=session.revision_id or materialized.manifest["revision"]["id"],
                    case=case,
                ) as generated_trace_id:
                    result = _normalize_result(candidate(case))
            except Exception as error:
                if not capture_exceptions:
                    raise
                result = _exception_result(error)
            result = _merge_generated_trace(result, generated_trace_id)
            completed_at = _utc_now()
            latency_ms = round((time.monotonic() - started) * 1000)
            result = replace(
                result,
                latency_ms=result.latency_ms if result.latency_ms is not None else latency_ms,
                started_at=result.started_at or started_at,
                completed_at=result.completed_at or completed_at,
            )
            session.add_prediction(case.id, result)

        submitted = session.submit(
            identity=identity,
            metadata=metadata,
            idempotency_key=f"{prefix}-submit",
            trace_policy=trace_policy,
            trace_max_wait=trace_max_wait,
            trace_poll_interval=trace_poll_interval,
        )
        status = (
            session.wait(max_wait=max_wait, poll_interval=poll_interval)
            if wait
            else session.status()
        )
        return AgentGymRunResult(
            session.submission_id,
            session.revision_id or materialized.manifest["revision"]["id"],
            submitted["run_id"],
            submitted["variant_id"],
            status,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


async def submit_benchmark_async(
    client: Any,
    benchmark_id: str,
    candidate: AsyncCandidate[CaseInputT],
    *,
    name: str,
    version: str,
    architecture_description: str,
    repository_url: str | None,
    commit_hash: str | None,
    revision_id: str | None,
    variant_identity: VariantIdentity | None,
    metadata: dict[str, Any] | None,
    workdir: Path | None,
    idempotency_key: str | None,
    capture_exceptions: bool,
    wait: bool,
    max_wait: float,
    poll_interval: float,
    trace_max_wait: float,
    trace_poll_interval: float,
    trace_cases: bool,
    trace_policy: TracePolicy,
    input_model: type[CaseInputT] | None = None,
) -> AgentGymRunResult:
    """Execute a trusted sync or async callback over a frozen manifest."""
    resolved_input_model = input_model or _candidate_input_model(candidate)
    prefix = _idempotency_prefix(idempotency_key)
    identity = _variant_identity(
        name,
        version,
        architecture_description,
        variant_identity,
        repository_url,
        commit_hash,
    )
    session = await client.start_submission(
        benchmark_id,
        idempotency_key=f"{prefix}-session",
        variant_identity=identity,
        revision_id=revision_id,
    )
    root, temporary = _work_root(workdir)
    try:
        materialized = await session.materialize_manifest(root / "manifest")
        grouped = _files_by_case(materialized.files)
        for manifest_case in materialized.manifest["data"]:
            case_files = tuple(grouped.get(manifest_case["dataset_case_id"], []))
            case = AgentGymCase(
                manifest_case["dataset_case_id"],
                manifest_case["ordinal"],
                _bind_input_model(
                    materialized_case_input(manifest_case["input_payload"], case_files),
                    resolved_input_model,
                ),
                materialized.manifest["task"],
            )
            started_at = _utc_now()
            started = time.monotonic()
            generated_trace_id: str | None = None
            try:
                with _case_trace(
                    trace_cases,
                    benchmark_id=benchmark_id,
                    submission_id=session.submission_id,
                    revision_id=session.revision_id or materialized.manifest["revision"]["id"],
                    case=case,
                ) as generated_trace_id:
                    value = candidate(case)
                    if inspect.isawaitable(value):
                        value = await value
                    result = _normalize_result(cast(CandidateReturn, value))
            except Exception as error:
                if not capture_exceptions:
                    raise
                result = _exception_result(error)
            result = _merge_generated_trace(result, generated_trace_id)
            completed_at = _utc_now()
            latency_ms = round((time.monotonic() - started) * 1000)
            result = replace(
                result,
                latency_ms=result.latency_ms if result.latency_ms is not None else latency_ms,
                started_at=result.started_at or started_at,
                completed_at=result.completed_at or completed_at,
            )
            await session.add_prediction(case.id, result)

        submitted = await session.submit(
            identity=identity,
            metadata=metadata,
            idempotency_key=f"{prefix}-submit",
            trace_policy=trace_policy,
            trace_max_wait=trace_max_wait,
            trace_poll_interval=trace_poll_interval,
        )
        status = (
            await session.wait(max_wait=max_wait, poll_interval=poll_interval)
            if wait
            else await session.status()
        )
        return AgentGymRunResult(
            session.submission_id,
            session.revision_id or materialized.manifest["revision"]["id"],
            submitted["run_id"],
            submitted["variant_id"],
            status,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
