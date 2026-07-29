"""High-level local runner for Agent Gym benchmark submissions."""

from __future__ import annotations

import inspect
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast
from uuid import uuid4

from promptic_sdk.agent_gym import (
    AgentGymClient,
    AsyncAgentGymClient,
    MaterializedInputFile,
)
from promptic_sdk.agent_gym_models import (
    BundleIdentity,
    ExecutionRefs,
    ExternalPrediction,
    ExternalPredictionStatus,
    RunOutput,
    SubmissionStatus,
    TokenUsage,
)


@dataclass(frozen=True)
class AgentGymOutputArtifact:
    """A local file to upload as prediction output evidence."""

    source: Path
    path: str | None = None
    mime_type: str | None = None
    role: str = "output"

    def __post_init__(self) -> None:
        """Normalize the source to a path."""
        object.__setattr__(self, "source", Path(self.source))


@dataclass(frozen=True)
class AgentGymCase:
    """One immutable benchmark case with materialized local input files."""

    id: str
    ordinal: int
    input: dict[str, Any]
    files: tuple[MaterializedInputFile, ...]
    task: dict[str, Any]

    @property
    def instructions(self) -> str:
        """Return the benchmark task instructions."""
        return str(self.task.get("description") or "")


@dataclass
class AgentGymCaseResult:
    """Candidate result for one benchmark case."""

    status: ExternalPredictionStatus = "succeeded"
    output: RunOutput | None = None
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
    def structured(
        cls,
        value: Mapping[str, Any],
        *,
        artifacts: Sequence[AgentGymOutputArtifact] = (),
        raw_trace_ids: Sequence[str] = (),
    ) -> AgentGymCaseResult:
        """Create a successful structured-output result."""
        return cls(
            output={"kind": "structured", "value": dict(value)},
            output_artifacts=tuple(artifacts),
            raw_trace_ids=tuple(raw_trace_ids),
        )

    @classmethod
    def text(
        cls,
        value: str,
        *,
        artifacts: Sequence[AgentGymOutputArtifact] = (),
        raw_trace_ids: Sequence[str] = (),
    ) -> AgentGymCaseResult:
        """Create a successful text result."""
        return cls(
            output={"kind": "text", "value": value},
            output_artifacts=tuple(artifacts),
            raw_trace_ids=tuple(raw_trace_ids),
        )

    @classmethod
    def artifact(
        cls,
        *artifacts: AgentGymOutputArtifact | str | Path,
        value: Any = None,
        summary: str | None = None,
        raw_trace_ids: Sequence[str] = (),
    ) -> AgentGymCaseResult:
        """Create a successful artifact-output result."""
        if value is not None and summary is not None:
            raise ValueError("pass either value= or summary=, not both")
        output_value = {"summary": summary} if summary is not None else value
        return cls(
            output={"kind": "artifact", "value": output_value}
            if output_value is not None
            else None,
            output_artifacts=tuple(
                artifact
                if isinstance(artifact, AgentGymOutputArtifact)
                else AgentGymOutputArtifact(Path(artifact))
                for artifact in artifacts
            ),
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
    """Finalized external benchmark run and its current scoring state."""

    submission_id: str
    revision_id: str
    run_id: str
    bundle_id: str
    status: SubmissionStatus


CandidateReturn: TypeAlias = AgentGymCaseResult | Mapping[str, Any] | str | Path
Candidate: TypeAlias = Callable[[AgentGymCase], CandidateReturn]
AsyncCandidate: TypeAlias = Callable[[AgentGymCase], CandidateReturn | Awaitable[CandidateReturn]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_candidate_result(value: CandidateReturn) -> AgentGymCaseResult:
    if isinstance(value, AgentGymCaseResult):
        return value
    if isinstance(value, Path):
        return AgentGymCaseResult.artifact(AgentGymOutputArtifact(value))
    if isinstance(value, str):
        return AgentGymCaseResult.text(value)
    if isinstance(value, Mapping):
        return AgentGymCaseResult.structured(value)
    raise TypeError(
        "candidate must return AgentGymCaseResult, a mapping, a string, or pathlib.Path"
    )


def _case_result_from_exception(error: Exception) -> AgentGymCaseResult:
    message = str(error) or error.__class__.__name__
    return AgentGymCaseResult.failed(
        error_code="candidate_exception",
        error=message[:10_000],
        error_category=error.__class__.__name__[:200],
    )


def _work_root(workdir: str | Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if workdir is not None:
        root = Path(workdir)
        root.mkdir(parents=True, exist_ok=True)
        return root, None
    temporary = tempfile.TemporaryDirectory(prefix="promptic-agent-gym-")
    return Path(temporary.name), temporary


def _files_by_case(
    files: Sequence[MaterializedInputFile],
) -> dict[str, list[MaterializedInputFile]]:
    result: dict[str, list[MaterializedInputFile]] = {}
    for item in files:
        result.setdefault(item.case_id, []).append(item)
    return result


def _bundle_identity(
    name: str,
    version: str,
    value: BundleIdentity | None,
    architecture_description: str | None,
) -> BundleIdentity:
    identity: BundleIdentity = dict(value or {})  # type: ignore[assignment]
    identity["name"] = name
    identity["version"] = version
    if architecture_description is not None:
        identity["architecture_description"] = architecture_description
    return identity


def _idempotency_prefix(value: str | None) -> str:
    prefix = value or f"python-sdk-{uuid4().hex}"
    if not prefix.strip() or len(prefix) > 180:
        raise ValueError("idempotency_key must contain 1-180 non-whitespace characters")
    return prefix.strip()


def _flush_traces() -> None:
    try:
        from opentelemetry import trace

        force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception:  # noqa: BLE001
        return


def _chunks(values: Sequence[str], size: int = 100) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _prediction(
    *,
    case: AgentGymCase,
    result: AgentGymCaseResult,
    artifact_ids: list[str],
    artifact_paths: list[str],
    measured_latency_ms: int,
    measured_started_at: str,
    measured_completed_at: str,
) -> ExternalPrediction:
    prediction: ExternalPrediction = {
        "revision_case_id": case.id,
        "status": result.status,
        "artifact_ids": artifact_ids,
        "latency_ms": result.latency_ms if result.latency_ms is not None else measured_latency_ms,
        "started_at": result.started_at or measured_started_at,
        "completed_at": result.completed_at or measured_completed_at,
    }
    if result.status == "succeeded":
        output = result.output
        if output is None and artifact_paths:
            output = cast(RunOutput, {"kind": "artifact", "value": {"paths": artifact_paths}})
        if output is None:
            output = cast(RunOutput, {"kind": "none"})
        prediction["output"] = output
    else:
        prediction["error_code"] = result.error_code or "candidate_failed"
        if result.error is not None:
            prediction["error"] = result.error
        if result.error_category is not None:
            prediction["error_category"] = result.error_category
        if result.retryable is not None:
            prediction["retryable"] = result.retryable
    if result.implementation_reference_id is not None:
        prediction["implementation_reference_id"] = result.implementation_reference_id
    if result.executor_id is not None:
        prediction["executor_id"] = result.executor_id
    if result.executor_version is not None:
        prediction["executor_version"] = result.executor_version
    if result.token_usage is not None:
        prediction["token_usage"] = result.token_usage
    if result.diagnostics is not None:
        prediction["diagnostics"] = result.diagnostics
    return prediction


def _attach_resolved_traces(
    session: Any,
    pending: Sequence[tuple[ExternalPrediction, AgentGymCaseResult]],
    *,
    max_wait: float,
    poll_interval: float,
) -> None:
    raw_ids = list(
        dict.fromkeys(trace_id for _, result in pending for trace_id in result.raw_trace_ids)
    )
    resolved: dict[str, str] = {}
    if raw_ids:
        _flush_traces()
        for chunk in _chunks(raw_ids):
            database_ids = session.wait_for_resolved_traces(
                chunk,
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
            resolved.update(zip((value.lower() for value in chunk), database_ids, strict=True))
    for prediction, result in pending:
        trace_ids = [resolved[value.lower()] for value in result.raw_trace_ids]
        refs: ExecutionRefs = {}
        if trace_ids:
            refs["trace_ids"] = trace_ids
        if result.trace_artifact_ids:
            refs["trace_artifact_ids"] = list(result.trace_artifact_ids)
        if refs:
            prediction["execution_refs"] = refs


async def _attach_resolved_traces_async(
    session: Any,
    pending: Sequence[tuple[ExternalPrediction, AgentGymCaseResult]],
    *,
    max_wait: float,
    poll_interval: float,
) -> None:
    raw_ids = list(
        dict.fromkeys(trace_id for _, result in pending for trace_id in result.raw_trace_ids)
    )
    resolved: dict[str, str] = {}
    if raw_ids:
        _flush_traces()
        for chunk in _chunks(raw_ids):
            database_ids = await session.wait_for_resolved_traces(
                chunk,
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
            resolved.update(zip((value.lower() for value in chunk), database_ids, strict=True))
    for prediction, result in pending:
        trace_ids = [resolved[value.lower()] for value in result.raw_trace_ids]
        refs: ExecutionRefs = {}
        if trace_ids:
            refs["trace_ids"] = trace_ids
        if result.trace_artifact_ids:
            refs["trace_artifact_ids"] = list(result.trace_artifact_ids)
        if refs:
            prediction["execution_refs"] = refs


def submit_benchmark(
    client: AgentGymClient,
    benchmark_id: str,
    candidate: Candidate,
    *,
    name: str,
    version: str,
    architecture_description: str | None = None,
    revision_id: str | None = None,
    bundle_identity: BundleIdentity | None = None,
    metadata: dict[str, Any] | None = None,
    workdir: str | Path | None = None,
    idempotency_key: str | None = None,
    capture_exceptions: bool = True,
    wait: bool = True,
    max_wait: float = 600,
    poll_interval: float = 2,
    trace_max_wait: float = 30,
    trace_poll_interval: float = 0.5,
) -> AgentGymRunResult:
    """Run a local candidate over a frozen manifest and submit it for scoring."""
    prefix = _idempotency_prefix(idempotency_key)
    session = client.start_submission(
        benchmark_id,
        revision_id=revision_id,
        idempotency_key=f"{prefix}:create",
    )
    root, temporary = _work_root(workdir)
    try:
        materialized = session.materialize_manifest(root / "inputs")
        by_case = _files_by_case(materialized.files)
        pending: list[tuple[ExternalPrediction, AgentGymCaseResult]] = []
        for manifest_case in materialized.manifest["data"]:
            case = AgentGymCase(
                id=manifest_case["case_id"],
                ordinal=manifest_case["ordinal"],
                input=manifest_case["input_payload"],
                files=tuple(by_case.get(manifest_case["case_id"], [])),
                task=cast(dict[str, Any], materialized.manifest["task"]),
            )
            started_at = _utc_now()
            started_clock = time.monotonic()
            try:
                result = _normalize_candidate_result(candidate(case))
            except Exception as error:
                if not capture_exceptions:
                    raise
                result = _case_result_from_exception(error)
            artifact_ids: list[str] = []
            artifact_paths: list[str] = []
            for artifact in result.output_artifacts:
                logical_name = artifact.path or artifact.source.name
                uploaded = session.upload_artifact_file(
                    artifact.source,
                    path=f"cases/{case.id}/{logical_name}",
                    mime_type=artifact.mime_type,
                    role=artifact.role,
                )
                artifact_ids.append(uploaded["artifact_id"])
                artifact_paths.append(uploaded["path"])
            prediction = _prediction(
                case=case,
                result=result,
                artifact_ids=artifact_ids,
                artifact_paths=artifact_paths,
                measured_latency_ms=int((time.monotonic() - started_clock) * 1000),
                measured_started_at=started_at,
                measured_completed_at=_utc_now(),
            )
            pending.append((prediction, result))

        _attach_resolved_traces(
            session,
            pending,
            max_wait=trace_max_wait,
            poll_interval=trace_poll_interval,
        )
        finalized = session.finalize(
            bundle_identity=_bundle_identity(
                name,
                version,
                bundle_identity,
                architecture_description,
            ),
            predictions=[prediction for prediction, _ in pending],
            idempotency_key=f"{prefix}:finalize",
            metadata=metadata,
        )
        status = (
            session.wait(max_wait=max_wait, poll_interval=poll_interval)
            if wait
            else session.status()
        )
        return AgentGymRunResult(
            submission_id=finalized["submission_id"],
            revision_id=session.revision_id or materialized.manifest["revision"]["id"],
            run_id=finalized["run_id"],
            bundle_id=finalized["bundle_id"],
            status=status,
        )
    except Exception:
        with suppress(Exception):
            session.cancel()
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


async def submit_benchmark_async(
    client: AsyncAgentGymClient,
    benchmark_id: str,
    candidate: AsyncCandidate,
    *,
    name: str,
    version: str,
    architecture_description: str | None = None,
    revision_id: str | None = None,
    bundle_identity: BundleIdentity | None = None,
    metadata: dict[str, Any] | None = None,
    workdir: str | Path | None = None,
    idempotency_key: str | None = None,
    capture_exceptions: bool = True,
    wait: bool = True,
    max_wait: float = 600,
    poll_interval: float = 2,
    trace_max_wait: float = 30,
    trace_poll_interval: float = 0.5,
) -> AgentGymRunResult:
    """Run a local async candidate over a frozen manifest and submit it for scoring."""
    prefix = _idempotency_prefix(idempotency_key)
    session = await client.start_submission(
        benchmark_id,
        revision_id=revision_id,
        idempotency_key=f"{prefix}:create",
    )
    root, temporary = _work_root(workdir)
    try:
        materialized = await session.materialize_manifest(root / "inputs")
        by_case = _files_by_case(materialized.files)
        pending: list[tuple[ExternalPrediction, AgentGymCaseResult]] = []
        for manifest_case in materialized.manifest["data"]:
            case = AgentGymCase(
                id=manifest_case["case_id"],
                ordinal=manifest_case["ordinal"],
                input=manifest_case["input_payload"],
                files=tuple(by_case.get(manifest_case["case_id"], [])),
                task=cast(dict[str, Any], materialized.manifest["task"]),
            )
            started_at = _utc_now()
            started_clock = time.monotonic()
            try:
                candidate_value = candidate(case)
                if inspect.isawaitable(candidate_value):
                    candidate_value = await cast(Awaitable[CandidateReturn], candidate_value)
                result = _normalize_candidate_result(candidate_value)
            except Exception as error:
                if not capture_exceptions:
                    raise
                result = _case_result_from_exception(error)
            artifact_ids: list[str] = []
            artifact_paths: list[str] = []
            for artifact in result.output_artifacts:
                logical_name = artifact.path or artifact.source.name
                uploaded = await session.upload_artifact_file(
                    artifact.source,
                    path=f"cases/{case.id}/{logical_name}",
                    mime_type=artifact.mime_type,
                    role=artifact.role,
                )
                artifact_ids.append(uploaded["artifact_id"])
                artifact_paths.append(uploaded["path"])
            prediction = _prediction(
                case=case,
                result=result,
                artifact_ids=artifact_ids,
                artifact_paths=artifact_paths,
                measured_latency_ms=int((time.monotonic() - started_clock) * 1000),
                measured_started_at=started_at,
                measured_completed_at=_utc_now(),
            )
            pending.append((prediction, result))

        await _attach_resolved_traces_async(
            session,
            pending,
            max_wait=trace_max_wait,
            poll_interval=trace_poll_interval,
        )
        finalized = await session.finalize(
            bundle_identity=_bundle_identity(
                name,
                version,
                bundle_identity,
                architecture_description,
            ),
            predictions=[prediction for prediction, _ in pending],
            idempotency_key=f"{prefix}:finalize",
            metadata=metadata,
        )
        status = (
            await session.wait(max_wait=max_wait, poll_interval=poll_interval)
            if wait
            else await session.status()
        )
        return AgentGymRunResult(
            submission_id=finalized["submission_id"],
            revision_id=session.revision_id or materialized.manifest["revision"]["id"],
            run_id=finalized["run_id"],
            bundle_id=finalized["bundle_id"],
            status=status,
        )
    except Exception:
        with suppress(Exception):
            await session.cancel()
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()
