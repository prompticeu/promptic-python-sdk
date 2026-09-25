"""Typed synchronous and asynchronous Agent Gym clients."""

from __future__ import annotations

import asyncio
import math
import mimetypes
import os
import time
import warnings
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx

from promptic_sdk.agent_gym._deadline import bounded_call, bounded_call_async, remaining
from promptic_sdk.agent_gym._http import (
    AgentGymAPIError,
    ArtifactTransferError,
    AsyncTransport,
    SyncTransport,
    resolve_client_config,
)
from promptic_sdk.agent_gym.artifacts import (
    atomic_write_bytes,
    download_manifest_input_async,
    download_manifest_input_sync,
    download_prediction_artifact_async,
    download_prediction_artifact_sync,
    read_artifact,
    sha256_bytes,
    upload_reserved_async,
    upload_reserved_sync,
)
from promptic_sdk.agent_gym.authoring import AgentGymBenchmarks, AsyncAgentGymBenchmarks
from promptic_sdk.agent_gym.models import (
    BenchmarkCaseResult,
    BenchmarkCaseResultPage,
    BenchmarkRunComparison,
    BenchmarkRunResults,
    CancelledSubmission,
    CaseResultSort,
    CompletedSubmissionArtifact,
    ExternalPrediction,
    ExternalSubmissionCreated,
    ExternalSubmissionManifest,
    ManifestCase,
    ManifestPage,
    PredictionArtifact,
    QueuedBenchmarkReevaluation,
    ReservedSubmissionArtifact,
    RetriedBenchmarkScoring,
    RevisionManifest,
    RevisionManifestPage,
    StagedPredictionBatch,
    SubmissionArtifact,
    SubmissionStatus,
    SubmitSubmissionRequest,
    SubmittedSubmission,
    TraceResolutionList,
    VariantIdentity,
)
from promptic_sdk.agent_gym.results import (
    case_result_request,
    case_results_request,
    compare_runs_request,
    prediction_artifact_request,
    reevaluate_run_request,
    retry_scoring_request,
    run_results_request,
)
from promptic_sdk.agent_gym.submissions import (
    MAX_ARTIFACT_BYTES,
    TERMINAL_SUBMISSION_STATUSES,
    ArtifactIntegrityError,
    DownloadedBenchmarkDataset,
    MaterializedInputFile,
    MaterializedManifest,
    SubmissionPredictionBuilder,
    TracePolicy,
    UnresolvedTraceError,
    cancel_submission_request,
    collect_manifest,
    collect_revision_manifest,
    complete_artifact_request,
    create_submission_request,
    downloaded_dataset,
    manifest_page_request,
    materialized_path,
    normalize_trace_ids,
    normalize_trace_policy,
    prediction_upload_batches,
    require_uuid,
    reserve_artifact_request,
    resolve_traces_request,
    revision_manifest_page_request,
    sanitized_manifest_json,
    submission_status_request,
    submit_submission_request,
    upload_prediction_batch_request,
    validate_sha256,
)

if TYPE_CHECKING:
    from promptic_sdk.agent_gym.runner import (
        AgentGymCaseResult,
        AgentGymRunResult,
        AsyncCandidate,
        Candidate,
    )

CaseInputT = TypeVar("CaseInputT")
_PREDICTION_AUTO_FLUSH_SIZE = 1
_PREDICTION_UPLOAD_MAX_ATTEMPTS = 3
_PREDICTION_UPLOAD_RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def _validate_wait(max_wait: float, poll_interval: float) -> None:
    if not math.isfinite(max_wait) or max_wait < 0:
        raise ValueError("max_wait must be non-negative")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be positive")


def _trace_chunks(trace_ids: Sequence[str], size: int = 100) -> list[list[str]]:
    return [list(trace_ids[index : index + size]) for index in range(0, len(trace_ids), size)]


def _retryable_prediction_upload(error: Exception) -> bool:
    return isinstance(error, httpx.TransportError) or (
        isinstance(error, AgentGymAPIError)
        and error.status_code in _PREDICTION_UPLOAD_RETRY_STATUS_CODES
    )


def _warn_omitted_traces(trace_ids: Sequence[str], error: Exception | None = None) -> None:
    if not trace_ids:
        return
    detail = f" Last error: {error}." if error is not None else ""
    warnings.warn(
        f"Omitted {len(trace_ids)} unresolved trace link(s) under "
        f"trace_policy='best_effort'; predictions will still be submitted.{detail}",
        RuntimeWarning,
        stacklevel=3,
    )


class AgentGymClient:
    """Synchronous Agent Gym client using normal Promptic authentication."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        access_token: str | None = None,
        ai_application_id: str | None = None,
        workspace_id: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client from explicit, environment, or saved credentials."""
        if ai_application_id and workspace_id and ai_application_id != workspace_id:
            raise ValueError("ai_application_id and workspace_id cannot disagree")
        config = resolve_client_config(
            api_key=api_key,
            access_token=access_token,
            ai_application_id=ai_application_id or workspace_id,
            endpoint=endpoint,
        )
        self.endpoint = config.endpoint
        self._transport = SyncTransport(config, timeout)
        self._client = self._transport.api
        self._direct_client = self._transport.direct
        self.benchmarks = AgentGymBenchmarks(
            self,
            ai_application_id
            or workspace_id
            or os.environ.get("PROMPTIC_AI_APPLICATION_ID")
            or os.environ.get("PROMPTIC_WORKSPACE_ID"),
        )

    def create_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        variant_identity: VariantIdentity,
        revision_id: str | None = None,
        ttl_seconds: int = 86_400,
    ) -> ExternalSubmissionCreated:
        """Create or idempotently replay an external submission session."""
        return self._transport.request(
            create_submission_request(
                benchmark_id,
                idempotency_key=idempotency_key,
                variant_identity=variant_identity,
                revision_id=revision_id,
                ttl_seconds=ttl_seconds,
            )
        )

    def start_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        variant_identity: VariantIdentity,
        revision_id: str | None = None,
        ttl_seconds: int = 86_400,
    ) -> ExternalSubmissionSession:
        """Create a submission and return a bound session helper."""
        created = self.create_submission(
            benchmark_id,
            idempotency_key=idempotency_key,
            variant_identity=variant_identity,
            revision_id=revision_id,
            ttl_seconds=ttl_seconds,
        )
        return ExternalSubmissionSession(
            self,
            benchmark_id,
            created["submission_id"],
            created["revision"]["id"],
            created,
        )

    def resume_submission(self, benchmark_id: str, submission_id: str) -> ExternalSubmissionSession:
        """Bind a session helper to an existing submission."""
        require_uuid(benchmark_id, "benchmark_id")
        require_uuid(submission_id, "submission_id")
        return ExternalSubmissionSession(self, benchmark_id, submission_id)

    def get_manifest_page(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ManifestPage:
        """Fetch one immutable manifest page."""
        return self._transport.request(
            manifest_page_request(benchmark_id, submission_id, cursor=cursor, limit=limit)
        )

    def iter_manifest_cases(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> Iterator[ManifestCase]:
        """Iterate every immutable manifest case."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            yield from page["data"]
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen.add(cursor)

    def get_manifest(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> ExternalSubmissionManifest:
        """Collect and validate every immutable manifest page."""
        first = self.get_manifest_page(benchmark_id, submission_id, limit=page_size)
        cursor = first["next_cursor"]
        seen: set[str] = set()
        pages: list[ManifestPage] = []
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen.add(cursor)
            page = self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            pages.append(page)
            cursor = page["next_cursor"]
        return collect_manifest(first, pages)

    def materialize_manifest(
        self,
        benchmark_id: str,
        submission_id: str,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download manifest inputs beneath a safe local root."""
        manifest = self.get_manifest(benchmark_id, submission_id, page_size=page_size)
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = materialized_path(root, case, input_file)
                download_manifest_input_sync(
                    self._transport, input_file, local_path, overwrite=overwrite
                )
                files.append(
                    MaterializedInputFile(
                        case["dataset_case_id"],
                        input_file["artifact_id"],
                        input_file["field_path"],
                        input_file["path"],
                        local_path,
                        input_file["mime_type"],
                        input_file["size_bytes"],
                        input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        atomic_write_bytes(manifest_path, sanitized_manifest_json(manifest, files, root))
        return MaterializedManifest(root, manifest_path, tuple(files), manifest)

    def download_dataset(
        self,
        benchmark_id: str,
        destination: str | os.PathLike[str],
        *,
        revision_id: str | None = None,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> DownloadedBenchmarkDataset:
        """Download public frozen inputs without creating a submission or run."""
        first = cast(
            RevisionManifestPage,
            self._transport.request(
                revision_manifest_page_request(
                    benchmark_id, revision_id, cursor=None, limit=page_size
                )
            ),
        )
        pages: list[RevisionManifestPage] = []
        cursor = first["next_cursor"]
        seen_cursors: set[str] = set()
        while cursor is not None:
            if cursor in seen_cursors:
                raise RuntimeError("revision manifest pagination repeated a cursor")
            seen_cursors.add(cursor)
            page = cast(
                RevisionManifestPage,
                self._transport.request(
                    revision_manifest_page_request(
                        benchmark_id, revision_id, cursor=cursor, limit=page_size
                    )
                ),
            )
            pages.append(page)
            cursor = page["next_cursor"]
        manifest = collect_revision_manifest(first, pages)
        materialized = self._materialize_revision_manifest(
            manifest, destination, overwrite=overwrite
        )
        return downloaded_dataset(materialized)

    def _materialize_revision_manifest(
        self,
        manifest: RevisionManifest,
        destination: str | os.PathLike[str],
        *,
        overwrite: bool,
    ) -> MaterializedManifest:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = materialized_path(root, case, input_file)
                download_manifest_input_sync(
                    self._transport, input_file, local_path, overwrite=overwrite
                )
                files.append(
                    MaterializedInputFile(
                        case["dataset_case_id"],
                        input_file["artifact_id"],
                        input_file["field_path"],
                        input_file["path"],
                        local_path,
                        input_file["mime_type"],
                        input_file["size_bytes"],
                        input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        atomic_write_bytes(manifest_path, sanitized_manifest_json(manifest, files, root))
        return MaterializedManifest(root, manifest_path, tuple(files), manifest)

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
        """Reserve a submission-owned output artifact."""
        return self._transport.request(
            reserve_artifact_request(
                benchmark_id,
                submission_id,
                path=path,
                mime_type=mime_type,
                size_bytes=size_bytes,
                sha256=sha256,
                role=role,
            )
        )

    def upload_reserved_artifact(
        self,
        reservation: ReservedSubmissionArtifact,
        content: bytes,
        *,
        mime_type: str,
        filename: str | None = None,
    ) -> None:
        """Upload bytes to a reservation's credential-free storage target."""
        upload_reserved_sync(
            self._transport,
            reservation,
            content,
            mime_type=mime_type,
            filename=filename or Path(reservation["path"]).name,
        )

    def complete_artifact(
        self, benchmark_id: str, submission_id: str, artifact_id: str
    ) -> CompletedSubmissionArtifact:
        """Ask the platform to verify an uploaded artifact."""
        return self._transport.request(
            complete_artifact_request(benchmark_id, submission_id, artifact_id)
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
        """Reserve, directly upload, and verify an in-memory artifact."""
        reservation = self.reserve_artifact(
            benchmark_id,
            submission_id,
            path=path,
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=sha256_bytes(content),
            role=role,
        )
        self.upload_reserved_artifact(
            reservation, content, mime_type=mime_type, filename=Path(path).name
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
        """Reserve, directly upload, and verify a local output file."""
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        resolved_mime = (
            mime_type or mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        )
        return self.upload_artifact_bytes(
            benchmark_id,
            submission_id,
            read_artifact(source_path),
            path=path or source_path.name,
            mime_type=resolved_mime,
            role=role,
        )

    def resolve_traces(
        self,
        benchmark_id: str,
        submission_id: str,
        trace_ids: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> TraceResolutionList:
        """Resolve raw OTEL IDs to database UUIDs accepted with predictions."""
        return self._transport.request(
            replace(resolve_traces_request(benchmark_id, submission_id, trace_ids), timeout=timeout)
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
        """Poll trace ingestion until every raw OTEL ID resolves."""
        _validate_wait(max_wait, poll_interval)
        normalized = normalize_trace_ids(trace_ids)
        deadline = time.monotonic() + max_wait
        by_id: dict[str, str | None] = {}
        while True:
            try:
                resolution = bounded_call(
                    lambda: self.resolve_traces(
                        benchmark_id, submission_id, normalized, timeout=remaining(deadline)
                    ),
                    deadline,
                )
            except (TimeoutError, httpx.TimeoutException) as error:
                raise UnresolvedTraceError(
                    [trace_id for trace_id in normalized if not by_id.get(trace_id)],
                    resolved={key: value for key, value in by_id.items() if value is not None},
                ) from error
            by_id = {item["trace_id"]: item["trace_db_id"] for item in resolution["data"]}
            unresolved = [trace_id for trace_id in normalized if not by_id.get(trace_id)]
            if not unresolved:
                return [cast(str, by_id[trace_id]) for trace_id in normalized]
            if time.monotonic() >= deadline:
                raise UnresolvedTraceError(
                    unresolved,
                    resolved={
                        trace_id: trace_db_id
                        for trace_id, trace_db_id in by_id.items()
                        if trace_db_id is not None
                    },
                )
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    def upload_predictions(
        self,
        benchmark_id: str,
        submission_id: str,
        predictions: Sequence[ExternalPrediction],
    ) -> StagedPredictionBatch:
        """Idempotently upload canonical predictions with bounded transient retries."""
        request = upload_prediction_batch_request(benchmark_id, submission_id, predictions)
        for attempt in range(_PREDICTION_UPLOAD_MAX_ATTEMPTS):
            try:
                return self._transport.request(request)
            except (AgentGymAPIError, httpx.TransportError) as error:
                if (
                    attempt + 1 >= _PREDICTION_UPLOAD_MAX_ATTEMPTS
                    or not _retryable_prediction_upload(error)
                ):
                    raise
                time.sleep(0.1 * 2**attempt)
        raise RuntimeError("prediction upload retry loop exited unexpectedly")

    def submit_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        body: SubmitSubmissionRequest,
        *,
        idempotency_key: str,
    ) -> SubmittedSubmission:
        """Close prediction uploads and request scoring."""
        return self._transport.request(
            submit_submission_request(
                benchmark_id,
                submission_id,
                body,
                idempotency_key=idempotency_key,
            )
        )

    def get_submission_status(self, benchmark_id: str, submission_id: str) -> SubmissionStatus:
        """Fetch submission and linked benchmark-run state."""
        return self._transport.request(submission_status_request(benchmark_id, submission_id))

    def wait_for_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        max_wait: float = 600,
        poll_interval: float = 2,
    ) -> SubmissionStatus:
        """Poll until scoring reaches a terminal submission state."""
        _validate_wait(max_wait, poll_interval)
        deadline = time.monotonic() + max_wait
        while True:
            status = self.get_submission_status(benchmark_id, submission_id)
            if status["status"] in TERMINAL_SUBMISSION_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Submission {submission_id} did not complete within {max_wait}s "
                    f"(last status: {status['status']})"
                )
            time.sleep(poll_interval)

    def cancel_submission(self, benchmark_id: str, submission_id: str) -> CancelledSubmission:
        """Cancel a submission that has not been submitted for scoring."""
        return self._transport.request(cancel_submission_request(benchmark_id, submission_id))

    def get_run_results(self, benchmark_id: str, run_id: str) -> BenchmarkRunResults:
        """Get aggregate metrics and run evidence metadata."""
        return self._transport.request(run_results_request(benchmark_id, run_id))

    def reevaluate_run(self, benchmark_id: str, run_id: str) -> QueuedBenchmarkReevaluation:
        """Re-score an execution-current run with the active evaluator revision."""
        return self._transport.request(reevaluate_run_request(benchmark_id, run_id))

    def retry_scoring(self, benchmark_id: str, run_id: str) -> RetriedBenchmarkScoring:
        """Restore scoring delivery for an existing external run."""
        return self._transport.request(retry_scoring_request(benchmark_id, run_id))

    def list_case_results(
        self,
        benchmark_id: str,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
        sort: CaseResultSort = "score",
    ) -> BenchmarkCaseResultPage:
        """Get one sorted, cursor-paginated page of case results."""
        return self._transport.request(
            case_results_request(benchmark_id, run_id, cursor=cursor, limit=limit, sort=sort)
        )

    def iter_case_results(
        self,
        benchmark_id: str,
        run_id: str,
        *,
        page_size: int = 100,
        sort: CaseResultSort = "score",
    ) -> Iterator[BenchmarkCaseResult]:
        """Iterate every case result in the selected server sort order."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = self.list_case_results(
                benchmark_id, run_id, cursor=cursor, limit=page_size, sort=sort
            )
            yield from page["data"]
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen:
                raise RuntimeError("case-result pagination returned a repeated cursor")
            seen.add(cursor)

    def get_case_result(
        self, benchmark_id: str, run_id: str, case_id: int | str
    ) -> BenchmarkCaseResult:
        """Get one complete case result by integer case ID or prediction UUID."""
        return self._transport.request(case_result_request(benchmark_id, run_id, case_id))

    def compare_runs(
        self,
        benchmark_id: str,
        *,
        parent_run_id: str,
        candidate_run_id: str,
    ) -> BenchmarkRunComparison:
        """Compare two compatible single-architecture runs case by case."""
        return self._transport.request(
            compare_runs_request(
                benchmark_id,
                parent_run_id=parent_run_id,
                candidate_run_id=candidate_run_id,
            )
        )

    def download_prediction_artifact(
        self,
        artifact: PredictionArtifact,
        destination: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
        max_bytes: int = MAX_ARTIFACT_BYTES,
        overwrite: bool = False,
    ) -> None:
        """Download an authenticated result artifact with size/hash safety.

        The current platform advertises exact size but not SHA-256 in result
        descriptors. Pass ``expected_sha256`` when an out-of-band digest is
        available; future descriptors containing ``sha256`` are verified
        automatically.
        """
        checked = cast(PredictionArtifact, dict(artifact))
        if expected_sha256 is not None:
            checked["sha256"] = validate_sha256(expected_sha256)
        spec = prediction_artifact_request(checked)
        download_prediction_artifact_sync(
            self._transport,
            checked,
            spec.path,
            Path(destination),
            max_bytes=max_bytes,
            overwrite=overwrite,
        )

    def run_and_submit(
        self,
        benchmark_id: str,
        executor: Candidate[CaseInputT],
        *,
        name: str,
        version: str,
        architecture_description: str,
        repository_url: str | None = None,
        commit_hash: str | None = None,
        revision_id: str | None = None,
        variant_identity: VariantIdentity | None = None,
        metadata: dict[str, Any] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        idempotency_key: str | None = None,
        capture_exceptions: bool = True,
        wait: bool = True,
        max_wait: float = 600,
        poll_interval: float = 2,
        trace_max_wait: float = 30,
        trace_poll_interval: float = 0.5,
        trace_cases: bool = False,
        trace_policy: TracePolicy = "best_effort",
        input_model: type[CaseInputT] | None = None,
    ) -> AgentGymRunResult:
        """Execute trusted Python code, inferring its typed case input when annotated."""
        from promptic_sdk.agent_gym.runner import submit_benchmark

        return submit_benchmark(
            self,
            benchmark_id,
            executor,
            name=name,
            version=version,
            architecture_description=architecture_description,
            repository_url=repository_url,
            commit_hash=commit_hash,
            revision_id=revision_id,
            variant_identity=variant_identity,
            metadata=metadata,
            workdir=Path(workdir) if workdir is not None else None,
            idempotency_key=idempotency_key,
            capture_exceptions=capture_exceptions,
            wait=wait,
            max_wait=max_wait,
            poll_interval=poll_interval,
            trace_max_wait=trace_max_wait,
            trace_poll_interval=trace_poll_interval,
            trace_cases=trace_cases,
            trace_policy=trace_policy,
            input_model=input_model,
        )

    def close(self) -> None:
        """Close API and direct-transfer clients."""
        self._transport.close()

    def __enter__(self) -> AgentGymClient:
        """Enter a synchronous context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the client on context exit."""
        self.close()


class AsyncAgentGymClient:
    """Asynchronous Agent Gym client sharing all request and validation logic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        access_token: str | None = None,
        ai_application_id: str | None = None,
        workspace_id: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the async client from normal Promptic credentials."""
        if ai_application_id and workspace_id and ai_application_id != workspace_id:
            raise ValueError("ai_application_id and workspace_id cannot disagree")
        config = resolve_client_config(
            api_key=api_key,
            access_token=access_token,
            ai_application_id=ai_application_id or workspace_id,
            endpoint=endpoint,
        )
        self.endpoint = config.endpoint
        self._transport = AsyncTransport(config, timeout)
        self._client = self._transport.api
        self._direct_client = self._transport.direct
        self.benchmarks = AsyncAgentGymBenchmarks(
            self,
            ai_application_id
            or workspace_id
            or os.environ.get("PROMPTIC_AI_APPLICATION_ID")
            or os.environ.get("PROMPTIC_WORKSPACE_ID"),
        )

    async def create_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        variant_identity: VariantIdentity,
        revision_id: str | None = None,
        ttl_seconds: int = 86_400,
    ) -> ExternalSubmissionCreated:
        """Create or replay an external submission session."""
        return await self._transport.request(
            create_submission_request(
                benchmark_id,
                idempotency_key=idempotency_key,
                variant_identity=variant_identity,
                revision_id=revision_id,
                ttl_seconds=ttl_seconds,
            )
        )

    async def start_submission(
        self,
        benchmark_id: str,
        *,
        idempotency_key: str,
        variant_identity: VariantIdentity,
        revision_id: str | None = None,
        ttl_seconds: int = 86_400,
    ) -> AsyncExternalSubmissionSession:
        """Create a submission and return a bound async session helper."""
        created = await self.create_submission(
            benchmark_id,
            idempotency_key=idempotency_key,
            variant_identity=variant_identity,
            revision_id=revision_id,
            ttl_seconds=ttl_seconds,
        )
        return AsyncExternalSubmissionSession(
            self,
            benchmark_id,
            created["submission_id"],
            created["revision"]["id"],
            created,
        )

    def resume_submission(
        self, benchmark_id: str, submission_id: str
    ) -> AsyncExternalSubmissionSession:
        """Bind an async session helper to an existing submission."""
        require_uuid(benchmark_id, "benchmark_id")
        require_uuid(submission_id, "submission_id")
        return AsyncExternalSubmissionSession(self, benchmark_id, submission_id)

    async def get_manifest_page(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ManifestPage:
        """Fetch one immutable manifest page."""
        return await self._transport.request(
            manifest_page_request(benchmark_id, submission_id, cursor=cursor, limit=limit)
        )

    async def iter_manifest_cases(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> AsyncIterator[ManifestCase]:
        """Iterate every immutable manifest case."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = await self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            for case in page["data"]:
                yield case
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen.add(cursor)

    async def get_manifest(
        self, benchmark_id: str, submission_id: str, *, page_size: int = 100
    ) -> ExternalSubmissionManifest:
        """Collect and validate every immutable manifest page."""
        first = await self.get_manifest_page(benchmark_id, submission_id, limit=page_size)
        cursor = first["next_cursor"]
        seen: set[str] = set()
        pages: list[ManifestPage] = []
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("manifest pagination returned a repeated cursor")
            seen.add(cursor)
            page = await self.get_manifest_page(
                benchmark_id, submission_id, cursor=cursor, limit=page_size
            )
            pages.append(page)
            cursor = page["next_cursor"]
        return collect_manifest(first, pages)

    async def materialize_manifest(
        self,
        benchmark_id: str,
        submission_id: str,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Download manifest inputs beneath a safe local root."""
        manifest = await self.get_manifest(benchmark_id, submission_id, page_size=page_size)
        root = Path(destination)
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = materialized_path(root, case, input_file)
                await download_manifest_input_async(
                    self._transport, input_file, local_path, overwrite=overwrite
                )
                files.append(
                    MaterializedInputFile(
                        case["dataset_case_id"],
                        input_file["artifact_id"],
                        input_file["field_path"],
                        input_file["path"],
                        local_path,
                        input_file["mime_type"],
                        input_file["size_bytes"],
                        input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        await asyncio.to_thread(
            atomic_write_bytes,
            manifest_path,
            sanitized_manifest_json(manifest, files, root),
        )
        return MaterializedManifest(root, manifest_path, tuple(files), manifest)

    async def download_dataset(
        self,
        benchmark_id: str,
        destination: str | os.PathLike[str],
        *,
        revision_id: str | None = None,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> DownloadedBenchmarkDataset:
        """Download public frozen inputs without creating a submission or run."""
        first = cast(
            RevisionManifestPage,
            await self._transport.request(
                revision_manifest_page_request(
                    benchmark_id, revision_id, cursor=None, limit=page_size
                )
            ),
        )
        pages: list[RevisionManifestPage] = []
        cursor = first["next_cursor"]
        seen_cursors: set[str] = set()
        while cursor is not None:
            if cursor in seen_cursors:
                raise RuntimeError("revision manifest pagination repeated a cursor")
            seen_cursors.add(cursor)
            page = cast(
                RevisionManifestPage,
                await self._transport.request(
                    revision_manifest_page_request(
                        benchmark_id, revision_id, cursor=cursor, limit=page_size
                    )
                ),
            )
            pages.append(page)
            cursor = page["next_cursor"]
        manifest = collect_revision_manifest(first, pages)
        root = Path(destination)
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        files: list[MaterializedInputFile] = []
        for case in manifest["data"]:
            for input_file in case["input_files"]:
                local_path = materialized_path(root, case, input_file)
                await download_manifest_input_async(
                    self._transport, input_file, local_path, overwrite=overwrite
                )
                files.append(
                    MaterializedInputFile(
                        case["dataset_case_id"],
                        input_file["artifact_id"],
                        input_file["field_path"],
                        input_file["path"],
                        local_path,
                        input_file["mime_type"],
                        input_file["size_bytes"],
                        input_file["sha256"],
                    )
                )
        manifest_path = root / "manifest.json"
        await asyncio.to_thread(
            atomic_write_bytes,
            manifest_path,
            sanitized_manifest_json(manifest, files, root),
        )
        return downloaded_dataset(MaterializedManifest(root, manifest_path, tuple(files), manifest))

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
        """Reserve a submission-owned output artifact."""
        return await self._transport.request(
            reserve_artifact_request(
                benchmark_id,
                submission_id,
                path=path,
                mime_type=mime_type,
                size_bytes=size_bytes,
                sha256=sha256,
                role=role,
            )
        )

    async def upload_reserved_artifact(
        self,
        reservation: ReservedSubmissionArtifact,
        content: bytes,
        *,
        mime_type: str,
        filename: str | None = None,
    ) -> None:
        """Upload bytes to a reservation's credential-free storage target."""
        await upload_reserved_async(
            self._transport,
            reservation,
            content,
            mime_type=mime_type,
            filename=filename or Path(reservation["path"]).name,
        )

    async def complete_artifact(
        self, benchmark_id: str, submission_id: str, artifact_id: str
    ) -> CompletedSubmissionArtifact:
        """Ask the platform to verify an uploaded artifact."""
        return await self._transport.request(
            complete_artifact_request(benchmark_id, submission_id, artifact_id)
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
        """Reserve, upload, and verify an in-memory artifact."""
        reservation = await self.reserve_artifact(
            benchmark_id,
            submission_id,
            path=path,
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=sha256_bytes(content),
            role=role,
        )
        await self.upload_reserved_artifact(
            reservation, content, mime_type=mime_type, filename=Path(path).name
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
        resolved_mime = (
            mime_type or mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        )
        content = await asyncio.to_thread(read_artifact, source_path)
        return await self.upload_artifact_bytes(
            benchmark_id,
            submission_id,
            content,
            path=path or source_path.name,
            mime_type=resolved_mime,
            role=role,
        )

    async def resolve_traces(
        self,
        benchmark_id: str,
        submission_id: str,
        trace_ids: Sequence[str],
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - HTTP phase timeout; caller owns deadline
    ) -> TraceResolutionList:
        """Resolve raw OTEL IDs to database UUIDs accepted with predictions."""
        return await self._transport.request(
            replace(resolve_traces_request(benchmark_id, submission_id, trace_ids), timeout=timeout)
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
        """Poll trace ingestion until every raw OTEL ID resolves."""
        _validate_wait(max_wait, poll_interval)
        normalized = normalize_trace_ids(trace_ids)
        deadline = time.monotonic() + max_wait
        by_id: dict[str, str | None] = {}
        while True:
            try:
                budget = remaining(deadline)
                resolution = await asyncio.wait_for(
                    self.resolve_traces(benchmark_id, submission_id, normalized, timeout=budget),
                    timeout=budget,
                )
            except (TimeoutError, httpx.TimeoutException) as error:
                raise UnresolvedTraceError(
                    [trace_id for trace_id in normalized if not by_id.get(trace_id)],
                    resolved={key: value for key, value in by_id.items() if value is not None},
                ) from error
            by_id = {item["trace_id"]: item["trace_db_id"] for item in resolution["data"]}
            unresolved = [trace_id for trace_id in normalized if not by_id.get(trace_id)]
            if not unresolved:
                return [cast(str, by_id[trace_id]) for trace_id in normalized]
            if time.monotonic() >= deadline:
                raise UnresolvedTraceError(
                    unresolved,
                    resolved={
                        trace_id: trace_db_id
                        for trace_id, trace_db_id in by_id.items()
                        if trace_db_id is not None
                    },
                )
            await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    async def upload_predictions(
        self,
        benchmark_id: str,
        submission_id: str,
        predictions: Sequence[ExternalPrediction],
    ) -> StagedPredictionBatch:
        """Idempotently upload canonical predictions with bounded transient retries."""
        request = upload_prediction_batch_request(benchmark_id, submission_id, predictions)
        for attempt in range(_PREDICTION_UPLOAD_MAX_ATTEMPTS):
            try:
                return await self._transport.request(request)
            except (AgentGymAPIError, httpx.TransportError) as error:
                if (
                    attempt + 1 >= _PREDICTION_UPLOAD_MAX_ATTEMPTS
                    or not _retryable_prediction_upload(error)
                ):
                    raise
                await asyncio.sleep(0.1 * 2**attempt)
        raise RuntimeError("prediction upload retry loop exited unexpectedly")

    async def submit_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        body: SubmitSubmissionRequest,
        *,
        idempotency_key: str,
    ) -> SubmittedSubmission:
        """Close prediction uploads and request scoring."""
        return await self._transport.request(
            submit_submission_request(
                benchmark_id,
                submission_id,
                body,
                idempotency_key=idempotency_key,
            )
        )

    async def get_submission_status(
        self, benchmark_id: str, submission_id: str
    ) -> SubmissionStatus:
        """Fetch submission and linked benchmark-run state."""
        return await self._transport.request(submission_status_request(benchmark_id, submission_id))

    async def wait_for_submission(
        self,
        benchmark_id: str,
        submission_id: str,
        *,
        max_wait: float = 600,
        poll_interval: float = 2,
    ) -> SubmissionStatus:
        """Poll until scoring reaches a terminal submission state."""
        _validate_wait(max_wait, poll_interval)
        deadline = time.monotonic() + max_wait
        while True:
            status = await self.get_submission_status(benchmark_id, submission_id)
            if status["status"] in TERMINAL_SUBMISSION_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Submission {submission_id} did not complete within {max_wait}s "
                    f"(last status: {status['status']})"
                )
            await asyncio.sleep(poll_interval)

    async def cancel_submission(self, benchmark_id: str, submission_id: str) -> CancelledSubmission:
        """Cancel a submission that has not been submitted for scoring."""
        return await self._transport.request(cancel_submission_request(benchmark_id, submission_id))

    async def get_run_results(self, benchmark_id: str, run_id: str) -> BenchmarkRunResults:
        """Get aggregate metrics and run evidence metadata."""
        return await self._transport.request(run_results_request(benchmark_id, run_id))

    async def reevaluate_run(self, benchmark_id: str, run_id: str) -> QueuedBenchmarkReevaluation:
        """Re-score an execution-current run with the active evaluator revision."""
        return await self._transport.request(reevaluate_run_request(benchmark_id, run_id))

    async def retry_scoring(self, benchmark_id: str, run_id: str) -> RetriedBenchmarkScoring:
        """Restore scoring delivery for an existing external run."""
        return await self._transport.request(retry_scoring_request(benchmark_id, run_id))

    async def list_case_results(
        self,
        benchmark_id: str,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
        sort: CaseResultSort = "score",
    ) -> BenchmarkCaseResultPage:
        """Get one sorted, cursor-paginated page of case results."""
        return await self._transport.request(
            case_results_request(benchmark_id, run_id, cursor=cursor, limit=limit, sort=sort)
        )

    async def iter_case_results(
        self,
        benchmark_id: str,
        run_id: str,
        *,
        page_size: int = 100,
        sort: CaseResultSort = "score",
    ) -> AsyncIterator[BenchmarkCaseResult]:
        """Iterate every case result in the selected server sort order."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = await self.list_case_results(
                benchmark_id, run_id, cursor=cursor, limit=page_size, sort=sort
            )
            for result in page["data"]:
                yield result
            cursor = page["next_cursor"]
            if cursor is None:
                return
            if cursor in seen:
                raise RuntimeError("case-result pagination returned a repeated cursor")
            seen.add(cursor)

    async def get_case_result(
        self, benchmark_id: str, run_id: str, case_id: int | str
    ) -> BenchmarkCaseResult:
        """Get one complete case result by integer case ID or prediction UUID."""
        return await self._transport.request(case_result_request(benchmark_id, run_id, case_id))

    async def compare_runs(
        self,
        benchmark_id: str,
        *,
        parent_run_id: str,
        candidate_run_id: str,
    ) -> BenchmarkRunComparison:
        """Compare two compatible single-architecture runs case by case."""
        return await self._transport.request(
            compare_runs_request(
                benchmark_id,
                parent_run_id=parent_run_id,
                candidate_run_id=candidate_run_id,
            )
        )

    async def download_prediction_artifact(
        self,
        artifact: PredictionArtifact,
        destination: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
        max_bytes: int = MAX_ARTIFACT_BYTES,
        overwrite: bool = False,
    ) -> None:
        """Download an authenticated result artifact with size/hash safety."""
        checked = cast(PredictionArtifact, dict(artifact))
        if expected_sha256 is not None:
            checked["sha256"] = validate_sha256(expected_sha256)
        spec = prediction_artifact_request(checked)
        await download_prediction_artifact_async(
            self._transport,
            checked,
            spec.path,
            Path(destination),
            max_bytes=max_bytes,
            overwrite=overwrite,
        )

    async def run_and_submit(
        self,
        benchmark_id: str,
        executor: AsyncCandidate[CaseInputT],
        *,
        name: str,
        version: str,
        architecture_description: str,
        repository_url: str | None = None,
        commit_hash: str | None = None,
        revision_id: str | None = None,
        variant_identity: VariantIdentity | None = None,
        metadata: dict[str, Any] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        idempotency_key: str | None = None,
        capture_exceptions: bool = True,
        wait: bool = True,
        max_wait: float = 600,
        poll_interval: float = 2,
        trace_max_wait: float = 30,
        trace_poll_interval: float = 0.5,
        trace_cases: bool = False,
        trace_policy: TracePolicy = "best_effort",
        input_model: type[CaseInputT] | None = None,
    ) -> AgentGymRunResult:
        """Execute trusted code asynchronously, inferring an annotated case input type."""
        from promptic_sdk.agent_gym.runner import submit_benchmark_async

        return await submit_benchmark_async(
            self,
            benchmark_id,
            executor,
            name=name,
            version=version,
            architecture_description=architecture_description,
            repository_url=repository_url,
            commit_hash=commit_hash,
            revision_id=revision_id,
            variant_identity=variant_identity,
            metadata=metadata,
            workdir=Path(workdir) if workdir is not None else None,
            idempotency_key=idempotency_key,
            capture_exceptions=capture_exceptions,
            wait=wait,
            max_wait=max_wait,
            poll_interval=poll_interval,
            trace_max_wait=trace_max_wait,
            trace_poll_interval=trace_poll_interval,
            trace_cases=trace_cases,
            trace_policy=trace_policy,
            input_model=input_model,
        )

    async def close(self) -> None:
        """Close API and direct-transfer clients."""
        await self._transport.close()

    async def __aenter__(self) -> AsyncAgentGymClient:
        """Enter an asynchronous context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client on context exit."""
        await self.close()


class ExternalSubmissionSession:
    """Submission-scoped convenience API."""

    def __init__(
        self,
        client: AgentGymClient,
        benchmark_id: str,
        submission_id: str,
        revision_id: str | None = None,
        created_response: ExternalSubmissionCreated | None = None,
    ) -> None:
        """Bind a client to one benchmark submission."""
        self.client = client
        self.benchmark_id = benchmark_id
        self.submission_id = submission_id
        self.revision_id = revision_id
        self.created_response = created_response
        self._builder = SubmissionPredictionBuilder()
        self._uploaded_predictions: dict[int, ExternalPrediction] = {}

    def __enter__(self) -> ExternalSubmissionSession:
        """Enter without implicitly submitting or cancelling the remote session."""
        return self

    def __exit__(self, *_: object) -> None:
        """Leave an unfinished session resumable until its server-side expiry."""

    def get_manifest(self, *, page_size: int = 100) -> ExternalSubmissionManifest:
        """Collect this session's manifest."""
        manifest = self.client.get_manifest(
            self.benchmark_id, self.submission_id, page_size=page_size
        )
        self._builder.set_manifest(manifest)
        return manifest

    def materialize_manifest(
        self,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Materialize this session's manifest inputs."""
        materialized = self.client.materialize_manifest(
            self.benchmark_id,
            self.submission_id,
            destination,
            page_size=page_size,
            overwrite=overwrite,
        )
        self._builder.set_manifest(cast(ExternalSubmissionManifest, materialized.manifest))
        return materialized

    def upload_artifact_file(
        self, source: str | os.PathLike[str], **kwargs: Any
    ) -> SubmissionArtifact:
        """Upload and verify one output artifact."""
        return self.client.upload_artifact_file(
            self.benchmark_id, self.submission_id, source, **kwargs
        )

    def resolve_traces(self, trace_ids: Sequence[str]) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs for this session."""
        return self.client.resolve_traces(self.benchmark_id, self.submission_id, trace_ids)

    def wait_for_resolved_traces(self, trace_ids: Sequence[str], **kwargs: Any) -> list[str]:
        """Wait for raw OTEL trace IDs to resolve."""
        return self.client.wait_for_resolved_traces(
            self.benchmark_id, self.submission_id, trace_ids, **kwargs
        )

    def _upload_prediction_values(
        self, predictions: Sequence[ExternalPrediction]
    ) -> list[StagedPredictionBatch]:
        pending = [
            prediction
            for prediction in predictions
            if self._uploaded_predictions.get(prediction["dataset_case_id"]) != prediction
        ]
        results = []
        for batch in prediction_upload_batches(pending):
            results.append(
                self.client.upload_predictions(self.benchmark_id, self.submission_id, batch)
            )
            self._uploaded_predictions.update(
                (prediction["dataset_case_id"], cast(ExternalPrediction, dict(prediction)))
                for prediction in batch
            )
        return results

    def flush_predictions(self) -> list[StagedPredictionBatch]:
        """Upload locally staged predictions without requesting scoring."""
        return self._upload_prediction_values(list(self._builder.predictions.values()))

    def _resolve_staged_traces(
        self,
        trace_policy: TracePolicy,
        *,
        max_wait: float,
        poll_interval: float,
    ) -> dict[str, str]:
        """Resolve staged raw IDs once, without blocking best-effort submission."""
        policy = normalize_trace_policy(trace_policy)
        self._builder.validate_trace_coverage(policy)
        raw_trace_ids = self._builder.pending_trace_ids()
        if policy == "disabled" or not raw_trace_ids:
            return {}
        _validate_wait(max_wait, poll_interval)

        from promptic_sdk.agent_gym.runner import _flush_traces

        deadline = time.monotonic() + max_wait
        try:
            bounded_call(lambda: _flush_traces(max(1, int(remaining(deadline) * 1000))), deadline)
        except Exception as error:
            failure = UnresolvedTraceError(
                raw_trace_ids,
                code="trace_flush_timeout"
                if isinstance(error, TimeoutError)
                else "trace_export_failed",
                phase="flush",
            )
            if policy == "required":
                raise failure from error
            _warn_omitted_traces(raw_trace_ids, failure)
            return {}
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        last_error: Exception | None = None
        for chunk in _trace_chunks(raw_trace_ids):
            try:
                database_ids = self.wait_for_resolved_traces(
                    chunk, max_wait=remaining(deadline), poll_interval=poll_interval
                )
                resolved.update(zip(chunk, database_ids, strict=True))
            except UnresolvedTraceError as error:
                if policy == "required":
                    raise
                resolved.update(error.resolved)
                unresolved.extend(error.trace_ids)
                last_error = error
            except Exception as error:
                if policy == "required":
                    if isinstance(error, TimeoutError):
                        raise UnresolvedTraceError(chunk, resolved=resolved) from error
                    raise
                unresolved.extend(chunk)
                last_error = error
        _warn_omitted_traces(list(dict.fromkeys(unresolved)), last_error)
        return resolved

    def add_prediction(
        self,
        case_id: int,
        result: AgentGymCaseResult,
    ) -> ExternalPrediction:
        """Upload artifacts and stage a prediction without waiting for trace ingestion."""
        from promptic_sdk.agent_gym.runner import prediction_from_result

        if self._builder.manifest is None:
            self._builder.set_manifest(self.get_manifest())
        self._builder.validate_case(case_id)
        raw_trace_ids = normalize_trace_ids(result.raw_trace_ids) if result.raw_trace_ids else []

        artifacts = [
            self.upload_artifact_file(
                artifact.source,
                path=artifact.path,
                mime_type=artifact.mime_type,
                role=artifact.role,
            )
            for artifact in result.output_artifacts
        ]
        artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
        prediction = prediction_from_result(
            case_id,
            result,
            artifact_ids,
            [
                (spec.field_path, artifact["artifact_id"])
                for spec, artifact in zip(result.output_artifacts, artifacts, strict=True)
            ],
        )
        staged = self._builder.stage(
            case_id,
            prediction,
            raw_trace_ids=raw_trace_ids,
        )
        if (
            len(self._builder.predictions) - len(self._uploaded_predictions)
            >= _PREDICTION_AUTO_FLUSH_SIZE
        ):
            self.flush_predictions()
        return staged

    def submit(
        self,
        body: SubmitSubmissionRequest | None = None,
        *,
        identity: VariantIdentity | None = None,
        idempotency_key: str,
        metadata: dict[str, Any] | None = None,
        trace_policy: TracePolicy = "best_effort",
        trace_max_wait: float = 30,
        trace_poll_interval: float = 0.5,
    ) -> SubmittedSubmission:
        """Upload predictions, freeze the complete run, and request scoring."""
        if body is not None and identity is not None:
            raise ValueError("pass either body or identity, not both")
        if body is None:
            if self._builder.manifest is None:
                self._builder.set_manifest(self.get_manifest())
            self._builder.validate_coverage()
            policy = normalize_trace_policy(trace_policy)
            resolved = self._resolve_staged_traces(
                policy,
                max_wait=trace_max_wait,
                poll_interval=trace_poll_interval,
            )
            predictions, body = self._builder.submission_payloads(
                identity,
                metadata,
                trace_policy=policy,
                resolved_trace_ids=resolved,
            )
            self._upload_prediction_values(predictions)
        return self.client.submit_submission(
            self.benchmark_id,
            self.submission_id,
            body,
            idempotency_key=idempotency_key,
        )

    def status(self) -> SubmissionStatus:
        """Fetch this session's current status."""
        return self.client.get_submission_status(self.benchmark_id, self.submission_id)

    def wait(self, **kwargs: Any) -> SubmissionStatus:
        """Wait for this session to reach a terminal state."""
        return self.client.wait_for_submission(self.benchmark_id, self.submission_id, **kwargs)

    def cancel(self) -> CancelledSubmission:
        """Cancel this session before requesting scoring."""
        return self.client.cancel_submission(self.benchmark_id, self.submission_id)

    def retry_scoring(self) -> RetriedBenchmarkScoring:
        """Restore scoring delivery for this submission's existing run."""
        run = self.status()["run"]
        if run is None:
            raise RuntimeError("Submission has not been submitted for scoring")
        return self.client.retry_scoring(self.benchmark_id, run["id"])


class AsyncExternalSubmissionSession:
    """Asynchronous submission-scoped convenience API."""

    def __init__(
        self,
        client: AsyncAgentGymClient,
        benchmark_id: str,
        submission_id: str,
        revision_id: str | None = None,
        created_response: ExternalSubmissionCreated | None = None,
    ) -> None:
        """Bind an async client to one benchmark submission."""
        self.client = client
        self.benchmark_id = benchmark_id
        self.submission_id = submission_id
        self.revision_id = revision_id
        self.created_response = created_response
        self._builder = SubmissionPredictionBuilder()
        self._uploaded_predictions: dict[int, ExternalPrediction] = {}

    async def __aenter__(self) -> AsyncExternalSubmissionSession:
        """Enter without implicitly submitting or cancelling the remote session."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Leave an unfinished session resumable until its server-side expiry."""

    async def get_manifest(self, *, page_size: int = 100) -> ExternalSubmissionManifest:
        """Collect this session's manifest."""
        manifest = await self.client.get_manifest(
            self.benchmark_id, self.submission_id, page_size=page_size
        )
        self._builder.set_manifest(manifest)
        return manifest

    async def materialize_manifest(
        self,
        destination: str | os.PathLike[str],
        *,
        page_size: int = 100,
        overwrite: bool = False,
    ) -> MaterializedManifest:
        """Materialize this session's manifest inputs."""
        materialized = await self.client.materialize_manifest(
            self.benchmark_id,
            self.submission_id,
            destination,
            page_size=page_size,
            overwrite=overwrite,
        )
        self._builder.set_manifest(cast(ExternalSubmissionManifest, materialized.manifest))
        return materialized

    async def upload_artifact_file(
        self, source: str | os.PathLike[str], **kwargs: Any
    ) -> SubmissionArtifact:
        """Upload and verify one output artifact."""
        return await self.client.upload_artifact_file(
            self.benchmark_id, self.submission_id, source, **kwargs
        )

    async def resolve_traces(self, trace_ids: Sequence[str]) -> TraceResolutionList:
        """Resolve raw OTEL trace IDs for this session."""
        return await self.client.resolve_traces(self.benchmark_id, self.submission_id, trace_ids)

    async def wait_for_resolved_traces(self, trace_ids: Sequence[str], **kwargs: Any) -> list[str]:
        """Wait for raw OTEL trace IDs to resolve."""
        return await self.client.wait_for_resolved_traces(
            self.benchmark_id, self.submission_id, trace_ids, **kwargs
        )

    async def _upload_prediction_values(
        self, predictions: Sequence[ExternalPrediction]
    ) -> list[StagedPredictionBatch]:
        pending = [
            prediction
            for prediction in predictions
            if self._uploaded_predictions.get(prediction["dataset_case_id"]) != prediction
        ]
        results = []
        for batch in prediction_upload_batches(pending):
            results.append(
                await self.client.upload_predictions(self.benchmark_id, self.submission_id, batch)
            )
            self._uploaded_predictions.update(
                (prediction["dataset_case_id"], cast(ExternalPrediction, dict(prediction)))
                for prediction in batch
            )
        return results

    async def flush_predictions(self) -> list[StagedPredictionBatch]:
        """Upload locally staged predictions without requesting scoring."""
        return await self._upload_prediction_values(list(self._builder.predictions.values()))

    async def _resolve_staged_traces(
        self,
        trace_policy: TracePolicy,
        *,
        max_wait: float,
        poll_interval: float,
    ) -> dict[str, str]:
        """Resolve staged raw IDs once, without blocking best-effort submission."""
        policy = normalize_trace_policy(trace_policy)
        self._builder.validate_trace_coverage(policy)
        raw_trace_ids = self._builder.pending_trace_ids()
        if policy == "disabled" or not raw_trace_ids:
            return {}
        _validate_wait(max_wait, poll_interval)

        from promptic_sdk.agent_gym.runner import _flush_traces

        deadline = time.monotonic() + max_wait
        try:
            await bounded_call_async(
                lambda: _flush_traces(max(1, int(remaining(deadline) * 1000))), deadline
            )
        except Exception as error:
            failure = UnresolvedTraceError(
                raw_trace_ids,
                code="trace_flush_timeout"
                if isinstance(error, TimeoutError)
                else "trace_export_failed",
                phase="flush",
            )
            if policy == "required":
                raise failure from error
            _warn_omitted_traces(raw_trace_ids, failure)
            return {}
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        last_error: Exception | None = None
        for chunk in _trace_chunks(raw_trace_ids):
            try:
                budget = remaining(deadline)
                database_ids = await self.wait_for_resolved_traces(
                    chunk, max_wait=budget, poll_interval=poll_interval
                )
                resolved.update(zip(chunk, database_ids, strict=True))
            except UnresolvedTraceError as error:
                if policy == "required":
                    raise
                resolved.update(error.resolved)
                unresolved.extend(error.trace_ids)
                last_error = error
            except Exception as error:
                if policy == "required":
                    if isinstance(error, TimeoutError):
                        raise UnresolvedTraceError(chunk, resolved=resolved) from error
                    raise
                unresolved.extend(chunk)
                last_error = error
        _warn_omitted_traces(list(dict.fromkeys(unresolved)), last_error)
        return resolved

    async def add_prediction(
        self,
        case_id: int,
        result: AgentGymCaseResult,
    ) -> ExternalPrediction:
        """Upload artifacts and stage a prediction without waiting for trace ingestion."""
        from promptic_sdk.agent_gym.runner import prediction_from_result

        if self._builder.manifest is None:
            self._builder.set_manifest(await self.get_manifest())
        self._builder.validate_case(case_id)
        raw_trace_ids = normalize_trace_ids(result.raw_trace_ids) if result.raw_trace_ids else []

        artifacts = []
        for artifact in result.output_artifacts:
            artifacts.append(
                await self.upload_artifact_file(
                    artifact.source,
                    path=artifact.path,
                    mime_type=artifact.mime_type,
                    role=artifact.role,
                )
            )
        artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
        prediction = prediction_from_result(
            case_id,
            result,
            artifact_ids,
            [
                (spec.field_path, artifact["artifact_id"])
                for spec, artifact in zip(result.output_artifacts, artifacts, strict=True)
            ],
        )
        staged = self._builder.stage(
            case_id,
            prediction,
            raw_trace_ids=raw_trace_ids,
        )
        if (
            len(self._builder.predictions) - len(self._uploaded_predictions)
            >= _PREDICTION_AUTO_FLUSH_SIZE
        ):
            await self.flush_predictions()
        return staged

    async def submit(
        self,
        body: SubmitSubmissionRequest | None = None,
        *,
        identity: VariantIdentity | None = None,
        idempotency_key: str,
        metadata: dict[str, Any] | None = None,
        trace_policy: TracePolicy = "best_effort",
        trace_max_wait: float = 30,
        trace_poll_interval: float = 0.5,
    ) -> SubmittedSubmission:
        """Upload predictions, freeze the complete run, and request scoring."""
        if body is not None and identity is not None:
            raise ValueError("pass either body or identity, not both")
        if body is None:
            if self._builder.manifest is None:
                self._builder.set_manifest(await self.get_manifest())
            self._builder.validate_coverage()
            policy = normalize_trace_policy(trace_policy)
            resolved = await self._resolve_staged_traces(
                policy,
                max_wait=trace_max_wait,
                poll_interval=trace_poll_interval,
            )
            predictions, body = self._builder.submission_payloads(
                identity,
                metadata,
                trace_policy=policy,
                resolved_trace_ids=resolved,
            )
            await self._upload_prediction_values(predictions)
        return await self.client.submit_submission(
            self.benchmark_id,
            self.submission_id,
            body,
            idempotency_key=idempotency_key,
        )

    async def status(self) -> SubmissionStatus:
        """Fetch this session's current status."""
        return await self.client.get_submission_status(self.benchmark_id, self.submission_id)

    async def wait(self, **kwargs: Any) -> SubmissionStatus:
        """Wait for this session to reach a terminal state."""
        return await self.client.wait_for_submission(
            self.benchmark_id, self.submission_id, **kwargs
        )

    async def cancel(self) -> CancelledSubmission:
        """Cancel this session before requesting scoring."""
        return await self.client.cancel_submission(self.benchmark_id, self.submission_id)

    async def retry_scoring(self) -> RetriedBenchmarkScoring:
        """Restore scoring delivery for this submission's existing run."""
        run = (await self.status())["run"]
        if run is None:
            raise RuntimeError("Submission has not been submitted for scoring")
        return await self.client.retry_scoring(self.benchmark_id, run["id"])


__all__ = [
    "AgentGymAPIError",
    "AgentGymClient",
    "ArtifactIntegrityError",
    "ArtifactTransferError",
    "AsyncAgentGymClient",
    "AsyncExternalSubmissionSession",
    "ExternalSubmissionSession",
    "MaterializedInputFile",
    "MaterializedManifest",
    "UnresolvedTraceError",
]
