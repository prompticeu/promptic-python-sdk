"""Sync and async Agent Gym benchmark authoring APIs."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import mimetypes
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from promptic_sdk.agent_gym._http import RequestSpec
from promptic_sdk.agent_gym.artifacts import upload_direct_async, upload_direct_sync
from promptic_sdk.agent_gym.authoring_models import (
    AgentEvaluator,
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkDraftStatus,
    BenchmarkFile,
    BenchmarkRevisionDraft,
    BenchmarkRevisionDraftResponse,
    BenchmarkRevisionList,
    BenchmarkSchemaConflict,
    BulkCaseProgress,
    BulkCaseResponse,
    CreatedBenchmarkCase,
    PublishedBenchmarkRevision,
    StagedBenchmarkUpdate,
)
from promptic_sdk.agent_gym.models import PresignedUpload
from promptic_sdk.agent_gym.submissions import require_uuid, validate_artifact_path

if TYPE_CHECKING:
    from promptic_sdk.agent_gym.client import AgentGymClient, AsyncAgentGymClient

_MAX_CASE_FILE_BYTES = 25_000_000
_MAX_CASE_FILES = 10
_MAX_CASE_TOTAL_BYTES = 100_000_000
_DEFAULT_BULK_CASE_BATCH_SIZE = 50
_MISSING = object()
_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "text/plain",
    "text/csv",
    "text/markdown",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
FileUploadCache = dict[tuple[str, str, int], dict[str, Any]]


def _workspace_id(explicit: str | None, configured: str | None) -> str:
    value = explicit or configured
    if value is None:
        raise ValueError(
            "workspace_id is required for benchmark authoring; pass it to AgentGymClient "
            "or benchmarks.create/list"
        )
    return require_uuid(value, "workspace_id")


def _create_body(
    *,
    workspace_id: str,
    name: str,
    goal: str,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
    evaluators: Sequence[AgentEvaluator],
    description: str | None,
) -> dict[str, Any]:
    normalized_name = name.strip()
    normalized_goal = goal.strip()
    if not normalized_name:
        raise ValueError("benchmark name cannot be empty")
    if not normalized_goal:
        raise ValueError("agent goal cannot be empty")
    if input_schema is not None and input_schema.get("type") != "object":
        raise ValueError("input_schema must be a JSON object schema")
    if output_schema is not None and output_schema.get("type") != "object":
        raise ValueError("output_schema must be a JSON object schema")
    body: dict[str, Any] = {
        "workspaceId": require_uuid(workspace_id, "workspace_id"),
        "name": normalized_name,
        "taskDescription": normalized_goal,
    }
    body["evaluators"] = [evaluator.as_request() for evaluator in evaluators]
    if input_schema is not None:
        body["inputSchema"] = dict(input_schema)
    if output_schema is not None:
        body["targetSchema"] = dict(output_schema)
    if description is not None:
        body["componentDescription"] = description.strip() or None
    return body


def _read_case_file(value: BenchmarkFile) -> tuple[bytes, str, str, str]:
    source = value.source
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if size > _MAX_CASE_FILE_BYTES:
        raise ValueError(f"benchmark case files cannot exceed {_MAX_CASE_FILE_BYTES} bytes")
    logical_path = validate_artifact_path(value.path or source.name)
    mime_type = value.mime_type or mimetypes.guess_type(logical_path)[0]
    if mime_type not in _ALLOWED_MIME_TYPES:
        raise ValueError(f"unsupported benchmark case file MIME type: {mime_type}")
    content = source.read_bytes()
    return content, logical_path, mime_type, hashlib.sha256(content).hexdigest()


def _presign_request(filename: str, mime_type: str, size: int) -> RequestSpec:
    return RequestSpec(
        "POST",
        "/storage-objects/presign",
        json={
            "folder": "artifacts",
            "filename": filename,
            "contentType": mime_type,
            "access": "private",
            "maxSizeBytes": _MAX_CASE_FILE_BYTES,
            "sizeBytes": size,
        },
    )


def _upload_file_sync(
    client: AgentGymClient,
    value: BenchmarkFile,
    cache: FileUploadCache | None = None,
) -> dict[str, Any]:
    content, logical_path, mime_type, sha256 = _read_case_file(value)
    cache_key = (sha256, mime_type, len(content))
    if cache is not None and cache_key in cache:
        reused = dict(cache[cache_key])
        reused["path"] = logical_path
        return reused
    response = cast(
        dict[str, Any],
        client._transport.request(
            _presign_request(  # noqa: SLF001
                value.source.name, mime_type, len(content)
            )
        ),
    )
    storage_object_id = require_uuid(response["storageObjectId"], "storageObjectId")
    upload_direct_sync(
        client._transport,  # noqa: SLF001
        cast(PresignedUpload, response),
        content,
        mime_type=mime_type,
        filename=value.source.name,
    )
    uploaded = {
        "storageObjectId": storage_object_id,
        "path": logical_path,
        "mime": mime_type,
        "size": len(content),
        "sha256": sha256,
    }
    if cache is not None:
        cache[cache_key] = uploaded
    return dict(uploaded)


async def _upload_file_async(
    client: AsyncAgentGymClient,
    value: BenchmarkFile,
    cache: FileUploadCache | None = None,
) -> dict[str, Any]:
    content, logical_path, mime_type, sha256 = await asyncio.to_thread(_read_case_file, value)
    cache_key = (sha256, mime_type, len(content))
    if cache is not None and cache_key in cache:
        reused = dict(cache[cache_key])
        reused["path"] = logical_path
        return reused
    response = cast(
        dict[str, Any],
        await client._transport.request(  # noqa: SLF001
            _presign_request(value.source.name, mime_type, len(content))
        ),
    )
    storage_object_id = require_uuid(response["storageObjectId"], "storageObjectId")
    await upload_direct_async(
        client._transport,  # noqa: SLF001
        cast(PresignedUpload, response),
        content,
        mime_type=mime_type,
        filename=value.source.name,
    )
    uploaded = {
        "storageObjectId": storage_object_id,
        "path": logical_path,
        "mime": mime_type,
        "size": len(content),
        "sha256": sha256,
    }
    if cache is not None:
        cache[cache_key] = uploaded
    return dict(uploaded)


def _validate_file_count(files: Sequence[BenchmarkFile], field_name: str) -> None:
    if len(files) > _MAX_CASE_FILES:
        raise ValueError(f"{field_name} cannot contain more than {_MAX_CASE_FILES} files")


def _validate_case_file_total(files: Sequence[BenchmarkFile]) -> None:
    total_bytes = 0
    for value in files:
        if not value.source.is_file():
            raise FileNotFoundError(value.source)
        total_bytes += value.source.stat().st_size
    if total_bytes > _MAX_CASE_TOTAL_BYTES:
        raise ValueError(
            f"benchmark case files cannot exceed {_MAX_CASE_TOTAL_BYTES} bytes in total"
        )


def _case_values(
    value: Any,
    *,
    schema: Mapping[str, Any] | None = None,
    path: tuple[str, ...] = (),
) -> tuple[Any, list[tuple[str, str, BenchmarkFile]]]:
    """Replace local files with manifest paths and retain upload bindings.

    A schema node marked ``x-promptic-type=file`` is one logical field even
    when its value is an array, so its uploads share the field path. Arrays of
    objects retain their element indexes for nested file fields.
    """
    if isinstance(value, BenchmarkFile):
        upload_path = value.path or value.source.name
        return upload_path, [(".".join(path), upload_path, value)]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        files: list[tuple[str, str, BenchmarkFile]] = []
        for key, child in value.items():
            child_schema = None
            if schema is not None and isinstance(schema.get("properties"), Mapping):
                candidate = schema["properties"].get(str(key))
                if isinstance(candidate, Mapping):
                    child_schema = candidate
            normalized, nested = _case_values(
                child,
                schema=child_schema,
                path=(*path, str(key)),
            )
            result[str(key)] = normalized
            files.extend(nested)
        return result, files
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        files = []
        direct_file_array = bool(
            schema is not None and schema.get("x-promptic-type") == "file"
        ) or all(isinstance(child, BenchmarkFile) for child in value)
        item_schema = schema.get("items") if schema is not None else None
        if not isinstance(item_schema, Mapping):
            item_schema = None
        for index, child in enumerate(value):
            normalized, nested = _case_values(
                child,
                schema=schema if direct_file_array else item_schema,
                path=path if direct_file_array else (*path, str(index)),
            )
            result.append(normalized)
            files.extend(nested)
        return result, files
    return value, []


def _case_body_values(
    *,
    title: str | None,
    input: Mapping[str, Any],
    output: Mapping[str, Any] | None,
    expected_behavior: str | None,
    input_schema: Mapping[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str, str, BenchmarkFile]]]:
    normalized_input, input_files = _case_values(input, schema=input_schema)
    normalized_output, output_files = _case_values(output, schema=output_schema)
    body: dict[str, Any] = {"input": normalized_input}
    if title is not None:
        body["title"] = title.strip()
    if output is not None:
        body["output"] = normalized_output
    if expected_behavior is not None:
        body["expectedBehavior"] = expected_behavior.strip()
    files = [
        (role, field_path, upload_path, file)
        for role, values in (("input", input_files), ("output", output_files))
        for field_path, upload_path, file in values
    ]
    case_files = [item[3] for item in files]
    _validate_file_count(case_files, "case files")
    _validate_case_file_total(case_files)
    return body, files


def _bulk_case_values(
    case: BenchmarkCase,
    *,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[tuple[str, str, str, BenchmarkFile]]]:
    """Normalize one in-memory case for the canonical bulk import contract."""
    return _case_body_values(
        title=case.title,
        input=case.input,
        output=case.output,
        expected_behavior=case.expected_behavior,
        input_schema=input_schema,
        output_schema=output_schema,
    )


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return batch_size


def _empty_bulk_result(total: int) -> BulkCaseResponse:
    return {
        "total": total,
        "processed": 0,
        "created": 0,
        "failed": 0,
        "batchesCompleted": 0,
        "draftCaseCount": None,
        "published": False,
        "results": [],
    }


class BulkCaseUploadError(RuntimeError):
    """A bulk upload stopped before every batch returned a confirmed response."""

    def __init__(self, batch_index: int, result: BulkCaseResponse, cause: Exception) -> None:
        self.batch_index = batch_index
        self.result = result
        self.cause = cause
        super().__init__(
            f"bulk case upload stopped at batch {batch_index + 1} after "
            f"{result['processed']} of {result['total']} cases were confirmed: {cause}"
        )


def _bulk_progress(
    result: BulkCaseResponse, *, batch_index: int, total_batches: int
) -> BulkCaseProgress:
    return {
        "batchIndex": batch_index,
        "totalBatches": total_batches,
        "total": result["total"],
        "processed": result["processed"],
        "created": result["created"],
        "failed": result["failed"],
        "draftCaseCount": result["draftCaseCount"],
    }


def _merge_bulk_response(
    result: BulkCaseResponse, response: Mapping[str, Any], batch_length: int
) -> None:
    result["processed"] += batch_length
    result["created"] += int(response.get("created", 0))
    result["failed"] += int(response.get("failed", 0))
    result["batchesCompleted"] += 1
    raw_results = response.get("results")
    if isinstance(raw_results, list):
        result["results"].extend(cast(list[Any], raw_results))
    draft_case_count = response.get("draftCaseCount")
    if isinstance(draft_case_count, int) and not isinstance(draft_case_count, bool):
        result["draftCaseCount"] = draft_case_count


def _bulk_request(
    benchmark_id: str,
    items: Sequence[Mapping[str, Any]],
    *,
    draft: bool = False,
) -> RequestSpec:
    """Build one bulk-case request from normalized cases and upload bindings."""
    if not items:
        raise ValueError("cases cannot be empty")
    path = f"/benchmarks/{benchmark_id}/cases/bulk{'?revision=draft' if draft else ''}"
    return RequestSpec("POST", path, json={"items": list(items)})


def _validate_bulk_file_field(
    role: str,
    field_path: str,
    *,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
) -> None:
    schema: Mapping[str, Any] | None = input_schema if role == "input" else output_schema
    node = schema
    for part in field_path.split(".") if field_path else ():
        if node is None:
            break
        if part.isdigit() and node.get("type") == "array":
            candidate = node.get("items")
        else:
            properties = node.get("properties")
            candidate = properties.get(part) if isinstance(properties, Mapping) else None
        node = candidate if isinstance(candidate, Mapping) else None
    if node is None or node.get("x-promptic-type") != "file":
        raise ValueError(
            f"bulk file field {role}.{field_path} must be declared with "
            "x-promptic-type='file' in its schema"
        )


def _prepare_bulk_items(
    cases: Sequence[BenchmarkCase],
    *,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
) -> list[tuple[dict[str, Any], list[tuple[str, str, str, BenchmarkFile]]]]:
    prepared: list[tuple[dict[str, Any], list[tuple[str, str, str, BenchmarkFile]]]] = []
    upload_paths: dict[str, tuple[str, str]] = {}
    for case in cases:
        body, case_uploads = _bulk_case_values(
            case,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        for role, field_path, upload_path, value in case_uploads:
            _validate_bulk_file_field(
                role,
                field_path,
                input_schema=input_schema,
                output_schema=output_schema,
            )
            _content, logical_path, mime_type, sha256 = _read_case_file(value)
            if logical_path != upload_path:
                raise RuntimeError("case file path normalization changed unexpectedly")
            previous = upload_paths.get(upload_path)
            if previous is not None and previous != (sha256, mime_type):
                raise ValueError(
                    f"bulk cases contain different files with the same path: {upload_path}"
                )
            upload_paths[upload_path] = (sha256, mime_type)
        prepared.append((body, case_uploads))
    return prepared


def _bulk_items_sync(
    client: AgentGymClient,
    cases: Sequence[BenchmarkCase],
    *,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
    cache: FileUploadCache,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for body, case_uploads in _prepare_bulk_items(
        cases,
        input_schema=input_schema,
        output_schema=output_schema,
    ):
        if case_uploads:
            body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = _upload_file_sync(client, value, cache)
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        items.append(body)
    return items


async def _bulk_items_async(
    client: AsyncAgentGymClient,
    cases: Sequence[BenchmarkCase],
    *,
    input_schema: Mapping[str, Any] | None,
    output_schema: Mapping[str, Any] | None,
    cache: FileUploadCache,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    prepared = await asyncio.to_thread(
        _prepare_bulk_items,
        cases,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    for body, case_uploads in prepared:
        if case_uploads:
            body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = await _upload_file_async(client, value, cache)
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        items.append(body)
    return items


class AgentGymBenchmarkCases:
    """Synchronous case authoring bound to one benchmark."""

    def __init__(self, benchmark: AgentGymBenchmark) -> None:
        """Bind case operations to an editable benchmark."""
        self._benchmark = benchmark
        self._uploaded_files: FileUploadCache = {}

    def _draft_query(self, *, create: bool) -> str:
        if not self._benchmark.data.get("activeRevisionId"):
            return ""
        draft = self._benchmark.revision_draft()
        if draft is None and create:
            draft = self._benchmark.create_revision_draft()
        return "?revision=draft" if draft is not None else ""

    def list(self) -> builtins.list[CreatedBenchmarkCase]:
        """List editable cases in deterministic draft order."""
        response = self._benchmark._manager._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "GET",
                f"/benchmarks/{self._benchmark.id}/cases{self._draft_query(create=False)}",
            )
        )
        return cast(list[CreatedBenchmarkCase], response["data"])

    def add(
        self,
        *,
        input: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        expected_behavior: str | None = None,
        title: str | None = None,
    ) -> CreatedBenchmarkCase:
        """Upload nested schema file values and add one validated test case."""
        draft_query = self._draft_query(create=True)
        body, case_uploads = _case_body_values(
            title=title,
            input=dict(input or {}),
            output=dict(output) if output is not None else None,
            expected_behavior=expected_behavior,
            input_schema=self._benchmark.data.get("inputSchema"),
            output_schema=self._benchmark.data.get("targetSchema"),
        )
        body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = _upload_file_sync(
                self._benchmark._manager._client,
                value,
                self._uploaded_files,
            )
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        return cast(
            CreatedBenchmarkCase,
            self._benchmark._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "POST",
                    f"/benchmarks/{self._benchmark.id}/cases{draft_query}",
                    json=body,
                )
            ),
        )

    def update(
        self,
        case_id: int,
        *,
        input: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        expected_behavior: str | None = None,
        title: str | None = None,
    ) -> CreatedBenchmarkCase:
        """Replace one case in the editable draft and reconcile its conflicts."""
        if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
            raise ValueError("case_id must be a positive integer")
        draft_query = self._draft_query(create=True)
        body, case_uploads = _case_body_values(
            title=title,
            input=dict(input or {}),
            output=dict(output) if output is not None else None,
            expected_behavior=expected_behavior,
            input_schema=self._benchmark.data.get("inputSchema"),
            output_schema=self._benchmark.data.get("targetSchema"),
        )
        body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = _upload_file_sync(
                self._benchmark._manager._client, value, self._uploaded_files
            )
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        separator = "&" if draft_query else "?"
        return cast(
            CreatedBenchmarkCase,
            self._benchmark._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "PATCH",
                    f"/benchmarks/{self._benchmark.id}/cases{draft_query}{separator}caseId={case_id}",
                    json=body,
                )
            ),
        )

    def add_many(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        batch_size: int = _DEFAULT_BULK_CASE_BATCH_SIZE,
        on_progress: Callable[[BulkCaseProgress], None] | None = None,
    ) -> BulkCaseResponse:
        """Add cases in confirmed batches without publishing the benchmark draft."""
        if not cases:
            raise ValueError("cases cannot be empty")
        size = _validate_batch_size(batch_size)
        result = _empty_bulk_result(len(cases))
        total_batches = (len(cases) + size - 1) // size
        draft = bool(self._draft_query(create=True))
        for batch_index, offset in enumerate(range(0, len(cases), size)):
            batch = cases[offset : offset + size]
            try:
                items = _bulk_items_sync(
                    self._benchmark._manager._client,
                    batch,
                    input_schema=self._benchmark.data.get("inputSchema"),
                    output_schema=self._benchmark.data.get("targetSchema"),
                    cache=self._uploaded_files,
                )
                response = cast(
                    Mapping[str, Any],
                    self._benchmark._manager._client._transport.request(  # noqa: SLF001
                        _bulk_request(self._benchmark.id, items, draft=draft)
                    ),
                )
            except ValueError:
                raise
            except Exception as error:
                raise BulkCaseUploadError(batch_index, result, error) from error
            _merge_bulk_response(result, response, len(batch))
            if on_progress is not None:
                on_progress(
                    _bulk_progress(
                        result,
                        batch_index=batch_index,
                        total_batches=total_batches,
                    )
                )
        return result


class AgentGymBenchmark:
    """Synchronous editable benchmark resource."""

    def __init__(self, manager: AgentGymBenchmarks, data: BenchmarkDefinition) -> None:
        """Initialize from one benchmark definition response."""
        self._manager = manager
        self.data = data
        self.staged_message: str | None = None
        self.staged_conflicts: list[BenchmarkSchemaConflict] = []
        self.cases = AgentGymBenchmarkCases(self)

    @property
    def id(self) -> str:
        """Return the benchmark UUID."""
        return self.data["id"]

    @property
    def name(self) -> str:
        """Return the benchmark name."""
        return self.data["name"]

    def refresh(self) -> AgentGymBenchmark:
        """Reload this editable definition."""
        self.data = self._manager._get_data(self.id)
        return self

    @property
    def ready_for_submission(self) -> bool:
        """Return whether schemas, cases, and evaluators are ready to publish and run."""
        return bool(self.data.get("configuration", {}).get("readyForSubmission"))

    def configure(
        self,
        *,
        name: str | None = None,
        goal: str | None = None,
        description: str | None = None,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        evaluators: Sequence[AgentEvaluator] | None = None,
    ) -> AgentGymBenchmark:
        """Configure the Agent goal, schemas, scoring, and independent evaluators."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if goal is not None:
            body["taskDescription"] = goal
        if description is not None:
            body["componentDescription"] = description
        if input_schema is not None:
            body["inputSchema"] = dict(input_schema)
        if output_schema is not None:
            body["targetSchema"] = dict(output_schema)
        if evaluators is not None:
            body["evaluators"] = [evaluator.as_request() for evaluator in evaluators]
        response = self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "PATCH",
                f"/benchmarks/{self.id}",
                json=body,
            )
        )
        if response.get("staged") is True:
            staged = cast(StagedBenchmarkUpdate, response)
            draft = staged["draft"]
            task_snapshot = draft.get("taskSnapshot")
            if task_snapshot is not None:
                schema = task_snapshot.get("inputContract", {}).get("text", {}).get("schema")
                if schema is not None:
                    self.data["inputSchema"] = cast(dict[str, Any], schema)
            if "targetSchema" in draft:
                self.data["targetSchema"] = draft["targetSchema"]
            evaluator_snapshot = draft.get("evaluatorSnapshot")
            if evaluator_snapshot is not None and "evaluatorGraph" in evaluator_snapshot:
                self.data["evaluatorGraph"] = cast(
                    dict[str, Any], evaluator_snapshot["evaluatorGraph"]
                )
            self.staged_message = staged["message"]
            self.staged_conflicts = list(staged["conflicts"])
        else:
            self.data = cast(BenchmarkDefinition, response)
            self.staged_message = None
            self.staged_conflicts = []
        return self

    def create_revision_draft(self) -> BenchmarkRevisionDraft:
        """Create or reuse the persisted draft for a published benchmark."""
        response = cast(
            BenchmarkRevisionDraftResponse,
            self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("POST", f"/benchmarks/{self.id}/revisions/draft", json={})
            ),
        )
        draft = response["draft"]
        if draft is None:
            raise RuntimeError("revision draft creation returned no draft")
        return draft

    def revisions(self) -> BenchmarkRevisionList:
        """List immutable revisions and current draft status."""
        return cast(
            BenchmarkRevisionList,
            self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{self.id}/revisions")
            ),
        )

    def draft_status(self) -> BenchmarkDraftStatus | None:
        """Return whether the editable draft differs from its active revision."""
        return self.revisions()["status"]

    def revision_draft(self) -> BenchmarkRevisionDraft | None:
        """Return the persisted schema-migration draft, if one exists."""
        response = cast(
            BenchmarkRevisionDraftResponse,
            self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{self.id}/revisions/draft")
            ),
        )
        return response["draft"]

    def resolve_revision_draft(
        self,
        field: str,
        *,
        kind: str,
        value: Any = _MISSING,
    ) -> BenchmarkRevisionDraft:
        """Resolve one schema conflict by making it optional, defaulting, or editing cases."""
        normalized_field = field.strip()
        if not normalized_field:
            raise ValueError("field cannot be empty")
        if kind not in {"make_optional", "default_value", "edit_cases"}:
            raise ValueError("kind must be make_optional, default_value, or edit_cases")
        if kind == "default_value" and value is _MISSING:
            raise ValueError("value is required for a default_value resolution")
        if kind != "default_value" and value is not _MISSING:
            raise ValueError("value is only supported for a default_value resolution")
        resolution: dict[str, Any] = {"kind": kind}
        if value is not _MISSING:
            resolution["value"] = value
        response = cast(
            BenchmarkRevisionDraftResponse,
            self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "PATCH",
                    f"/benchmarks/{self.id}/revisions/draft",
                    json={"field": normalized_field, "resolution": resolution},
                )
            ),
        )
        draft = response["draft"]
        if draft is None:
            raise RuntimeError("revision draft resolution returned no draft")
        return draft

    def abandon_revision_draft(self) -> None:
        """Discard the persisted schema-migration proposal without changing the active revision."""
        self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec("DELETE", f"/benchmarks/{self.id}/revisions/draft")
        )

    def publish(self) -> PublishedBenchmarkRevision:
        """Idempotently publish the current draft fingerprint."""
        return PublishedBenchmarkRevision.from_response(
            self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("POST", f"/benchmarks/{self.id}/revisions", json={})
            )
        )

    def delete(self) -> None:
        """Delete this editable benchmark and its dependent records."""
        self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec("DELETE", f"/benchmarks/{self.id}")
        )


class AgentGymBenchmarks:
    """Synchronous benchmark authoring namespace."""

    def __init__(self, client: AgentGymClient, workspace_id: str | None) -> None:
        """Bind authoring operations to a client and optional workspace."""
        self._client = client
        self._workspace_id = workspace_id

    def create(
        self,
        *,
        name: str,
        goal: str,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        evaluators: Sequence[AgentEvaluator] = (),
        description: str | None = None,
        workspace_id: str | None = None,
    ) -> AgentGymBenchmark:
        """Create a typed editable benchmark definition."""
        resolved_workspace = _workspace_id(workspace_id, self._workspace_id)
        response = self._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "POST",
                "/benchmarks",
                json=_create_body(
                    workspace_id=resolved_workspace,
                    name=name,
                    goal=goal,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    evaluators=evaluators,
                    description=description,
                ),
            )
        )
        return AgentGymBenchmark(self, cast(BenchmarkDefinition, response))

    def list(self, *, workspace_id: str | None = None) -> builtins.list[AgentGymBenchmark]:
        """List editable benchmarks in a workspace."""
        resolved_workspace = _workspace_id(workspace_id, self._workspace_id)
        response = self._client._transport.request(  # noqa: SLF001
            RequestSpec("GET", "/benchmarks", params={"workspaceId": resolved_workspace})
        )
        return [
            AgentGymBenchmark(self, cast(BenchmarkDefinition, item)) for item in response["data"]
        ]

    def _get_data(self, benchmark_id: str) -> BenchmarkDefinition:
        require_uuid(benchmark_id, "benchmark_id")
        return cast(
            BenchmarkDefinition,
            self._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{benchmark_id}")
            ),
        )

    def get(self, benchmark_id: str) -> AgentGymBenchmark:
        """Get one editable benchmark by UUID."""
        return AgentGymBenchmark(self, self._get_data(benchmark_id))


class AsyncAgentGymBenchmarkCases:
    """Asynchronous case authoring bound to one benchmark."""

    def __init__(self, benchmark: AsyncAgentGymBenchmark) -> None:
        """Bind asynchronous case operations to an editable benchmark."""
        self._benchmark = benchmark
        self._uploaded_files: FileUploadCache = {}

    async def _draft_query(self, *, create: bool) -> str:
        if not self._benchmark.data.get("activeRevisionId"):
            return ""
        draft = await self._benchmark.revision_draft()
        if draft is None and create:
            draft = await self._benchmark.create_revision_draft()
        return "?revision=draft" if draft is not None else ""

    async def list(self) -> builtins.list[CreatedBenchmarkCase]:
        """List editable cases in deterministic draft order."""
        response = await self._benchmark._manager._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "GET",
                f"/benchmarks/{self._benchmark.id}/cases{await self._draft_query(create=False)}",
            )
        )
        return cast(list[CreatedBenchmarkCase], response["data"])

    async def add(
        self,
        *,
        input: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        expected_behavior: str | None = None,
        title: str | None = None,
    ) -> CreatedBenchmarkCase:
        """Upload nested schema file values and add one validated test case."""
        draft_query = await self._draft_query(create=True)
        body, case_uploads = _case_body_values(
            title=title,
            input=dict(input or {}),
            output=dict(output) if output is not None else None,
            expected_behavior=expected_behavior,
            input_schema=self._benchmark.data.get("inputSchema"),
            output_schema=self._benchmark.data.get("targetSchema"),
        )
        body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = await _upload_file_async(
                self._benchmark._manager._client,
                value,
                self._uploaded_files,
            )
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        return cast(
            CreatedBenchmarkCase,
            await self._benchmark._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "POST",
                    f"/benchmarks/{self._benchmark.id}/cases{draft_query}",
                    json=body,
                )
            ),
        )

    async def update(
        self,
        case_id: int,
        *,
        input: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        expected_behavior: str | None = None,
        title: str | None = None,
    ) -> CreatedBenchmarkCase:
        """Replace one case in the editable draft and reconcile its conflicts."""
        if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
            raise ValueError("case_id must be a positive integer")
        draft_query = await self._draft_query(create=True)
        body, case_uploads = _case_body_values(
            title=title,
            input=dict(input or {}),
            output=dict(output) if output is not None else None,
            expected_behavior=expected_behavior,
            input_schema=self._benchmark.data.get("inputSchema"),
            output_schema=self._benchmark.data.get("targetSchema"),
        )
        body["uploads"] = []
        for role, field_path, upload_path, value in case_uploads:
            uploaded = await _upload_file_async(
                self._benchmark._manager._client, value, self._uploaded_files
            )
            body["uploads"].append(
                {"role": role, "fieldPath": field_path, "uploadPath": upload_path, **uploaded}
            )
        separator = "&" if draft_query else "?"
        return cast(
            CreatedBenchmarkCase,
            await self._benchmark._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "PATCH",
                    f"/benchmarks/{self._benchmark.id}/cases{draft_query}{separator}caseId={case_id}",
                    json=body,
                )
            ),
        )

    async def add_many(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        batch_size: int = _DEFAULT_BULK_CASE_BATCH_SIZE,
        on_progress: Callable[[BulkCaseProgress], None] | None = None,
    ) -> BulkCaseResponse:
        """Add cases in confirmed batches without publishing the benchmark draft."""
        if not cases:
            raise ValueError("cases cannot be empty")
        size = _validate_batch_size(batch_size)
        result = _empty_bulk_result(len(cases))
        total_batches = (len(cases) + size - 1) // size
        draft = bool(await self._draft_query(create=True))
        for batch_index, offset in enumerate(range(0, len(cases), size)):
            batch = cases[offset : offset + size]
            try:
                items = await _bulk_items_async(
                    self._benchmark._manager._client,
                    batch,
                    input_schema=self._benchmark.data.get("inputSchema"),
                    output_schema=self._benchmark.data.get("targetSchema"),
                    cache=self._uploaded_files,
                )
                response = cast(
                    Mapping[str, Any],
                    await self._benchmark._manager._client._transport.request(  # noqa: SLF001
                        _bulk_request(self._benchmark.id, items, draft=draft)
                    ),
                )
            except ValueError:
                raise
            except Exception as error:
                raise BulkCaseUploadError(batch_index, result, error) from error
            _merge_bulk_response(result, response, len(batch))
            if on_progress is not None:
                on_progress(
                    _bulk_progress(
                        result,
                        batch_index=batch_index,
                        total_batches=total_batches,
                    )
                )
        return result


class AsyncAgentGymBenchmark:
    """Asynchronous editable benchmark resource."""

    def __init__(self, manager: AsyncAgentGymBenchmarks, data: BenchmarkDefinition) -> None:
        """Initialize from one benchmark definition response."""
        self._manager = manager
        self.data = data
        self.staged_message: str | None = None
        self.staged_conflicts: list[BenchmarkSchemaConflict] = []
        self.cases = AsyncAgentGymBenchmarkCases(self)

    @property
    def id(self) -> str:
        """Return the benchmark UUID."""
        return self.data["id"]

    @property
    def name(self) -> str:
        """Return the benchmark name."""
        return self.data["name"]

    async def refresh(self) -> AsyncAgentGymBenchmark:
        """Reload this editable definition."""
        self.data = await self._manager._get_data(self.id)
        return self

    @property
    def ready_for_submission(self) -> bool:
        """Return whether schemas, cases, and evaluators are ready to publish and run."""
        return bool(self.data.get("configuration", {}).get("readyForSubmission"))

    async def configure(
        self,
        *,
        name: str | None = None,
        goal: str | None = None,
        description: str | None = None,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        evaluators: Sequence[AgentEvaluator] | None = None,
    ) -> AsyncAgentGymBenchmark:
        """Configure the Agent goal, schemas, scoring, and independent evaluators."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if goal is not None:
            body["taskDescription"] = goal
        if description is not None:
            body["componentDescription"] = description
        if input_schema is not None:
            body["inputSchema"] = dict(input_schema)
        if output_schema is not None:
            body["targetSchema"] = dict(output_schema)
        if evaluators is not None:
            body["evaluators"] = [evaluator.as_request() for evaluator in evaluators]
        response = await self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "PATCH",
                f"/benchmarks/{self.id}",
                json=body,
            )
        )
        if response.get("staged") is True:
            staged = cast(StagedBenchmarkUpdate, response)
            draft = staged["draft"]
            task_snapshot = draft.get("taskSnapshot")
            if task_snapshot is not None:
                schema = task_snapshot.get("inputContract", {}).get("text", {}).get("schema")
                if schema is not None:
                    self.data["inputSchema"] = cast(dict[str, Any], schema)
            if "targetSchema" in draft:
                self.data["targetSchema"] = draft["targetSchema"]
            evaluator_snapshot = draft.get("evaluatorSnapshot")
            if evaluator_snapshot is not None and "evaluatorGraph" in evaluator_snapshot:
                self.data["evaluatorGraph"] = cast(
                    dict[str, Any], evaluator_snapshot["evaluatorGraph"]
                )
            self.staged_message = staged["message"]
            self.staged_conflicts = list(staged["conflicts"])
        else:
            self.data = cast(BenchmarkDefinition, response)
            self.staged_message = None
            self.staged_conflicts = []
        return self

    async def create_revision_draft(self) -> BenchmarkRevisionDraft:
        """Create or reuse the persisted draft for a published benchmark."""
        response = cast(
            BenchmarkRevisionDraftResponse,
            await self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("POST", f"/benchmarks/{self.id}/revisions/draft", json={})
            ),
        )
        draft = response["draft"]
        if draft is None:
            raise RuntimeError("revision draft creation returned no draft")
        return draft

    async def revisions(self) -> BenchmarkRevisionList:
        """List immutable revisions and current draft status."""
        return cast(
            BenchmarkRevisionList,
            await self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{self.id}/revisions")
            ),
        )

    async def draft_status(self) -> BenchmarkDraftStatus | None:
        """Return whether the editable draft differs from its active revision."""
        return (await self.revisions())["status"]

    async def revision_draft(self) -> BenchmarkRevisionDraft | None:
        """Return the persisted schema-migration draft, if one exists."""
        response = cast(
            BenchmarkRevisionDraftResponse,
            await self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{self.id}/revisions/draft")
            ),
        )
        return response["draft"]

    async def resolve_revision_draft(
        self,
        field: str,
        *,
        kind: str,
        value: Any = _MISSING,
    ) -> BenchmarkRevisionDraft:
        """Resolve one schema conflict by making it optional, defaulting, or editing cases."""
        normalized_field = field.strip()
        if not normalized_field:
            raise ValueError("field cannot be empty")
        if kind not in {"make_optional", "default_value", "edit_cases"}:
            raise ValueError("kind must be make_optional, default_value, or edit_cases")
        if kind == "default_value" and value is _MISSING:
            raise ValueError("value is required for a default_value resolution")
        if kind != "default_value" and value is not _MISSING:
            raise ValueError("value is only supported for a default_value resolution")
        resolution: dict[str, Any] = {"kind": kind}
        if value is not _MISSING:
            resolution["value"] = value
        response = cast(
            BenchmarkRevisionDraftResponse,
            await self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec(
                    "PATCH",
                    f"/benchmarks/{self.id}/revisions/draft",
                    json={"field": normalized_field, "resolution": resolution},
                )
            ),
        )
        draft = response["draft"]
        if draft is None:
            raise RuntimeError("revision draft resolution returned no draft")
        return draft

    async def abandon_revision_draft(self) -> None:
        """Discard the persisted schema-migration proposal without changing the active revision."""
        await self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec("DELETE", f"/benchmarks/{self.id}/revisions/draft")
        )

    async def publish(self) -> PublishedBenchmarkRevision:
        """Idempotently publish the current draft fingerprint."""
        return PublishedBenchmarkRevision.from_response(
            await self._manager._client._transport.request(  # noqa: SLF001
                RequestSpec("POST", f"/benchmarks/{self.id}/revisions", json={})
            )
        )

    async def delete(self) -> None:
        """Delete this editable benchmark and its dependent records."""
        await self._manager._client._transport.request(  # noqa: SLF001
            RequestSpec("DELETE", f"/benchmarks/{self.id}")
        )


class AsyncAgentGymBenchmarks:
    """Asynchronous benchmark authoring namespace."""

    def __init__(self, client: AsyncAgentGymClient, workspace_id: str | None) -> None:
        """Bind authoring operations to a client and optional workspace."""
        self._client = client
        self._workspace_id = workspace_id

    async def create(
        self,
        *,
        name: str,
        goal: str,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        evaluators: Sequence[AgentEvaluator] = (),
        description: str | None = None,
        workspace_id: str | None = None,
    ) -> AsyncAgentGymBenchmark:
        """Create a typed editable benchmark definition."""
        resolved_workspace = _workspace_id(workspace_id, self._workspace_id)
        response = await self._client._transport.request(  # noqa: SLF001
            RequestSpec(
                "POST",
                "/benchmarks",
                json=_create_body(
                    workspace_id=resolved_workspace,
                    name=name,
                    goal=goal,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    evaluators=evaluators,
                    description=description,
                ),
            )
        )
        return AsyncAgentGymBenchmark(self, cast(BenchmarkDefinition, response))

    async def list(
        self, *, workspace_id: str | None = None
    ) -> builtins.list[AsyncAgentGymBenchmark]:
        """List editable benchmarks in a workspace."""
        resolved_workspace = _workspace_id(workspace_id, self._workspace_id)
        response = await self._client._transport.request(  # noqa: SLF001
            RequestSpec("GET", "/benchmarks", params={"workspaceId": resolved_workspace})
        )
        return [
            AsyncAgentGymBenchmark(self, cast(BenchmarkDefinition, item))
            for item in response["data"]
        ]

    async def _get_data(self, benchmark_id: str) -> BenchmarkDefinition:
        require_uuid(benchmark_id, "benchmark_id")
        return cast(
            BenchmarkDefinition,
            await self._client._transport.request(  # noqa: SLF001
                RequestSpec("GET", f"/benchmarks/{benchmark_id}")
            ),
        )

    async def get(self, benchmark_id: str) -> AsyncAgentGymBenchmark:
        """Get one editable benchmark by UUID."""
        return AsyncAgentGymBenchmark(self, await self._get_data(benchmark_id))


__all__ = [
    "AgentGymBenchmark",
    "AgentGymBenchmarkCases",
    "AgentGymBenchmarks",
    "AsyncAgentGymBenchmark",
    "AsyncAgentGymBenchmarkCases",
    "AsyncAgentGymBenchmarks",
]
