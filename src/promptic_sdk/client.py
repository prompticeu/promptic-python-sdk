"""REST client for the Promptic platform API."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from typing_extensions import Unpack

from promptic_sdk.models import (
    AgentGymCodeFunctionTestRow,
    AIApplication,
    AvailableModelList,
    CodeFunctionConfig,
    CodeFunctionRowContract,
    CodeFunctionTestRow,
    Component,
    ComponentCreated,
    ComponentList,
    Dataset,
    DatasetCase,
    DatasetCaseCreate,
    DatasetCaseList,
    DatasetCaseUpdate,
    DatasetList,
    DatasetWithCases,
    DeployedPrompt,
    Deployment,
    DeploymentCreated,
    Evaluator,
    EvaluatorList,
    Experiment,
    ExperimentCancellation,
    ExperimentCasePage,
    ExperimentList,
    ExperimentStarted,
    IterationList,
    IterationWithScores,
    OptimizationBrief,
    PromptExperimentTaskType,
    PromptFormat,
    PromptMessage,
    ToolSelectionTestCase,
    ToolSelectionTool,
    ToolSource,
    Trace,
    TraceArtifact,
    TraceArtifactList,
    TraceList,
    TracingStats,
)

_DEFAULT_ENDPOINT = "https://promptic.eu"


def _atomic_write_bytes(path: str | os.PathLike[str], content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class PrompticAPIError(Exception):
    """Error returned by the Promptic API.

    Attributes:
        status_code: HTTP status code.
        message: Error message from the server.
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


def _normalize_tool_selection_tool(tool: ToolSelectionTool) -> dict[str, Any]:
    """Normalize a tool definition to the API shape.

    Accepts ``input_schema`` (snake) or ``inputSchema`` (camel); both are sent
    as ``inputSchema``.
    """
    out: dict[str, Any] = {"name": tool["name"], "description": tool["description"]}
    schema = tool.get("input_schema", tool.get("inputSchema"))
    if schema is not None:
        out["inputSchema"] = schema
    return out


def _normalize_tool_selection_test_case(test_case: ToolSelectionTestCase) -> dict[str, Any]:
    """Normalize a test case to the API shape.

    Accepts ``expected_tool`` (snake) or ``expectedTool`` (camel). Use an empty
    string (or ``"none"``) for queries that should trigger no tool.
    """
    raw = cast("dict[str, Any]", test_case)
    if "expected_tool" not in raw and "expectedTool" not in raw:
        msg = "Tool-selection test cases require expected_tool or expectedTool"
        raise ValueError(msg)
    expected = raw["expected_tool"] if "expected_tool" in raw else raw["expectedTool"]
    return {"query": test_case["query"], "expectedTool": expected}


def _prompt_hyperparameters(
    hyperparameters: dict[str, Any] | None, evaluation_only: bool
) -> dict[str, Any] | None:
    if not evaluation_only:
        return hyperparameters
    if hyperparameters is not None and hyperparameters.get("epochs", 0) != 0:
        raise ValueError("evaluation_only=True requires epochs=0 or no explicit epochs")
    return {"numFewShots": 8, **(hyperparameters or {}), "epochs": 0}


def code_function_test_body(
    config: CodeFunctionConfig,
    row: CodeFunctionTestRow | AgentGymCodeFunctionTestRow,
    row_contract: CodeFunctionRowContract,
) -> dict[str, Any]:
    """Build a function-test request; the default contract keeps the original payload."""
    body: dict[str, Any] = {"config": config, "row": row}
    if row_contract != "experiment":
        body["row_contract"] = row_contract
    return body


def _experiment_case_query(
    iteration_id: int,
    split: str,
    limit: int,
    cursor: str | None,
    sort: str | None,
    descending: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {"iterationId": iteration_id, "split": split, "limit": limit}
    if cursor is not None:
        params["cursor"] = cursor
    if sort is not None:
        params.update({"sort": sort, "descending": descending})
    return params


def _tool_selection_body(
    ai_component_id: str,
    tools: list[ToolSelectionTool],
    test_cases: list[ToolSelectionTestCase],
    *,
    target_model: str | None,
    tool_source: ToolSource,
    system_prompt: str | None,
    optimize_system_prompt: bool,
    evaluation_only: bool,
    epochs: int | None,
    train_split_ratio: float | None,
    name: str | None,
    description: str | None,
    optimization_guidance: OptimizationBrief | None = None,
) -> dict[str, Any]:
    """Build the request body shared by the sync and async clients."""
    body: dict[str, Any] = {
        "aiComponentId": ai_component_id,
        "tools": [_normalize_tool_selection_tool(t) for t in tools],
        "testCases": [_normalize_tool_selection_test_case(tc) for tc in test_cases],
        "toolSource": tool_source,
        "optimizeSystemPrompt": optimize_system_prompt,
    }
    if optimization_guidance is not None:
        body["optimizationGuidance"] = optimization_guidance
    if target_model is not None:
        body["targetModel"] = target_model
    if system_prompt is not None:
        body["systemPrompt"] = system_prompt
    if evaluation_only and epochs not in (None, 0):
        raise ValueError("evaluation_only=True requires epochs=0 or no explicit epochs")
    if epochs is not None or evaluation_only:
        body["epochs"] = 0 if evaluation_only else epochs
    if train_split_ratio is not None:
        body["trainSplitRatio"] = train_split_ratio
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    return body


class ModelsResource:
    """Model discovery for a synchronous Promptic client."""

    def __init__(self, client: PrompticClient) -> None:
        """Bind this resource to an authenticated client."""
        self._client = client

    def list(self) -> AvailableModelList:
        """List models available to the current AI Application."""
        return self._client._get("/models")


class AsyncModelsResource:
    """Model discovery for an asynchronous Promptic client."""

    def __init__(self, client: AsyncPrompticClient) -> None:
        """Bind this resource to an authenticated client."""
        self._client = client

    async def list(self) -> AvailableModelList:
        """List models available to the current AI Application."""
        return await self._client._get("/models")


@dataclass
class PrompticClient:
    """Client for interacting with the Promptic platform API.

    Args:
        api_key: Promptic API key. Falls back to ``PROMPTIC_API_KEY`` env var.
        access_token: Session token from device auth login. Falls back to
            ``PROMPTIC_ACCESS_TOKEN`` env var.
        ai_application_id: AI Application ID for session-based auth. Falls back to
            ``PROMPTIC_AI_APPLICATION_ID`` env var.
        workspace_id: Deprecated alias for ``ai_application_id``. Falls back to
            ``PROMPTIC_WORKSPACE_ID`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://promptic.eu``.
        timeout: HTTP request timeout in seconds.
    """

    api_key: str | None = None
    access_token: str | None = None
    ai_application_id: str | None = None
    workspace_id: str | None = None
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.Client = field(init=False, repr=False)

    @property
    def models(self) -> ModelsResource:
        """Access model discovery for the current AI Application."""
        return ModelsResource(self)

    def __post_init__(self) -> None:
        """Initialize the HTTP client."""
        self.api_key = self.api_key or os.environ.get("PROMPTIC_API_KEY")
        self.access_token = self.access_token or os.environ.get("PROMPTIC_ACCESS_TOKEN")
        # Resolve the AI Application scope, accepting the deprecated ``workspace_id``
        # argument and env var for backward compatibility.
        self.ai_application_id = (
            self.ai_application_id
            or self.workspace_id
            or os.environ.get("PROMPTIC_AI_APPLICATION_ID")
            or os.environ.get("PROMPTIC_WORKSPACE_ID")
        )
        # Keep the deprecated attribute in sync so existing readers still work.
        self.workspace_id = self.ai_application_id

        if not self.api_key and not self.access_token:
            msg = (
                "Authentication required. "
                "Run 'promptic login' or 'promptic configure', "
                "or set PROMPTIC_API_KEY / PROMPTIC_ACCESS_TOKEN."
            )
            raise ValueError(msg)

        self.endpoint = (
            self.endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
        ).rstrip("/")

        # Prefer session auth (access_token) over API key when both are present
        auth_headers: dict[str, str] = {}
        if self.access_token:
            auth_headers["Authorization"] = f"Bearer {self.access_token}"
            if self.ai_application_id:
                auth_headers["X-AI-Application-Id"] = self.ai_application_id
        elif self.api_key:
            auth_headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=f"{self.endpoint}/api/v1",
            headers=auth_headers,
            timeout=self.timeout,
        )

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send a request and return parsed JSON (or None for 204)."""
        resp = self._client.request(method, path, params=params, json=json)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("error", resp.text)
            except Exception:
                message = resp.text
            raise PrompticAPIError(resp.status_code, message)
        if resp.status_code == 204:
            return None
        return resp.json()

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> bytes:
        resp = self._client.request("GET", path, params=params, follow_redirects=True)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("error", resp.text)
            except Exception:
                message = resp.text
            raise PrompticAPIError(resp.status_code, message)
        return resp.content

    def _post(self, path: str, *, json: Any = None) -> Any:
        return self._request("POST", path, json=json)

    def _patch(self, path: str, *, json: Any = None) -> Any:
        return self._request("PATCH", path, json=json)

    def _delete(self, path: str) -> None:
        self._request("DELETE", path)

    # ── Traces ───────────────────────────────────────────────────────

    def list_traces(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        start_after: str | None = None,
        start_before: str | None = None,
    ) -> TraceList:
        """List traces with pagination and filters.

        Args:
            limit: Maximum number of traces to return (max 100).
            offset: Number of traces to skip.
            status: Filter by status ("ok" or "error").
            start_after: Only traces after this ISO timestamp.
            start_before: Only traces before this ISO timestamp.

        Returns:
            Dict with ``traces`` list and ``total`` count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if start_after:
            params["start_after"] = start_after
        if start_before:
            params["start_before"] = start_before
        return self._get("/traces", params=params)

    def get_trace(self, trace_id: str) -> Trace:
        """Get a single trace with all its spans and events."""
        return self._get(f"/traces/{trace_id}")

    def list_trace_artifacts(self, trace_id: str) -> TraceArtifactList:
        """List artifacts referenced by a trace."""
        return self._get(f"/traces/{trace_id}/artifacts")

    def get_artifact(self, artifact_id: str) -> TraceArtifact:
        """Get artifact metadata."""
        return self._get(f"/artifacts/{artifact_id}")

    def get_artifact_content(self, artifact_id: str) -> bytes:
        """Fetch artifact bytes."""
        return self._get_bytes(f"/artifacts/{artifact_id}/content")

    def download_artifact(self, artifact_id: str, path: str | os.PathLike[str]) -> None:
        """Download artifact bytes to a local path."""
        content = self.get_artifact_content(artifact_id)
        _atomic_write_bytes(path, content)

    def get_stats(self, *, days_back: int = 30) -> TracingStats:
        """Get aggregated tracing stats."""
        return self._get("/traces/stats", params={"days_back": days_back})

    # ── AI Application ───────────────────────────────────────────────

    def get_ai_application(self) -> AIApplication:
        """Get AI Application info for the current API key."""
        return self._get("/ai-application")

    def get_workspace(self) -> AIApplication:
        """Deprecated alias for :meth:`get_ai_application`."""
        return self.get_ai_application()

    # ── Components ───────────────────────────────────────────────────

    def list_components(self) -> ComponentList:
        """List all AI components in the AI Application."""
        return self._get("/components")

    def create_component(self, name: str, *, description: str | None = None) -> ComponentCreated:
        """Create a new AI component."""
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return self._post("/components", json=body)

    def get_component(self, component_id: str) -> Component:
        """Get an AI component by ID."""
        return self._get(f"/components/{component_id}")

    def update_component_rules(
        self, component_id: str, brief: OptimizationBrief, *, expected_version: int
    ) -> Component:
        """Replace shared rules; use the component's current version (zero if unset).

        A stale version returns HTTP 409. Fetch the component and reconcile edits
        before retrying. Pass an empty brief to clear shared rules.
        """
        return self._patch(
            f"/components/{component_id}",
            json={"optimizationRules": {"version": expected_version, "brief": brief}},
        )

    def delete_component(self, component_id: str) -> None:
        """Delete an AI component."""
        self._delete(f"/components/{component_id}")

    # ── Experiments ──────────────────────────────────────────────────

    def list_experiments(
        self,
        *,
        component_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ExperimentList:
        """List experiments with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if component_id:
            params["component_id"] = component_id
        if status:
            params["status"] = status
        return self._get("/experiments", params=params)

    def create_experiment(
        self,
        ai_component_id: str,
        target_model: str,
        *,
        task_type: PromptExperimentTaskType = "classification",
        initial_prompt: str | None = None,
        name: str | None = None,
        description: str | None = None,
        provider: str = "openai",
        optimizer: str = "prompticV2",
        hyperparameters: dict[str, Any] | None = None,
        initial_prediction_model_schema: dict[str, Any] | None = None,
        optimization_guidance: OptimizationBrief | None = None,
        input_variables: list[dict[str, str]] | None = None,
        evaluation_only: bool = False,
        prompt_format: PromptFormat | None = None,
        initial_prompt_messages: list[PromptMessage] | None = None,
    ) -> Experiment:
        """Create a pending prompt experiment.

        Set evaluation_only=True to score the supplied prompt once (epochs=0),
        without prompt optimization. Existing hyperparameters are preserved;
        an explicit nonzero epochs value conflicts with evaluation_only=True.
        Supply prompt_format="multi_message" and initial_prompt_messages to
        preserve an ordered chat template. Call start_experiment separately.
        """
        hyperparameters = _prompt_hyperparameters(hyperparameters, evaluation_only)
        if initial_prompt is not None and initial_prompt_messages is not None:
            raise ValueError("Supply initial_prompt or initial_prompt_messages, not both")
        body: dict[str, Any] = {
            "aiComponentId": ai_component_id,
            "targetModel": target_model,
            "taskType": task_type,
            "provider": provider,
            "optimizer": optimizer,
        }
        if optimization_guidance is not None:
            body["optimizationGuidance"] = optimization_guidance
        if initial_prompt is not None:
            body["initialPrompt"] = initial_prompt
        if prompt_format is not None:
            body["promptFormat"] = prompt_format
        if initial_prompt_messages is not None:
            body["initialPromptMessages"] = initial_prompt_messages
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if hyperparameters is not None:
            body["hyperparameters"] = hyperparameters
        if initial_prediction_model_schema is not None:
            body["initialPredictionModelSchema"] = initial_prediction_model_schema
        if input_variables is not None:
            body["inputVariables"] = input_variables
        return self._post("/experiments", json=body)

    def create_tool_selection_experiment(
        self,
        ai_component_id: str,
        *,
        tools: list[ToolSelectionTool],
        test_cases: list[ToolSelectionTestCase],
        target_model: str | None = None,
        tool_source: ToolSource = "manual",
        system_prompt: str | None = None,
        optimize_system_prompt: bool = False,
        optimization_guidance: OptimizationBrief | None = None,
        evaluation_only: bool = False,
        epochs: int | None = None,
        train_split_ratio: float | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Experiment:
        """Create a tool-selection ("toolSelection") experiment.

        Creates the experiment together with its tool definitions, test cases,
        and the tool-selection scorer in one call. The optimizer rewrites the
        tool descriptions (and, optionally, the system prompt) so the target
        model routes each query to the expected tool. The experiment is created
        pending — call ``start_experiment`` to run it.

        Args:
            ai_component_id: AI component the experiment belongs to.
            tools: Tool definitions, each ``{"name", "description"}`` with an
                optional ``input_schema`` (or ``inputSchema``). At least one.
            test_cases: Each ``{"query", "expected_tool"}`` (or ``expectedTool``).
                Use an empty string (or ``"none"``) for the expected tool when the
                query should trigger no tool. At least one.
            target_model: Model whose tool selection is optimized. When omitted,
                the platform applies its default.
            tool_source: Provenance of the tool definitions — ``"manual"``
                (supplied directly, the default) or ``"mcp"`` (fetched from an
                MCP server).
            system_prompt: Optional fixed system prompt used as context during
                evaluation.
            optimize_system_prompt: When True, the optimizer rewrites the system
                prompt alongside the tool descriptions.
            evaluation_only: Evaluate the supplied tools without rewriting them (epochs=0).
            epochs: Optional number of optimization rounds; 0 evaluates only.
            train_split_ratio: Optional training fraction in ``[0.3, 0.9]``.
                Omit to train and score on all test cases.
            name: Optional experiment name.
            optimization_guidance: Optimization rules for this experiment.
            description: Optional experiment description.

        Returns:
            The newly created (pending) experiment.
        """
        body = _tool_selection_body(
            ai_component_id,
            tools,
            test_cases,
            target_model=target_model,
            tool_source=tool_source,
            system_prompt=system_prompt,
            optimize_system_prompt=optimize_system_prompt,
            optimization_guidance=optimization_guidance,
            evaluation_only=evaluation_only,
            epochs=epochs,
            train_split_ratio=train_split_ratio,
            name=name,
            description=description,
        )
        return self._post("/experiments/tool-selection", json=body)

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID."""
        return self._get(f"/experiments/{experiment_id}")

    def update_experiment(
        self,
        experiment_id: str,
        *,
        optimization_guidance: OptimizationBrief | None = None,
        **updates: Any,
    ) -> Experiment:
        """Update a pending or failed experiment.

        Accepts keyword arguments matching experiment fields:
        name, description, targetModel, provider, taskType, optimizer,
        promptFormat, initialPromptMessages, hyperparameters, initialPredictionModelSchema.
        ``optimization_guidance`` replaces this experiment's rules and locks;
        pass ``{}`` to clear them. Inherited component rules remain server-managed.
        """
        if optimization_guidance is not None:
            if "optimizationGuidance" in updates:
                raise ValueError("Supply optimization_guidance or optimizationGuidance, not both")
            updates["optimizationGuidance"] = optimization_guidance
        return self._patch(f"/experiments/{experiment_id}", json=updates)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        self._delete(f"/experiments/{experiment_id}")

    def cancel_experiment(self, experiment_id: str) -> ExperimentCancellation:
        """Request cancellation without waiting for the worker to stop.

        Ownerless pending/scheduled runs return ``canceled`` immediately.
        Running runs return ``canceling`` while in-flight work drains; poll
        ``get_experiment`` for confirmation. Repeated requests are idempotent.
        Already-started work may still incur usage charges.

        Raises:
            PrompticAPIError: ``409`` for a non-cancelable state, ``404`` when
                not found or outside the current AI Application, or ``401``
                when unauthenticated.
        """
        return self._post(f"/experiments/{experiment_id}/cancel")

    def start_experiment(self, experiment_id: str) -> ExperimentStarted:
        """Start a pending experiment in its configured evaluation or optimization mode.

        Raises:
            PrompticAPIError: ``402`` when platform billing is enabled and the
                AI Application's organization has no active subscription and payment
                method, or is blocked by the free-tier limit.
        """
        return self._post(f"/experiments/{experiment_id}/start")

    def duplicate_experiment(
        self,
        experiment_id: str,
        *,
        continue_from_optimized: bool = False,
        initial_prompt_override: str | None = None,
    ) -> Experiment:
        """Duplicate an experiment (clones dataset cases + evaluators).

        Creates a new experiment under the same AI component as the source.
        By default the new experiment starts from the source's initial
        prompt; pass ``continue_from_optimized=True`` to seed it from the
        source's best optimized prompt instead (the "continue" flow), or
        ``initial_prompt_override`` to override with custom text.

        Args:
            experiment_id: Source experiment ID.
            continue_from_optimized: When True, use the source's best
                iteration prompt as the new experiment's initial prompt.
            initial_prompt_override: Optional explicit initial prompt
                text. Ignored if ``continue_from_optimized`` is True.

        Returns:
            The newly created experiment (with a ``modelUnavailable`` flag
            set when the source's target model is no longer available in
            the AI Application).
        """
        body: dict[str, Any] = {}
        if continue_from_optimized:
            body["continueFromOptimized"] = True
        if initial_prompt_override is not None:
            body["initialPromptOverride"] = initial_prompt_override
        return self._post(f"/experiments/{experiment_id}/duplicate", json=body)

    def list_experiment_cases(
        self,
        experiment_id: str,
        *,
        iteration_id: int,
        split: Literal["all", "train", "eval"] = "all",
        limit: int = 25,
        cursor: str | None = None,
        sort: str | None = None,
        descending: bool = False,
    ) -> ExperimentCasePage:
        """Read one page of predictions and per-evaluator scores for an iteration.

        iteration_id is the persisted iteration ID, not its iteration number.
        Follow nextCursor explicitly; limits are 1-100. Use sort="score" to
        inspect low-scoring cases first across the complete selected split.
        Requires a server release with the experiment case-results API.
        """
        return self._get(
            f"/experiments/{experiment_id}/cases",
            params=_experiment_case_query(iteration_id, split, limit, cursor, sort, descending),
        )

    # ── Evaluators ───────────────────────────────────────────────────

    def test_code_evaluator(
        self,
        config: CodeFunctionConfig,
        row: CodeFunctionTestRow | AgentGymCodeFunctionTestRow,
        *,
        row_contract: CodeFunctionRowContract = "experiment",
    ) -> dict[str, Any]:
        """Queue a sandbox function test with explicit output, without model calls.

        Use ``row_contract="agent_gym"`` with an ``AgentGymCodeFunctionTestRow`` to test
        an Agent Gym ``CodeFunction``. Sandbox usage may be billed. Poll
        get_code_evaluator_test with the returned id. This neither saves an evaluator
        nor adds a model-quality score.
        """
        return self._post(
            "/code-evaluator-tests", json=code_function_test_body(config, row, row_contract)
        )

    def get_code_evaluator_test(self, test_id: str) -> dict[str, Any]:
        """Read a function test's status, score/reason, or safe error diagnostics."""
        return self._get(f"/code-evaluator-tests/{test_id}")

    def list_evaluators(self, experiment_id: str) -> EvaluatorList:
        """List evaluators for an experiment."""
        return self._get(f"/experiments/{experiment_id}/evaluators")

    def create_evaluators(
        self, experiment_id: str, evaluators: list[dict[str, Any]]
    ) -> EvaluatorList:
        """Create evaluators for an experiment (batch)."""
        return self._post(f"/experiments/{experiment_id}/evaluators", json=evaluators)

    def update_evaluator(self, experiment_id: str, evaluator_id: str, **data: Any) -> Evaluator:
        """Update an evaluator."""
        return self._patch(f"/experiments/{experiment_id}/evaluators/{evaluator_id}", json=data)

    def delete_evaluator(self, experiment_id: str, evaluator_id: str) -> None:
        """Delete an evaluator."""
        self._delete(f"/experiments/{experiment_id}/evaluators/{evaluator_id}")

    # ── Iterations ───────────────────────────────────────────────────

    def list_iterations(self, experiment_id: str) -> IterationList:
        """List iterations for an experiment."""
        return self._get(f"/experiments/{experiment_id}/iterations")

    def get_iteration(self, experiment_id: str, iteration_id: int) -> IterationWithScores:
        """Get an iteration with evaluator scores."""
        return self._get(f"/experiments/{experiment_id}/iterations/{iteration_id}")

    def get_best_iteration(self, experiment_id: str) -> IterationWithScores:
        """Get the best-scoring iteration for an experiment."""
        return self._get(f"/experiments/{experiment_id}/iterations/best")

    # ── Deployments ──────────────────────────────────────────────────

    def get_deployment(self, component_id: str) -> Deployment | None:
        """Get current deployment for a component. Returns None if not deployed."""
        return self._get(f"/components/{component_id}/deployment")

    def deploy(self, component_id: str, experiment_id: str) -> DeploymentCreated:
        """Deploy an experiment to a component."""
        return self._post(
            f"/components/{component_id}/deployment",
            json={"experimentId": experiment_id},
        )

    def undeploy(self, component_id: str) -> None:
        """Remove deployment from a component."""
        self._delete(f"/components/{component_id}/deployment")

    def get_deployed_prompt(self, component_id: str) -> DeployedPrompt | None:
        """Get the deployed prompt for a component. Returns None if not deployed."""
        return self._get(f"/components/{component_id}/deployment/prompt")

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
        component_id: str,
        name: str,
        *,
        description: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Dataset:
        """Create a dataset for an AI component.

        Args:
            component_id: AI component ID.
            name: Dataset name.
            description: Optional description.
            trace_ids: Optional list of trace DB IDs to include.
        """
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if trace_ids is not None:
            body["traceIds"] = trace_ids
        return self._post(f"/components/{component_id}/datasets", json=body)

    def list_datasets(self, component_id: str) -> DatasetList:
        """List datasets for an AI component."""
        return self._get(f"/components/{component_id}/datasets")

    def get_dataset(self, component_id: str, dataset_id: str) -> DatasetWithCases:
        """Get a dataset with its canonical cases."""
        return self._get(f"/components/{component_id}/datasets/{dataset_id}")

    def delete_dataset(self, component_id: str, dataset_id: str) -> None:
        """Delete a dataset."""
        self._delete(f"/components/{component_id}/datasets/{dataset_id}")

    def list_dataset_cases(self, component_id: str, dataset_id: str) -> DatasetCaseList:
        """List the canonical cases in a dataset."""
        return self._get(f"/components/{component_id}/datasets/{dataset_id}/cases")

    def get_dataset_case(self, component_id: str, dataset_id: str, case_id: int) -> DatasetCase:
        """Get one canonical dataset case."""
        return self._get(f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}")

    def create_dataset_cases(
        self,
        component_id: str,
        dataset_id: str,
        cases: list[DatasetCaseCreate],
    ) -> DatasetCaseList:
        """Create canonical JSON cases in a dataset."""
        return self._post(
            f"/components/{component_id}/datasets/{dataset_id}/cases",
            json=cases,
        )

    def update_dataset_case(
        self,
        component_id: str,
        dataset_id: str,
        case_id: int,
        **data: Unpack[DatasetCaseUpdate],
    ) -> DatasetCase:
        """Update one canonical dataset case."""
        return self._patch(
            f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}",
            json=data,
        )

    def delete_dataset_case(self, component_id: str, dataset_id: str, case_id: int) -> None:
        """Delete one canonical dataset case."""
        self._delete(f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}")

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> PrompticClient:
        """Support use as context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close on context manager exit."""
        self.close()


@dataclass
class AsyncPrompticClient:
    """Async client for interacting with the Promptic platform API.

    Args:
        api_key: Promptic API key. Falls back to ``PROMPTIC_API_KEY`` env var.
        access_token: Session token from device auth login. Falls back to
            ``PROMPTIC_ACCESS_TOKEN`` env var.
        ai_application_id: AI Application ID for session-based auth. Falls back to
            ``PROMPTIC_AI_APPLICATION_ID`` env var.
        workspace_id: Deprecated alias for ``ai_application_id``. Falls back to
            ``PROMPTIC_WORKSPACE_ID`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://promptic.eu``.
        timeout: HTTP request timeout in seconds.
    """

    api_key: str | None = None
    access_token: str | None = None
    ai_application_id: str | None = None
    workspace_id: str | None = None
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.AsyncClient = field(init=False, repr=False)

    @property
    def models(self) -> AsyncModelsResource:
        """Access model discovery for the current AI Application."""
        return AsyncModelsResource(self)

    def __post_init__(self) -> None:
        """Initialize the HTTP client."""
        self.api_key = self.api_key or os.environ.get("PROMPTIC_API_KEY")
        self.access_token = self.access_token or os.environ.get("PROMPTIC_ACCESS_TOKEN")
        # Resolve the AI Application scope, accepting the deprecated ``workspace_id``
        # argument and env var for backward compatibility.
        self.ai_application_id = (
            self.ai_application_id
            or self.workspace_id
            or os.environ.get("PROMPTIC_AI_APPLICATION_ID")
            or os.environ.get("PROMPTIC_WORKSPACE_ID")
        )
        # Keep the deprecated attribute in sync so existing readers still work.
        self.workspace_id = self.ai_application_id

        if not self.api_key and not self.access_token:
            msg = (
                "Authentication required. "
                "Run 'promptic login' or 'promptic configure', "
                "or set PROMPTIC_API_KEY / PROMPTIC_ACCESS_TOKEN."
            )
            raise ValueError(msg)

        self.endpoint = (
            self.endpoint or os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)
        ).rstrip("/")

        # Prefer session auth (access_token) over API key when both are present
        auth_headers: dict[str, str] = {}
        if self.access_token:
            auth_headers["Authorization"] = f"Bearer {self.access_token}"
            if self.ai_application_id:
                auth_headers["X-AI-Application-Id"] = self.ai_application_id
        elif self.api_key:
            auth_headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=f"{self.endpoint}/api/v1",
            headers=auth_headers,
            timeout=self.timeout,
        )

    # ── HTTP helpers ─────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send a request and return parsed JSON (or None for 204)."""
        resp = await self._client.request(method, path, params=params, json=json)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("error", resp.text)
            except Exception:
                message = resp.text
            raise PrompticAPIError(resp.status_code, message)
        if resp.status_code == 204:
            return None
        return resp.json()

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def _get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> bytes:
        resp = await self._client.request("GET", path, params=params, follow_redirects=True)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("error", resp.text)
            except Exception:
                message = resp.text
            raise PrompticAPIError(resp.status_code, message)
        return resp.content

    async def _post(self, path: str, *, json: Any = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _patch(self, path: str, *, json: Any = None) -> Any:
        return await self._request("PATCH", path, json=json)

    async def _delete(self, path: str) -> None:
        await self._request("DELETE", path)

    # ── Traces ───────────────────────────────────────────────────────

    async def list_traces(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        start_after: str | None = None,
        start_before: str | None = None,
    ) -> TraceList:
        """List traces with pagination and filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if start_after:
            params["start_after"] = start_after
        if start_before:
            params["start_before"] = start_before
        return await self._get("/traces", params=params)

    async def get_trace(self, trace_id: str) -> Trace:
        """Get a single trace with all its spans and events."""
        return await self._get(f"/traces/{trace_id}")

    async def list_trace_artifacts(self, trace_id: str) -> TraceArtifactList:
        """List artifacts referenced by a trace."""
        return await self._get(f"/traces/{trace_id}/artifacts")

    async def get_artifact(self, artifact_id: str) -> TraceArtifact:
        """Get artifact metadata."""
        return await self._get(f"/artifacts/{artifact_id}")

    async def get_artifact_content(self, artifact_id: str) -> bytes:
        """Fetch artifact bytes."""
        return await self._get_bytes(f"/artifacts/{artifact_id}/content")

    async def download_artifact(self, artifact_id: str, path: str | os.PathLike[str]) -> None:
        """Download artifact bytes to a local path."""
        content = await self.get_artifact_content(artifact_id)
        await asyncio.to_thread(_atomic_write_bytes, path, content)

    async def get_stats(self, *, days_back: int = 30) -> TracingStats:
        """Get aggregated tracing stats."""
        return await self._get("/traces/stats", params={"days_back": days_back})

    # ── AI Application ───────────────────────────────────────────────

    async def get_ai_application(self) -> AIApplication:
        """Get AI Application info for the current API key."""
        return await self._get("/ai-application")

    async def get_workspace(self) -> AIApplication:
        """Deprecated alias for :meth:`get_ai_application`."""
        return await self.get_ai_application()

    # ── Components ───────────────────────────────────────────────────

    async def list_components(self) -> ComponentList:
        """List all AI components in the AI Application."""
        return await self._get("/components")

    async def create_component(
        self, name: str, *, description: str | None = None
    ) -> ComponentCreated:
        """Create a new AI component."""
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return await self._post("/components", json=body)

    async def get_component(self, component_id: str) -> Component:
        """Get an AI component by ID."""
        return await self._get(f"/components/{component_id}")

    async def update_component_rules(
        self, component_id: str, brief: OptimizationBrief, *, expected_version: int
    ) -> Component:
        """Replace shared rules; use the component's current version (zero if unset).

        A stale version returns HTTP 409. Fetch the component and reconcile edits
        before retrying. Pass an empty brief to clear shared rules.
        """
        return await self._patch(
            f"/components/{component_id}",
            json={"optimizationRules": {"version": expected_version, "brief": brief}},
        )

    async def delete_component(self, component_id: str) -> None:
        """Delete an AI component."""
        await self._delete(f"/components/{component_id}")

    # ── Experiments ──────────────────────────────────────────────────

    async def list_experiments(
        self,
        *,
        component_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ExperimentList:
        """List experiments with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if component_id:
            params["component_id"] = component_id
        if status:
            params["status"] = status
        return await self._get("/experiments", params=params)

    async def create_experiment(
        self,
        ai_component_id: str,
        target_model: str,
        *,
        task_type: PromptExperimentTaskType = "classification",
        initial_prompt: str | None = None,
        name: str | None = None,
        description: str | None = None,
        provider: str = "openai",
        optimizer: str = "prompticV2",
        hyperparameters: dict[str, Any] | None = None,
        initial_prediction_model_schema: dict[str, Any] | None = None,
        optimization_guidance: OptimizationBrief | None = None,
        input_variables: list[dict[str, str]] | None = None,
        evaluation_only: bool = False,
        prompt_format: PromptFormat | None = None,
        initial_prompt_messages: list[PromptMessage] | None = None,
    ) -> Experiment:
        """Create a pending prompt experiment.

        Set evaluation_only=True to score the supplied prompt once (epochs=0),
        without prompt optimization. Existing hyperparameters are preserved;
        an explicit nonzero epochs value conflicts with evaluation_only=True.
        Supply prompt_format="multi_message" and initial_prompt_messages to
        preserve an ordered chat template. Call start_experiment separately.
        """
        hyperparameters = _prompt_hyperparameters(hyperparameters, evaluation_only)
        if initial_prompt is not None and initial_prompt_messages is not None:
            raise ValueError("Supply initial_prompt or initial_prompt_messages, not both")
        body: dict[str, Any] = {
            "aiComponentId": ai_component_id,
            "targetModel": target_model,
            "taskType": task_type,
            "provider": provider,
            "optimizer": optimizer,
        }
        if optimization_guidance is not None:
            body["optimizationGuidance"] = optimization_guidance
        if initial_prompt is not None:
            body["initialPrompt"] = initial_prompt
        if prompt_format is not None:
            body["promptFormat"] = prompt_format
        if initial_prompt_messages is not None:
            body["initialPromptMessages"] = initial_prompt_messages
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if hyperparameters is not None:
            body["hyperparameters"] = hyperparameters
        if initial_prediction_model_schema is not None:
            body["initialPredictionModelSchema"] = initial_prediction_model_schema
        if input_variables is not None:
            body["inputVariables"] = input_variables
        return await self._post("/experiments", json=body)

    async def create_tool_selection_experiment(
        self,
        ai_component_id: str,
        *,
        tools: list[ToolSelectionTool],
        test_cases: list[ToolSelectionTestCase],
        target_model: str | None = None,
        tool_source: ToolSource = "manual",
        system_prompt: str | None = None,
        optimize_system_prompt: bool = False,
        optimization_guidance: OptimizationBrief | None = None,
        evaluation_only: bool = False,
        epochs: int | None = None,
        train_split_ratio: float | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Experiment:
        """Create a tool-selection ("toolSelection") experiment.

        Creates the experiment together with its tool definitions, test cases,
        and the tool-selection scorer in one call. The optimizer rewrites the
        tool descriptions (and, optionally, the system prompt) so the target
        model routes each query to the expected tool. The experiment is created
        pending — call ``start_experiment`` to run it.

        Args:
            ai_component_id: AI component the experiment belongs to.
            tools: Tool definitions, each ``{"name", "description"}`` with an
                optional ``input_schema`` (or ``inputSchema``). At least one.
            test_cases: Each ``{"query", "expected_tool"}`` (or ``expectedTool``).
                Use an empty string (or ``"none"``) for the expected tool when the
                query should trigger no tool. At least one.
            target_model: Model whose tool selection is optimized. When omitted,
                the platform applies its default.
            tool_source: Provenance of the tool definitions — ``"manual"``
                (supplied directly, the default) or ``"mcp"`` (fetched from an
                MCP server).
            system_prompt: Optional fixed system prompt used as context during
                evaluation.
            optimize_system_prompt: When True, the optimizer rewrites the system
                prompt alongside the tool descriptions.
            evaluation_only: Evaluate the supplied tools without rewriting them (epochs=0).
            epochs: Optional number of optimization rounds; 0 evaluates only.
            train_split_ratio: Optional training fraction in ``[0.3, 0.9]``.
                Omit to train and score on all test cases.
            name: Optional experiment name.
            optimization_guidance: Optimization rules for this experiment.
            description: Optional experiment description.

        Returns:
            The newly created (pending) experiment.
        """
        body = _tool_selection_body(
            ai_component_id,
            tools,
            test_cases,
            target_model=target_model,
            tool_source=tool_source,
            system_prompt=system_prompt,
            optimize_system_prompt=optimize_system_prompt,
            optimization_guidance=optimization_guidance,
            evaluation_only=evaluation_only,
            epochs=epochs,
            train_split_ratio=train_split_ratio,
            name=name,
            description=description,
        )
        return await self._post("/experiments/tool-selection", json=body)

    async def get_experiment(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID."""
        return await self._get(f"/experiments/{experiment_id}")

    async def update_experiment(
        self,
        experiment_id: str,
        *,
        optimization_guidance: OptimizationBrief | None = None,
        **updates: Any,
    ) -> Experiment:
        """Update a pending or failed experiment."""
        if optimization_guidance is not None:
            if "optimizationGuidance" in updates:
                raise ValueError("Supply optimization_guidance or optimizationGuidance, not both")
            updates["optimizationGuidance"] = optimization_guidance
        return await self._patch(f"/experiments/{experiment_id}", json=updates)

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        await self._delete(f"/experiments/{experiment_id}")

    async def cancel_experiment(self, experiment_id: str) -> ExperimentCancellation:
        """Request cancellation without waiting for the worker to stop.

        Returns ``canceled`` immediately for ownerless pending/scheduled runs,
        or ``canceling`` while a running worker drains in-flight work. Poll
        ``get_experiment`` for confirmation. Repeated requests are idempotent;
        already-started work may still incur usage charges.

        Raises:
            PrompticAPIError: ``409`` for a non-cancelable state, ``404`` when
                not found or outside the current AI Application, or ``401``
                when unauthenticated.
        """
        return await self._post(f"/experiments/{experiment_id}/cancel")

    async def start_experiment(self, experiment_id: str) -> ExperimentStarted:
        """Start a pending experiment in its configured evaluation or optimization mode.

        Raises:
            PrompticAPIError: ``402`` when platform billing is enabled and the
                AI Application's organization has no active subscription and payment
                method, or is blocked by the free-tier limit.
        """
        return await self._post(f"/experiments/{experiment_id}/start")

    async def duplicate_experiment(
        self,
        experiment_id: str,
        *,
        continue_from_optimized: bool = False,
        initial_prompt_override: str | None = None,
    ) -> Experiment:
        """Duplicate an experiment (clones dataset cases + evaluators).

        Creates a new experiment under the same AI component as the source.
        By default the new experiment starts from the source's initial
        prompt; pass ``continue_from_optimized=True`` to seed it from the
        source's best optimized prompt instead (the "continue" flow), or
        ``initial_prompt_override`` to override with custom text.

        Args:
            experiment_id: Source experiment ID.
            continue_from_optimized: When True, use the source's best
                iteration prompt as the new experiment's initial prompt.
            initial_prompt_override: Optional explicit initial prompt
                text. Ignored if ``continue_from_optimized`` is True.

        Returns:
            The newly created experiment (with a ``modelUnavailable`` flag
            set when the source's target model is no longer available in
            the AI Application).
        """
        body: dict[str, Any] = {}
        if continue_from_optimized:
            body["continueFromOptimized"] = True
        if initial_prompt_override is not None:
            body["initialPromptOverride"] = initial_prompt_override
        return await self._post(f"/experiments/{experiment_id}/duplicate", json=body)

    async def list_experiment_cases(
        self,
        experiment_id: str,
        *,
        iteration_id: int,
        split: Literal["all", "train", "eval"] = "all",
        limit: int = 25,
        cursor: str | None = None,
        sort: str | None = None,
        descending: bool = False,
    ) -> ExperimentCasePage:
        """Read one page of predictions and per-evaluator scores for an iteration.

        iteration_id is the persisted iteration ID, not its iteration number.
        Follow nextCursor explicitly; limits are 1-100. Use sort="score" to
        inspect low-scoring cases first across the complete selected split.
        Requires a server release with the experiment case-results API.
        """
        return await self._get(
            f"/experiments/{experiment_id}/cases",
            params=_experiment_case_query(iteration_id, split, limit, cursor, sort, descending),
        )

    # ── Evaluators ───────────────────────────────────────────────────

    async def test_code_evaluator(
        self,
        config: CodeFunctionConfig,
        row: CodeFunctionTestRow | AgentGymCodeFunctionTestRow,
        *,
        row_contract: CodeFunctionRowContract = "experiment",
    ) -> dict[str, Any]:
        """Queue a billable sandbox probe with explicit output and no model calls."""
        return await self._post(
            "/code-evaluator-tests", json=code_function_test_body(config, row, row_contract)
        )

    async def get_code_evaluator_test(self, test_id: str) -> dict[str, Any]:
        """Read a function test's status and safe diagnostics."""
        return await self._get(f"/code-evaluator-tests/{test_id}")

    async def list_evaluators(self, experiment_id: str) -> EvaluatorList:
        """List evaluators for an experiment."""
        return await self._get(f"/experiments/{experiment_id}/evaluators")

    async def create_evaluators(
        self, experiment_id: str, evaluators: list[dict[str, Any]]
    ) -> EvaluatorList:
        """Create evaluators for an experiment (batch)."""
        return await self._post(f"/experiments/{experiment_id}/evaluators", json=evaluators)

    async def update_evaluator(
        self, experiment_id: str, evaluator_id: str, **data: Any
    ) -> Evaluator:
        """Update an evaluator."""
        return await self._patch(
            f"/experiments/{experiment_id}/evaluators/{evaluator_id}", json=data
        )

    async def delete_evaluator(self, experiment_id: str, evaluator_id: str) -> None:
        """Delete an evaluator."""
        await self._delete(f"/experiments/{experiment_id}/evaluators/{evaluator_id}")

    # ── Iterations ───────────────────────────────────────────────────

    async def list_iterations(self, experiment_id: str) -> IterationList:
        """List iterations for an experiment."""
        return await self._get(f"/experiments/{experiment_id}/iterations")

    async def get_iteration(self, experiment_id: str, iteration_id: int) -> IterationWithScores:
        """Get an iteration with evaluator scores."""
        return await self._get(f"/experiments/{experiment_id}/iterations/{iteration_id}")

    async def get_best_iteration(self, experiment_id: str) -> IterationWithScores:
        """Get the best-scoring iteration for an experiment."""
        return await self._get(f"/experiments/{experiment_id}/iterations/best")

    # ── Deployments ──────────────────────────────────────────────────

    async def get_deployment(self, component_id: str) -> Deployment | None:
        """Get current deployment for a component. Returns None if not deployed."""
        return await self._get(f"/components/{component_id}/deployment")

    async def deploy(self, component_id: str, experiment_id: str) -> DeploymentCreated:
        """Deploy an experiment to a component."""
        return await self._post(
            f"/components/{component_id}/deployment",
            json={"experimentId": experiment_id},
        )

    async def undeploy(self, component_id: str) -> None:
        """Remove deployment from a component."""
        await self._delete(f"/components/{component_id}/deployment")

    async def get_deployed_prompt(self, component_id: str) -> DeployedPrompt | None:
        """Get the deployed prompt for a component. Returns None if not deployed."""
        return await self._get(f"/components/{component_id}/deployment/prompt")

    # ── Datasets ─────────────────────────────────────────────────────

    async def create_dataset(
        self,
        component_id: str,
        name: str,
        *,
        description: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Dataset:
        """Create a dataset for an AI component."""
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if trace_ids is not None:
            body["traceIds"] = trace_ids
        return await self._post(f"/components/{component_id}/datasets", json=body)

    async def list_datasets(self, component_id: str) -> DatasetList:
        """List datasets for an AI component."""
        return await self._get(f"/components/{component_id}/datasets")

    async def get_dataset(self, component_id: str, dataset_id: str) -> DatasetWithCases:
        """Get a dataset with its canonical cases."""
        return await self._get(f"/components/{component_id}/datasets/{dataset_id}")

    async def delete_dataset(self, component_id: str, dataset_id: str) -> None:
        """Delete a dataset."""
        await self._delete(f"/components/{component_id}/datasets/{dataset_id}")

    async def list_dataset_cases(self, component_id: str, dataset_id: str) -> DatasetCaseList:
        """List the canonical cases in a dataset."""
        return await self._get(f"/components/{component_id}/datasets/{dataset_id}/cases")

    async def get_dataset_case(
        self, component_id: str, dataset_id: str, case_id: int
    ) -> DatasetCase:
        """Get one canonical dataset case."""
        return await self._get(f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}")

    async def create_dataset_cases(
        self,
        component_id: str,
        dataset_id: str,
        cases: list[DatasetCaseCreate],
    ) -> DatasetCaseList:
        """Create canonical JSON cases in a dataset."""
        return await self._post(
            f"/components/{component_id}/datasets/{dataset_id}/cases",
            json=cases,
        )

    async def update_dataset_case(
        self,
        component_id: str,
        dataset_id: str,
        case_id: int,
        **data: Unpack[DatasetCaseUpdate],
    ) -> DatasetCase:
        """Update one canonical dataset case."""
        return await self._patch(
            f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}",
            json=data,
        )

    async def delete_dataset_case(self, component_id: str, dataset_id: str, case_id: int) -> None:
        """Delete one canonical dataset case."""
        await self._delete(f"/components/{component_id}/datasets/{dataset_id}/cases/{case_id}")

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncPrompticClient:
        """Support use as async context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close on context manager exit."""
        await self.close()
