"""REST client for the Promptic platform API."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from promptic_sdk.models import (
    AgentEvaluation,
    AgentEvaluationList,
    Annotation,
    AnnotationList,
    Component,
    ComponentCreated,
    ComponentList,
    Dataset,
    DatasetList,
    DatasetWithItems,
    DeployedPrompt,
    Deployment,
    DeploymentCreated,
    Evaluator,
    EvaluatorList,
    Experiment,
    ExperimentList,
    ExperimentStarted,
    IterationList,
    IterationWithScores,
    Observation,
    ObservationList,
    Run,
    RunList,
    RunWithTraces,
    Trace,
    TraceList,
    TracingStats,
    Workspace,
)

_DEFAULT_ENDPOINT = "https://promptic.eu"


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


def _normalize_observation_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy ``input`` observation payloads to ``variables``.

    The platform stores observation inputs as named variables. For convenience,
    the SDK still accepts the old ``input=...`` shape and sends it as the
    default ``input`` variable.
    """
    payload = dict(data)
    legacy_input = payload.pop("input", None)
    if legacy_input is not None and "variables" not in payload:
        payload["variables"] = {"input": legacy_input}
    return payload


def _normalize_observation_payloads(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_observation_payload(obs) for obs in observations]


@dataclass
class PrompticClient:
    """Client for interacting with the Promptic platform API.

    Args:
        api_key: Promptic API key. Falls back to ``PROMPTIC_API_KEY`` env var.
        access_token: Session token from device auth login. Falls back to
            ``PROMPTIC_ACCESS_TOKEN`` env var.
        workspace_id: Workspace ID for session-based auth. Falls back to
            ``PROMPTIC_WORKSPACE_ID`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://promptic.eu``.
        timeout: HTTP request timeout in seconds.
    """

    api_key: str | None = None
    access_token: str | None = None
    workspace_id: str | None = None
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the HTTP client."""
        self.api_key = self.api_key or os.environ.get("PROMPTIC_API_KEY")
        self.access_token = self.access_token or os.environ.get("PROMPTIC_ACCESS_TOKEN")
        self.workspace_id = self.workspace_id or os.environ.get("PROMPTIC_WORKSPACE_ID")

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
            if self.workspace_id:
                auth_headers["X-Workspace-Id"] = self.workspace_id
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

    def get_stats(self, *, days_back: int = 30) -> TracingStats:
        """Get aggregated tracing stats."""
        return self._get("/traces/stats", params={"days_back": days_back})

    # ── Workspace ────────────────────────────────────────────────────

    def get_workspace(self) -> Workspace:
        """Get workspace info for the current API key."""
        return self._get("/workspace")

    # ── Components ───────────────────────────────────────────────────

    def list_components(self) -> ComponentList:
        """List all AI components in the workspace."""
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
        task_type: str = "classification",
        initial_prompt: str | None = None,
        name: str | None = None,
        description: str | None = None,
        provider: str = "openai",
        optimizer: str = "prompticV2",
        hyperparameters: dict[str, Any] | None = None,
        initial_prediction_model_schema: dict[str, Any] | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        body: dict[str, Any] = {
            "aiComponentId": ai_component_id,
            "targetModel": target_model,
            "taskType": task_type,
            "provider": provider,
            "optimizer": optimizer,
        }
        if initial_prompt is not None:
            body["initialPrompt"] = initial_prompt
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if hyperparameters is not None:
            body["hyperparameters"] = hyperparameters
        if initial_prediction_model_schema is not None:
            body["initialPredictionModelSchema"] = initial_prediction_model_schema
        return self._post("/experiments", json=body)

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID."""
        return self._get(f"/experiments/{experiment_id}")

    def update_experiment(self, experiment_id: str, **updates: Any) -> Experiment:
        """Update a pending experiment.

        Accepts keyword arguments matching experiment fields:
        name, description, targetModel, provider, taskType, optimizer,
        initialPrompt, hyperparameters, initialPredictionModelSchema.
        """
        return self._patch(f"/experiments/{experiment_id}", json=updates)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        self._delete(f"/experiments/{experiment_id}")

    def start_experiment(self, experiment_id: str) -> ExperimentStarted:
        """Start a pending experiment (enqueue for training)."""
        return self._post(f"/experiments/{experiment_id}/start")

    def duplicate_experiment(
        self,
        experiment_id: str,
        *,
        continue_from_optimized: bool = False,
        initial_prompt_override: str | None = None,
    ) -> Experiment:
        """Duplicate an experiment (clones observations + evaluators).

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
            the workspace).
        """
        body: dict[str, Any] = {}
        if continue_from_optimized:
            body["continueFromOptimized"] = True
        if initial_prompt_override is not None:
            body["initialPromptOverride"] = initial_prompt_override
        return self._post(f"/experiments/{experiment_id}/duplicate", json=body)

    # ── Observations ─────────────────────────────────────────────────

    def list_observations(self, experiment_id: str) -> ObservationList:
        """List observations for an experiment."""
        return self._get(f"/experiments/{experiment_id}/observations")

    def create_observations(
        self, experiment_id: str, observations: list[dict[str, Any]]
    ) -> ObservationList:
        """Create observations for an experiment (batch)."""
        return self._post(
            f"/experiments/{experiment_id}/observations",
            json=_normalize_observation_payloads(observations),
        )

    def update_observation(
        self, experiment_id: str, observation_id: int, **data: Any
    ) -> Observation:
        """Update an observation."""
        return self._patch(
            f"/experiments/{experiment_id}/observations/{observation_id}",
            json=_normalize_observation_payload(data),
        )

    def delete_observation(self, experiment_id: str, observation_id: int) -> None:
        """Delete an observation."""
        self._delete(f"/experiments/{experiment_id}/observations/{observation_id}")

    # ── Evaluators ───────────────────────────────────────────────────

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

    def get_dataset(self, component_id: str, dataset_id: str) -> DatasetWithItems:
        """Get a dataset with its items."""
        return self._get(f"/components/{component_id}/datasets/{dataset_id}")

    def delete_dataset(self, component_id: str, dataset_id: str) -> None:
        """Delete a dataset."""
        self._delete(f"/components/{component_id}/datasets/{dataset_id}")

    # ── Runs ────────────────────────────────────────────────────────

    def create_run(
        self,
        component_id: str,
        dataset_id: str,
        *,
        name: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Run:
        """Create a run for a dataset."""
        body: dict[str, Any] = {"datasetId": dataset_id}
        if name is not None:
            body["name"] = name
        if trace_ids is not None:
            body["traceIds"] = trace_ids
        return self._post(f"/components/{component_id}/runs", json=body)

    def list_runs(self, component_id: str) -> RunList:
        """List runs for an AI component."""
        return self._get(f"/components/{component_id}/runs")

    def get_run(self, component_id: str, run_id: str) -> RunWithTraces:
        """Get a run with its traces."""
        return self._get(f"/components/{component_id}/runs/{run_id}")

    def delete_run(self, component_id: str, run_id: str) -> None:
        """Delete a run."""
        self._delete(f"/components/{component_id}/runs/{run_id}")

    # ── Annotations ─────────────────────────────────────────────────

    def upsert_annotation(
        self,
        component_id: str,
        run_id: str,
        trace_db_id: str,
        *,
        rating: str | None = None,
        comment: str | None = None,
    ) -> Annotation:
        """Create or update an annotation for a trace in a run."""
        body: dict[str, Any] = {"traceDbId": trace_db_id}
        if rating is not None:
            body["rating"] = rating
        if comment is not None:
            body["comment"] = comment
        return self._post(f"/components/{component_id}/runs/{run_id}/annotations", json=body)

    def list_annotations(self, component_id: str, run_id: str) -> AnnotationList:
        """List annotations for a run (includes all annotations across runs of the same dataset)."""
        return self._get(f"/components/{component_id}/runs/{run_id}/annotations")

    def list_dataset_annotations(self, component_id: str, dataset_id: str) -> AnnotationList:
        """List all annotations for traces across all runs of a dataset."""
        return self._get(f"/components/{component_id}/datasets/{dataset_id}/annotations")

    def delete_annotation(self, component_id: str, run_id: str, annotation_id: str) -> None:
        """Delete an annotation."""
        self._delete(f"/components/{component_id}/runs/{run_id}/annotations/{annotation_id}")

    # ── Agent Evaluations ────────────────────────────────────────────

    def create_evaluation(
        self,
        component_id: str,
        dataset_id: str,
        *,
        name: str | None = None,
        run_id: str | None = None,
    ) -> AgentEvaluation:
        """Start an evaluation on a dataset.

        Args:
            component_id: AI component ID.
            dataset_id: Dataset ID to evaluate.
            name: Optional evaluation name.
            run_id: Optional run ID to associate with.
        """
        body: dict[str, Any] = {"datasetId": dataset_id}
        if name is not None:
            body["name"] = name
        if run_id is not None:
            body["runId"] = run_id
        return self._post(f"/components/{component_id}/evaluations", json=body)

    def list_evaluations(self, component_id: str) -> AgentEvaluationList:
        """List evaluations for an AI component."""
        return self._get(f"/components/{component_id}/evaluations")

    def get_evaluation(self, component_id: str, evaluation_id: str) -> AgentEvaluation:
        """Get an evaluation with results."""
        return self._get(f"/components/{component_id}/evaluations/{evaluation_id}")

    def wait_for_evaluation(
        self,
        component_id: str,
        evaluation_id: str,
        *,
        max_wait: float = 300,
        poll_interval: float = 2,
    ) -> AgentEvaluation:
        """Poll an evaluation until it reaches a terminal status.

        Args:
            component_id: AI component ID.
            evaluation_id: Evaluation ID to poll.
            max_wait: Maximum seconds to wait before raising TimeoutError.
            poll_interval: Seconds between poll requests.

        Returns:
            The completed or failed evaluation.

        Raises:
            TimeoutError: If the evaluation does not finish within *max_wait* seconds.
        """
        deadline = time.monotonic() + max_wait
        while True:
            result = self.get_evaluation(component_id, evaluation_id)
            if result["status"] in ("completed", "failed"):
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Evaluation {evaluation_id} did not complete within {max_wait}s "
                    f"(last status: {result['status']})"
                )
            time.sleep(poll_interval)

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
        workspace_id: Workspace ID for session-based auth. Falls back to
            ``PROMPTIC_WORKSPACE_ID`` env var.
        endpoint: Promptic platform URL. Falls back to ``PROMPTIC_ENDPOINT`` env var,
            then to ``https://promptic.eu``.
        timeout: HTTP request timeout in seconds.
    """

    api_key: str | None = None
    access_token: str | None = None
    workspace_id: str | None = None
    endpoint: str | None = None
    timeout: float = 30.0
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the HTTP client."""
        self.api_key = self.api_key or os.environ.get("PROMPTIC_API_KEY")
        self.access_token = self.access_token or os.environ.get("PROMPTIC_ACCESS_TOKEN")
        self.workspace_id = self.workspace_id or os.environ.get("PROMPTIC_WORKSPACE_ID")

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
            if self.workspace_id:
                auth_headers["X-Workspace-Id"] = self.workspace_id
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

    async def get_stats(self, *, days_back: int = 30) -> TracingStats:
        """Get aggregated tracing stats."""
        return await self._get("/traces/stats", params={"days_back": days_back})

    # ── Workspace ────────────────────────────────────────────────────

    async def get_workspace(self) -> Workspace:
        """Get workspace info for the current API key."""
        return await self._get("/workspace")

    # ── Components ───────────────────────────────────────────────────

    async def list_components(self) -> ComponentList:
        """List all AI components in the workspace."""
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
        task_type: str = "classification",
        initial_prompt: str | None = None,
        name: str | None = None,
        description: str | None = None,
        provider: str = "openai",
        optimizer: str = "prompticV2",
        hyperparameters: dict[str, Any] | None = None,
        initial_prediction_model_schema: dict[str, Any] | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        body: dict[str, Any] = {
            "aiComponentId": ai_component_id,
            "targetModel": target_model,
            "taskType": task_type,
            "provider": provider,
            "optimizer": optimizer,
        }
        if initial_prompt is not None:
            body["initialPrompt"] = initial_prompt
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if hyperparameters is not None:
            body["hyperparameters"] = hyperparameters
        if initial_prediction_model_schema is not None:
            body["initialPredictionModelSchema"] = initial_prediction_model_schema
        return await self._post("/experiments", json=body)

    async def get_experiment(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID."""
        return await self._get(f"/experiments/{experiment_id}")

    async def update_experiment(self, experiment_id: str, **updates: Any) -> Experiment:
        """Update a pending experiment."""
        return await self._patch(f"/experiments/{experiment_id}", json=updates)

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        await self._delete(f"/experiments/{experiment_id}")

    async def start_experiment(self, experiment_id: str) -> ExperimentStarted:
        """Start a pending experiment (enqueue for training)."""
        return await self._post(f"/experiments/{experiment_id}/start")

    async def duplicate_experiment(
        self,
        experiment_id: str,
        *,
        continue_from_optimized: bool = False,
        initial_prompt_override: str | None = None,
    ) -> Experiment:
        """Duplicate an experiment (clones observations + evaluators).

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
            the workspace).
        """
        body: dict[str, Any] = {}
        if continue_from_optimized:
            body["continueFromOptimized"] = True
        if initial_prompt_override is not None:
            body["initialPromptOverride"] = initial_prompt_override
        return await self._post(f"/experiments/{experiment_id}/duplicate", json=body)

    # ── Observations ─────────────────────────────────────────────────

    async def list_observations(self, experiment_id: str) -> ObservationList:
        """List observations for an experiment."""
        return await self._get(f"/experiments/{experiment_id}/observations")

    async def create_observations(
        self, experiment_id: str, observations: list[dict[str, Any]]
    ) -> ObservationList:
        """Create observations for an experiment (batch)."""
        return await self._post(
            f"/experiments/{experiment_id}/observations",
            json=_normalize_observation_payloads(observations),
        )

    async def update_observation(
        self, experiment_id: str, observation_id: int, **data: Any
    ) -> Observation:
        """Update an observation."""
        return await self._patch(
            f"/experiments/{experiment_id}/observations/{observation_id}",
            json=_normalize_observation_payload(data),
        )

    async def delete_observation(self, experiment_id: str, observation_id: int) -> None:
        """Delete an observation."""
        await self._delete(f"/experiments/{experiment_id}/observations/{observation_id}")

    # ── Evaluators ───────────────────────────────────────────────────

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

    async def get_dataset(self, component_id: str, dataset_id: str) -> DatasetWithItems:
        """Get a dataset with its items."""
        return await self._get(f"/components/{component_id}/datasets/{dataset_id}")

    async def delete_dataset(self, component_id: str, dataset_id: str) -> None:
        """Delete a dataset."""
        await self._delete(f"/components/{component_id}/datasets/{dataset_id}")

    # ── Runs ────────────────────────────────────────────────────────

    async def create_run(
        self,
        component_id: str,
        dataset_id: str,
        *,
        name: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Run:
        """Create a run for a dataset."""
        body: dict[str, Any] = {"datasetId": dataset_id}
        if name is not None:
            body["name"] = name
        if trace_ids is not None:
            body["traceIds"] = trace_ids
        return await self._post(f"/components/{component_id}/runs", json=body)

    async def list_runs(self, component_id: str) -> RunList:
        """List runs for an AI component."""
        return await self._get(f"/components/{component_id}/runs")

    async def get_run(self, component_id: str, run_id: str) -> RunWithTraces:
        """Get a run with its traces."""
        return await self._get(f"/components/{component_id}/runs/{run_id}")

    async def delete_run(self, component_id: str, run_id: str) -> None:
        """Delete a run."""
        await self._delete(f"/components/{component_id}/runs/{run_id}")

    # ── Annotations ─────────────────────────────────────────────────

    async def upsert_annotation(
        self,
        component_id: str,
        run_id: str,
        trace_db_id: str,
        *,
        rating: str | None = None,
        comment: str | None = None,
    ) -> Annotation:
        """Create or update an annotation for a trace in a run."""
        body: dict[str, Any] = {"traceDbId": trace_db_id}
        if rating is not None:
            body["rating"] = rating
        if comment is not None:
            body["comment"] = comment
        return await self._post(f"/components/{component_id}/runs/{run_id}/annotations", json=body)

    async def list_annotations(self, component_id: str, run_id: str) -> AnnotationList:
        """List annotations for a run (includes all annotations across runs of the same dataset)."""
        return await self._get(f"/components/{component_id}/runs/{run_id}/annotations")

    async def list_dataset_annotations(self, component_id: str, dataset_id: str) -> AnnotationList:
        """List all annotations for traces across all runs of a dataset."""
        return await self._get(f"/components/{component_id}/datasets/{dataset_id}/annotations")

    async def delete_annotation(self, component_id: str, run_id: str, annotation_id: str) -> None:
        """Delete an annotation."""
        await self._delete(f"/components/{component_id}/runs/{run_id}/annotations/{annotation_id}")

    # ── Agent Evaluations ────────────────────────────────────────────

    async def create_evaluation(
        self,
        component_id: str,
        dataset_id: str,
        *,
        name: str | None = None,
        run_id: str | None = None,
    ) -> AgentEvaluation:
        """Start an evaluation on a dataset."""
        body: dict[str, Any] = {"datasetId": dataset_id}
        if name is not None:
            body["name"] = name
        if run_id is not None:
            body["runId"] = run_id
        return await self._post(f"/components/{component_id}/evaluations", json=body)

    async def list_evaluations(self, component_id: str) -> AgentEvaluationList:
        """List evaluations for an AI component."""
        return await self._get(f"/components/{component_id}/evaluations")

    async def get_evaluation(self, component_id: str, evaluation_id: str) -> AgentEvaluation:
        """Get an evaluation with results."""
        return await self._get(f"/components/{component_id}/evaluations/{evaluation_id}")

    async def wait_for_evaluation(
        self,
        component_id: str,
        evaluation_id: str,
        *,
        max_wait: float = 300,
        poll_interval: float = 2,
    ) -> AgentEvaluation:
        """Poll an evaluation until it reaches a terminal status.

        Args:
            component_id: AI component ID.
            evaluation_id: Evaluation ID to poll.
            max_wait: Maximum seconds to wait before raising TimeoutError.
            poll_interval: Seconds between poll requests.

        Returns:
            The completed or failed evaluation.

        Raises:
            TimeoutError: If the evaluation does not finish within *max_wait* seconds.
        """
        import asyncio

        deadline = time.monotonic() + max_wait
        while True:
            result = await self.get_evaluation(component_id, evaluation_id)
            if result["status"] in ("completed", "failed"):
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Evaluation {evaluation_id} did not complete within {max_wait}s "
                    f"(last status: {result['status']})"
                )
            await asyncio.sleep(poll_interval)

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
