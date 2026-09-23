"""HTTP transports shared by the Agent Gym sync and async clients."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import httpx

from promptic_sdk.client import PrompticAPIError

_DEFAULT_ENDPOINT = "https://promptic.eu"

RequestParams: TypeAlias = dict[str, Any] | list[tuple[str, Any]]
RequestFiles: TypeAlias = Sequence[tuple[str, tuple[str, bytes, str]]]


@dataclass(frozen=True)
class RequestSpec:
    """Transport-independent API request."""

    method: str
    path: str
    params: RequestParams | None = None
    json: Any = None
    data: Mapping[str, Any] | None = None
    files: RequestFiles | None = field(default=None, repr=False)
    headers: Mapping[str, str] | None = None


class AgentGymAPIError(PrompticAPIError):
    """Structured error returned by an Agent Gym endpoint."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an API error without retaining sensitive request data."""
        self.code = code
        self.details = details
        rendered = message or code
        super().__init__(status_code, f"{code}: {rendered}" if message else rendered)
        self.message = rendered


class ArtifactTransferError(Exception):
    """A credential-free direct storage transfer failed."""

    def __init__(self, status_code: int | None, message: str) -> None:
        """Initialize without retaining the signed storage URL."""
        self.status_code = status_code
        self.message = message
        prefix = f"[{status_code}] " if status_code is not None else ""
        super().__init__(f"{prefix}{message}")


def _api_error(response: httpx.Response) -> AgentGymAPIError:
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        raw_code = body.get("error")
        code = (
            raw_code if isinstance(raw_code, str) and raw_code else f"http_{response.status_code}"
        )
        raw_message = body.get("message")
        message = raw_message if isinstance(raw_message, str) and raw_message else None
        raw_details = body.get("details")
        details = raw_details if isinstance(raw_details, dict) else None
        if details is None and code == "runs_not_comparable":
            comparable = body.get("comparable_context")
            details = {"comparable_context": comparable} if isinstance(comparable, dict) else None
        if details is None:
            additional = {
                key: value for key, value in body.items() if key not in {"error", "message"}
            }
            details = additional or None
        return AgentGymAPIError(response.status_code, code, message, details)
    text = response.text.strip()
    return AgentGymAPIError(
        response.status_code,
        f"http_{response.status_code}",
        text or response.reason_phrase or "Request failed",
    )


@dataclass(frozen=True)
class ResolvedClientConfig:
    """Resolved endpoint and authentication headers."""

    endpoint: str
    headers: dict[str, str] = field(repr=False)


def resolve_client_config(
    *,
    api_key: str | None,
    access_token: str | None,
    ai_application_id: str | None,
    endpoint: str | None,
) -> ResolvedClientConfig:
    """Resolve explicit, environment, and saved CLI authentication."""
    if api_key or access_token:
        resolved_api_key = api_key
        resolved_access_token = access_token
    else:
        resolved_api_key = os.environ.get("PROMPTIC_API_KEY")
        resolved_access_token = os.environ.get("PROMPTIC_ACCESS_TOKEN")
    resolved_application_id = (
        ai_application_id
        or os.environ.get("PROMPTIC_AI_APPLICATION_ID")
        or os.environ.get("PROMPTIC_WORKSPACE_ID")
    )
    resolved_endpoint = endpoint or os.environ.get("PROMPTIC_ENDPOINT")

    if not resolved_api_key and not resolved_access_token:
        try:
            from promptic_sdk.cli.config import load_config

            config = load_config()
        except Exception:  # noqa: BLE001
            config = None
        if config is not None:
            resolved_api_key = config.api_key
            resolved_access_token = config.access_token
            resolved_application_id = resolved_application_id or config.workspace_id
            resolved_endpoint = resolved_endpoint or config.endpoint

    token = resolved_access_token or resolved_api_key
    if not token:
        raise ValueError(
            "Promptic authentication required. Run 'promptic login', pass api_key= or "
            "access_token=, or set PROMPTIC_API_KEY / PROMPTIC_ACCESS_TOKEN."
        )
    headers = {"Authorization": f"Bearer {token}"}
    if resolved_access_token:
        if not resolved_application_id:
            raise ValueError(
                "ai_application_id is required with login access tokens. Run 'promptic login' "
                "to select an AI Application or pass ai_application_id=."
            )
        headers["X-AI-Application-Id"] = resolved_application_id
    return ResolvedClientConfig(
        endpoint=(resolved_endpoint or _DEFAULT_ENDPOINT).rstrip("/"),
        headers=headers,
    )


class SyncTransport:
    """Synchronous authenticated API and credential-free storage transport."""

    def __init__(self, config: ResolvedClientConfig, timeout: float) -> None:
        self.api = httpx.Client(
            base_url=f"{config.endpoint}/api/v1",
            headers=config.headers,
            timeout=timeout,
        )
        self.direct = httpx.Client(timeout=timeout, follow_redirects=True)

    def request(self, spec: RequestSpec) -> Any:
        """Execute a shared request specification."""
        response = self.api.request(
            spec.method,
            spec.path,
            params=spec.params,
            json=spec.json,
            data=spec.data,
            files=spec.files,
            headers=spec.headers,
        )
        if response.status_code >= 400:
            raise _api_error(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def request_multipart(
        self,
        path: str,
        *,
        data: Mapping[str, str],
        files: list[tuple[str, tuple[str, bytes, str]]],
        params: RequestParams | None = None,
    ) -> Any:
        """Execute an authenticated multipart API request."""
        response = self.api.post(path, params=params, data=data, files=files)
        if response.status_code >= 400:
            raise _api_error(response)
        return response.json()

    def close(self) -> None:
        """Close both HTTP clients."""
        self.api.close()
        self.direct.close()


class AsyncTransport:
    """Asynchronous authenticated API and credential-free storage transport."""

    def __init__(self, config: ResolvedClientConfig, timeout: float) -> None:
        self.api = httpx.AsyncClient(
            base_url=f"{config.endpoint}/api/v1",
            headers=config.headers,
            timeout=timeout,
        )
        self.direct = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def request(self, spec: RequestSpec) -> Any:
        """Execute a shared request specification."""
        response = await self.api.request(
            spec.method,
            spec.path,
            params=spec.params,
            json=spec.json,
            data=spec.data,
            files=spec.files,
            headers=spec.headers,
        )
        if response.status_code >= 400:
            raise _api_error(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def request_multipart(
        self,
        path: str,
        *,
        data: Mapping[str, str],
        files: list[tuple[str, tuple[str, bytes, str]]],
        params: RequestParams | None = None,
    ) -> Any:
        """Execute an authenticated multipart API request."""
        response = await self.api.post(path, params=params, data=data, files=files)
        if response.status_code >= 400:
            raise _api_error(response)
        return response.json()

    async def close(self) -> None:
        """Close both HTTP clients."""
        await self.api.aclose()
        await self.direct.aclose()
