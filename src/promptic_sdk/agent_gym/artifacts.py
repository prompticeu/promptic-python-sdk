"""Safe Agent Gym artifact upload, materialization, and download helpers."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from urllib.parse import urljoin

import httpx

from promptic_sdk.agent_gym._http import (
    ArtifactTransferError,
    AsyncTransport,
    SyncTransport,
    _api_error,
)
from promptic_sdk.agent_gym.models import (
    ManifestInputFile,
    PredictionArtifact,
    PresignedUpload,
    ReservedSubmissionArtifact,
)
from promptic_sdk.agent_gym.submissions import MAX_ARTIFACT_BYTES, ArtifactIntegrityError

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()


def read_artifact(path: Path) -> bytes:
    """Read one bounded local output artifact."""
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact files cannot exceed {MAX_ARTIFACT_BYTES} bytes")
    return path.read_bytes()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically write bytes without truncating an existing target on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def existing_file_matches(path: Path, expected_size: int, expected_sha256: str | None) -> bool:
    """Return whether an existing regular file has the declared integrity."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
        return False
    if expected_sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


class _DownloadState:
    def __init__(
        self, expected_size: int | None, expected_sha256: str | None, max_bytes: int
    ) -> None:
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.max_bytes = max_bytes
        self.observed_size = 0
        self.digest = hashlib.sha256()

    def preflight(self, response: httpx.Response) -> None:
        raw_length = response.headers.get("Content-Length")
        if raw_length and raw_length.isdigit():
            length = int(raw_length)
            if length > self.max_bytes:
                raise ArtifactIntegrityError("artifact exceeds the configured download limit")
            if self.expected_size is not None and length != self.expected_size:
                raise ArtifactIntegrityError("artifact size does not match its declaration")

    def observe(self, chunk: bytes) -> None:
        self.observed_size += len(chunk)
        if self.observed_size > self.max_bytes:
            raise ArtifactIntegrityError("artifact exceeds the configured download limit")
        if self.expected_size is not None and self.observed_size > self.expected_size:
            raise ArtifactIntegrityError("artifact exceeds its declared size")
        self.digest.update(chunk)

    def verify(self) -> str:
        if self.expected_size is not None and self.observed_size != self.expected_size:
            raise ArtifactIntegrityError("artifact size does not match its declaration")
        digest = self.digest.hexdigest()
        if self.expected_sha256 is not None and digest != self.expected_sha256:
            raise ArtifactIntegrityError("artifact digest does not match its declaration")
        return digest


def _prepare_destination(destination: Path, overwrite: bool) -> None:
    if destination.is_symlink():
        raise ValueError("refusing to write through a symbolic-link destination")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing file: {destination}")
    if destination.exists() and not destination.is_file():
        raise ValueError("artifact destination must be a file path")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _replace_file(temporary_path: Path, destination: Path) -> None:
    temporary_path.replace(destination)


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _manifest_destination_ready(
    destination: Path, input_file: ManifestInputFile, overwrite: bool
) -> bool:
    if destination.exists() and not overwrite:
        if existing_file_matches(destination, input_file["size_bytes"], input_file["sha256"]):
            return False
        raise FileExistsError(f"refusing to replace existing file: {destination}")
    _prepare_destination(destination, overwrite=True)
    return True


def _redirect_url(response: httpx.Response) -> str | None:
    if not response.is_redirect:
        return None
    location = response.headers.get("Location")
    if not location:
        raise ArtifactTransferError(response.status_code, "Artifact redirect was missing Location")
    return urljoin(str(response.request.url), location)


@contextmanager
def _sync_prediction_stream(transport: SyncTransport, path: str) -> Iterator[httpx.Response]:
    with transport.api.stream("GET", path, follow_redirects=False) as response:
        if response.status_code >= 400:
            raise _api_error(response)
        redirect = _redirect_url(response)
        if redirect is None:
            yield response
            return
    try:
        with transport.direct.stream("GET", redirect) as direct_response:
            if direct_response.status_code >= 400:
                raise ArtifactTransferError(
                    direct_response.status_code, "Prediction artifact download failed"
                )
            yield direct_response
    except httpx.HTTPError:
        raise ArtifactTransferError(None, "Prediction artifact download failed") from None


@asynccontextmanager
async def _async_prediction_stream(
    transport: AsyncTransport, path: str
) -> AsyncIterator[httpx.Response]:
    async with transport.api.stream("GET", path, follow_redirects=False) as response:
        if response.status_code >= 400:
            raise _api_error(response)
        redirect = _redirect_url(response)
        if redirect is None:
            yield response
            return
    try:
        async with transport.direct.stream("GET", redirect) as direct_response:
            if direct_response.status_code >= 400:
                raise ArtifactTransferError(
                    direct_response.status_code, "Prediction artifact download failed"
                )
            yield direct_response
    except httpx.HTTPError:
        raise ArtifactTransferError(None, "Prediction artifact download failed") from None


def download_prediction_artifact_sync(
    transport: SyncTransport,
    artifact: PredictionArtifact,
    api_path: str,
    destination: Path,
    *,
    max_bytes: int,
    overwrite: bool,
) -> str:
    """Download a prediction artifact with bounded atomic integrity checks."""
    if not 1 <= max_bytes <= MAX_ARTIFACT_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_ARTIFACT_BYTES}")
    _prepare_destination(destination, overwrite)
    expected_size = artifact.get("size_bytes")
    expected_sha256 = artifact.get("sha256")
    state = _DownloadState(expected_size, expected_sha256, max_bytes)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with (
            os.fdopen(fd, "wb") as output,
            _sync_prediction_stream(transport, api_path) as response,
        ):
            state.preflight(response)
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                state.observe(chunk)
                output.write(chunk)
            digest = state.verify()
            output.flush()
            os.fsync(output.fileno())
        _replace_file(temporary_path, destination)
        return digest
    finally:
        _remove_file(temporary_path)


async def download_prediction_artifact_async(
    transport: AsyncTransport,
    artifact: PredictionArtifact,
    api_path: str,
    destination: Path,
    *,
    max_bytes: int,
    overwrite: bool,
) -> str:
    """Asynchronously download a bounded, atomically written prediction artifact."""
    if not 1 <= max_bytes <= MAX_ARTIFACT_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_ARTIFACT_BYTES}")
    _prepare_destination(destination, overwrite)
    state = _DownloadState(artifact.get("size_bytes"), artifact.get("sha256"), max_bytes)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            async with _async_prediction_stream(transport, api_path) as response:
                state.preflight(response)
                async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                    state.observe(chunk)
                    output.write(chunk)
            digest = state.verify()
            output.flush()
            os.fsync(output.fileno())
        await asyncio.to_thread(_replace_file, temporary_path, destination)
        return digest
    finally:
        await asyncio.to_thread(_remove_file, temporary_path)


def download_manifest_input_sync(
    transport: SyncTransport,
    input_file: ManifestInputFile,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Materialize one signed manifest input safely."""
    if not _manifest_destination_ready(destination, input_file, overwrite):
        return
    state = _DownloadState(input_file["size_bytes"], input_file["sha256"], input_file["size_bytes"])
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            try:
                with transport.direct.stream("GET", input_file["download_url"]) as response:
                    if response.status_code >= 400:
                        raise ArtifactTransferError(
                            response.status_code, "Manifest input download failed"
                        )
                    state.preflight(response)
                    for chunk in response.iter_bytes(_CHUNK_SIZE):
                        state.observe(chunk)
                        output.write(chunk)
            except httpx.HTTPError:
                raise ArtifactTransferError(None, "Manifest input download failed") from None
            state.verify()
            output.flush()
            os.fsync(output.fileno())
        _replace_file(temporary_path, destination)
    finally:
        _remove_file(temporary_path)


async def download_manifest_input_async(
    transport: AsyncTransport,
    input_file: ManifestInputFile,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Asynchronously materialize one signed manifest input safely."""
    if not await asyncio.to_thread(_manifest_destination_ready, destination, input_file, overwrite):
        return
    state = _DownloadState(input_file["size_bytes"], input_file["sha256"], input_file["size_bytes"])
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            try:
                async with transport.direct.stream("GET", input_file["download_url"]) as response:
                    if response.status_code >= 400:
                        raise ArtifactTransferError(
                            response.status_code, "Manifest input download failed"
                        )
                    state.preflight(response)
                    async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                        state.observe(chunk)
                        output.write(chunk)
            except httpx.HTTPError:
                raise ArtifactTransferError(None, "Manifest input download failed") from None
            state.verify()
            output.flush()
            os.fsync(output.fileno())
        await asyncio.to_thread(_replace_file, temporary_path, destination)
    finally:
        await asyncio.to_thread(_remove_file, temporary_path)


def upload_reserved_sync(
    transport: SyncTransport,
    reservation: ReservedSubmissionArtifact,
    content: bytes,
    *,
    mime_type: str,
    filename: str,
) -> None:
    """Upload exact bytes using a credential-free reservation descriptor."""
    upload_direct_sync(
        transport,
        reservation["upload"],
        content,
        mime_type=mime_type,
        filename=filename,
    )


def upload_direct_sync(
    transport: SyncTransport,
    upload: PresignedUpload,
    content: bytes,
    *,
    mime_type: str,
    filename: str,
) -> None:
    """Upload exact bytes using a credential-free presigned descriptor."""
    if len(content) > upload["maxSizeBytes"]:
        raise ValueError("artifact content exceeds the reserved upload size")
    try:
        if upload["method"] == "PUT":
            response = transport.direct.put(
                upload["uploadUrl"], content=content, headers=upload.get("headers", {})
            )
        else:
            response = transport.direct.post(
                upload["uploadUrl"],
                data=upload.get("fields", {}),
                files={"file": (filename, content, mime_type)},
                headers=upload.get("headers", {}),
            )
    except httpx.HTTPError:
        raise ArtifactTransferError(None, "Submission artifact upload failed") from None
    if response.status_code >= 400:
        raise ArtifactTransferError(response.status_code, "Submission artifact upload failed")


async def upload_reserved_async(
    transport: AsyncTransport,
    reservation: ReservedSubmissionArtifact,
    content: bytes,
    *,
    mime_type: str,
    filename: str,
) -> None:
    """Asynchronously upload exact bytes using a reservation descriptor."""
    await upload_direct_async(
        transport,
        reservation["upload"],
        content,
        mime_type=mime_type,
        filename=filename,
    )


async def upload_direct_async(
    transport: AsyncTransport,
    upload: PresignedUpload,
    content: bytes,
    *,
    mime_type: str,
    filename: str,
) -> None:
    """Asynchronously upload exact bytes using a presigned descriptor."""
    if len(content) > upload["maxSizeBytes"]:
        raise ValueError("artifact content exceeds the reserved upload size")
    try:
        if upload["method"] == "PUT":
            response = await transport.direct.put(
                upload["uploadUrl"], content=content, headers=upload.get("headers", {})
            )
        else:
            response = await transport.direct.post(
                upload["uploadUrl"],
                data=upload.get("fields", {}),
                files={"file": (filename, content, mime_type)},
                headers=upload.get("headers", {}),
            )
    except httpx.HTTPError:
        raise ArtifactTransferError(None, "Submission artifact upload failed") from None
    if response.status_code >= 400:
        raise ArtifactTransferError(response.status_code, "Submission artifact upload failed")
