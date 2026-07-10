"""Configuration management for the Promptic CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_DIR = Path.home() / ".promptic"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


@dataclass
class CliConfig:
    """Resolved CLI configuration."""

    endpoint: str
    api_key: str | None = None
    access_token: str | None = None
    ai_application_id: str | None = None

    @property
    def workspace_id(self) -> str | None:
        """Deprecated alias for :attr:`ai_application_id`."""
        return self.ai_application_id


def get_config_path() -> Path:
    """Return the path to the config file."""
    return _CONFIG_FILE


def load_config() -> CliConfig | None:
    """Load config from environment variables and config file.

    Priority: env vars > config file.

    Returns:
        CliConfig if valid credentials are found, None otherwise.
    """
    api_key = os.environ.get("PROMPTIC_API_KEY")
    access_token = os.environ.get("PROMPTIC_ACCESS_TOKEN")
    endpoint = os.environ.get("PROMPTIC_ENDPOINT")
    # Accept the deprecated ``PROMPTIC_WORKSPACE_ID`` env var as a fallback.
    ai_application_id = os.environ.get("PROMPTIC_AI_APPLICATION_ID") or os.environ.get(
        "PROMPTIC_WORKSPACE_ID"
    )

    # Try loading from config file if env vars are missing
    file_config = _read_config_file()
    if not api_key:
        api_key = file_config.get("api_key")
    if not access_token:
        access_token = file_config.get("access_token")
    if not endpoint:
        endpoint = file_config.get("endpoint")
    if not ai_application_id:
        # ``ai_application_id`` is the current key; ``workspace_id`` is legacy.
        ai_application_id = file_config.get("ai_application_id") or file_config.get("workspace_id")

    if not api_key and not access_token:
        return None

    return CliConfig(
        endpoint=endpoint or "https://promptic.eu",
        api_key=api_key,
        access_token=access_token,
        ai_application_id=ai_application_id,
    )


def save_config(api_key: str, endpoint: str) -> None:
    """Save API key configuration to the config file."""
    file_config = _read_config_file()
    file_config["api_key"] = api_key
    file_config["endpoint"] = endpoint
    _write_config_file(file_config)


def save_token(access_token: str, endpoint: str) -> None:
    """Save access token from device auth login."""
    file_config = _read_config_file()
    file_config["access_token"] = access_token
    file_config["endpoint"] = endpoint
    _write_config_file(file_config)


def save_ai_application(ai_application_id: str) -> None:
    """Save selected AI Application ID."""
    file_config = _read_config_file()
    file_config["ai_application_id"] = ai_application_id
    # Drop the legacy key so the two never drift apart.
    file_config.pop("workspace_id", None)
    _write_config_file(file_config)


# Deprecated alias — kept for backward compatibility.
save_workspace = save_ai_application


def clear_token() -> None:
    """Clear access token and AI Application from config."""
    file_config = _read_config_file()
    file_config.pop("access_token", None)
    file_config.pop("ai_application_id", None)
    file_config.pop("workspace_id", None)
    _write_config_file(file_config)


def _write_config_file(config: dict[str, str]) -> None:
    """Write config dict to the TOML config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f'{key} = "{value}"' for key, value in config.items()]
    _CONFIG_FILE.write_text("\n".join(lines) + "\n")
    _CONFIG_FILE.chmod(0o600)


def _read_config_file() -> dict[str, str]:
    """Read the TOML config file, returning an empty dict if missing."""
    if not _CONFIG_FILE.exists():
        return {}

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    try:
        with open(_CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}
