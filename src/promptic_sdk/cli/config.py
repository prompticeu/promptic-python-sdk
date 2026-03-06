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

    api_key: str
    endpoint: str


def get_config_path() -> Path:
    """Return the path to the config file."""
    return _CONFIG_FILE


def load_config() -> CliConfig | None:
    """Load config from environment variables and config file.

    Priority: env vars > config file.

    Returns:
        CliConfig if a valid api_key is found, None otherwise.
    """
    api_key = os.environ.get("PROMPTIC_API_KEY")
    endpoint = os.environ.get("PROMPTIC_ENDPOINT")

    # Try loading from config file if env vars are missing
    if not api_key or not endpoint:
        file_config = _read_config_file()
        if not api_key:
            api_key = file_config.get("api_key")
        if not endpoint:
            endpoint = file_config.get("endpoint")

    if not api_key:
        return None

    return CliConfig(
        api_key=api_key,
        endpoint=endpoint or "https://app.promptic.eu",
    )


def save_config(api_key: str, endpoint: str) -> None:
    """Save configuration to the config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    content = f'api_key = "{api_key}"\nendpoint = "{endpoint}"\n'
    _CONFIG_FILE.write_text(content)
    # Restrict permissions to owner only
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
