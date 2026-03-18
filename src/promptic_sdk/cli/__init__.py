"""Promptic CLI package."""

from __future__ import annotations

import typer
from rich.console import Console

from promptic_sdk.cli.config import load_config
from promptic_sdk.client import PrompticClient

_err_console = Console(stderr=True)


def get_client() -> PrompticClient:
    """Create an authenticated client from CLI config."""
    config = load_config()
    if not config:
        _err_console.print(
            "No configuration found. Run 'promptic login' or 'promptic configure'.",
            style="red",
        )
        raise typer.Exit(1)
    return PrompticClient(
        api_key=config.api_key,
        access_token=config.access_token,
        workspace_id=config.workspace_id,
        endpoint=config.endpoint,
    )
