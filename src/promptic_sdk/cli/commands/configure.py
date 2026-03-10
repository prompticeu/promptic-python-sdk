"""Configure command — save API key and endpoint."""

from __future__ import annotations

import typer
from rich.console import Console

from promptic_sdk.cli.config import get_config_path, save_config

console = Console()


def configure(
    api_key: str = typer.Option(
        ..., prompt="Promptic API key", help="Your Promptic API key (pk_...)."
    ),
    endpoint: str = typer.Option(
        "https://promptic.eu",
        prompt="Promptic endpoint",
        help="Promptic platform URL.",
    ),
) -> None:
    """Save Promptic API key and endpoint to ~/.promptic/config.toml."""
    save_config(api_key, endpoint)
    console.print(f"Configuration saved to {get_config_path()}", style="green")
