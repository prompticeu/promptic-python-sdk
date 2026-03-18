"""Login and logout commands — device authorization flow."""

from __future__ import annotations

import os
import time
import webbrowser

import httpx
import typer
from rich.console import Console

from promptic_sdk.cli.config import clear_token, get_config_path, save_token, save_workspace

console = Console()
err_console = Console(stderr=True)

CLIENT_ID = "promptic-cli"

_DEFAULT_ENDPOINT = "https://promptic.eu"


def _resolve_endpoint(explicit: str | None) -> str:
    """Resolve endpoint from explicit flag, env var, or default."""
    if explicit:
        return explicit
    return os.environ.get("PROMPTIC_ENDPOINT", _DEFAULT_ENDPOINT)


def login(
    endpoint: str | None = typer.Option(
        None,
        help="Promptic platform URL. Defaults to PROMPTIC_ENDPOINT env var or https://promptic.eu.",
    ),
) -> None:
    """Authenticate with the Promptic platform via browser login."""
    endpoint = _resolve_endpoint(endpoint)
    console.print(f"Connecting to [bold]{endpoint}[/bold]...", style="dim")

    try:
        resp = httpx.post(
            f"{endpoint.rstrip('/')}/api/auth/device/code",
            json={"client_id": CLIENT_ID, "scope": "openid profile email"},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        err_console.print(f"[red]Failed to request device code:[/red] {exc}")
        raise typer.Exit(1) from exc

    data = resp.json()
    device_code = data["device_code"]
    user_code = data["user_code"]
    verification_uri = data["verification_uri"]
    verification_uri_complete = data.get("verification_uri_complete")
    interval = data.get("interval", 5)

    console.print()
    console.print(f"Visit: [bold]{verification_uri}[/bold]")
    console.print(f"Enter code: [bold cyan]{user_code}[/bold cyan]")
    console.print()

    url_to_open = verification_uri_complete or verification_uri
    console.print("Opening browser...", style="dim")
    webbrowser.open(url_to_open)

    console.print(f"Waiting for authorization (polling every {interval}s)...", style="dim")

    while True:
        time.sleep(interval)

        try:
            token_resp = httpx.post(
                f"{endpoint.rstrip('/')}/api/auth/device/token",
                json={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": CLIENT_ID,
                },
                timeout=15,
            )
        except httpx.HTTPError:
            continue

        result = token_resp.json()

        if "access_token" in result:
            access_token = result["access_token"]
            save_token(access_token, endpoint)
            console.print()
            console.print("[green]Login successful![/green]")
            _auto_select_workspace(endpoint, access_token)
            console.print(f"Configuration saved to {get_config_path()}")
            return

        error = result.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
        elif error == "access_denied":
            err_console.print("[red]Access was denied.[/red]")
            raise typer.Exit(1)
        elif error == "expired_token":
            err_console.print("[red]Device code expired. Please try again.[/red]")
            raise typer.Exit(1)
        else:
            description = result.get("error_description", error)
            err_console.print(f"[red]Error:[/red] {description}")
            raise typer.Exit(1)


def _auto_select_workspace(endpoint: str, access_token: str) -> None:
    """Fetch workspaces and auto-select or prompt the user to pick one."""
    try:
        resp = httpx.get(
            f"{endpoint.rstrip('/')}/api/v1/workspaces",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        console.print(
            "Run [bold]promptic workspace list[/bold] to see your workspaces, "
            "then [bold]promptic workspace select <id>[/bold] to choose one.",
            style="dim",
        )
        return

    workspaces = resp.json().get("data", [])
    if not workspaces:
        console.print("No workspaces found.", style="dim")
        return

    if len(workspaces) == 1:
        ws = workspaces[0]
        save_workspace(ws["id"])
        console.print(f"Workspace [bold]{ws['name']}[/bold] selected automatically.")
        return

    # Multiple workspaces — let the user pick
    console.print()
    console.print("Select a workspace:")
    for i, ws in enumerate(workspaces, 1):
        console.print(f"  [bold]{i}[/bold]. {ws['name']} ({ws['id'][:8]}...)")

    console.print()
    while True:
        choice = console.input(f"Enter number [1-{len(workspaces)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(workspaces):
            ws = workspaces[int(choice) - 1]
            save_workspace(ws["id"])
            console.print(f"Workspace [bold]{ws['name']}[/bold] selected.")
            return
        console.print("Invalid choice, try again.", style="red")


def logout() -> None:
    """Clear saved login credentials."""
    clear_token()
    console.print("Logged out. Access token and workspace cleared.", style="green")
