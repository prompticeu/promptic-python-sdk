"""Promptic CLI entrypoint."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from promptic_sdk.cli.commands.annotations import annotations_app
from promptic_sdk.cli.commands.components import components_app
from promptic_sdk.cli.commands.configure import configure
from promptic_sdk.cli.commands.datasets import datasets_app
from promptic_sdk.cli.commands.deployments import deployments_app
from promptic_sdk.cli.commands.evaluations import evaluations_app
from promptic_sdk.cli.commands.evaluators import evaluators_app
from promptic_sdk.cli.commands.experiments import experiments_app
from promptic_sdk.cli.commands.iterations import iterations_app
from promptic_sdk.cli.commands.login import login, logout
from promptic_sdk.cli.commands.observations import observations_app
from promptic_sdk.cli.commands.runs import runs_app
from promptic_sdk.cli.commands.traces import traces_app
from promptic_sdk.cli.commands.workspace import workspace_app

app = typer.Typer(
    name="promptic",
    help="Promptic CLI — interact with the Promptic platform.",
    no_args_is_help=True,
)

app.command("login")(login)
app.command("logout")(logout)
app.command("configure")(configure)
app.add_typer(traces_app, name="traces")
app.add_typer(workspace_app, name="workspace")
app.add_typer(components_app, name="components")
app.add_typer(experiments_app, name="experiments")
app.add_typer(observations_app, name="observations")
app.add_typer(evaluators_app, name="evaluators")
app.add_typer(iterations_app, name="iterations")
app.add_typer(deployments_app, name="deployments")
app.add_typer(datasets_app, name="datasets")
app.add_typer(runs_app, name="runs")
app.add_typer(annotations_app, name="annotations")
app.add_typer(evaluations_app, name="evaluations")

_err_console = Console(stderr=True)


def main() -> None:
    """Run the CLI with clean error handling."""
    try:
        app()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        from promptic_sdk.client import PrompticAPIError

        if isinstance(exc, PrompticAPIError):
            _err_console.print(f"[red]Error ({exc.status_code}):[/red] {exc.message}")
        else:
            _err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
