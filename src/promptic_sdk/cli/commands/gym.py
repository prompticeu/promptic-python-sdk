"""Agent Gym benchmark runner commands."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from promptic_sdk.agent_gym import AgentGymClient
from promptic_sdk.agent_gym_runner import AgentGymCase, Candidate

gym_app = typer.Typer(help="Run trusted local candidates against Agent Gym benchmarks.")
console = Console()


def _load_candidate(reference: str) -> Candidate:
    module_name, separator, attribute_path = reference.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("candidate must use the format 'python.module:function'")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise ValueError(f"{reference} does not resolve to a callable")
    return cast(Callable[[AgentGymCase], Any], value)


@gym_app.command("run")
def run_candidate(
    benchmark_id: str = typer.Argument(help="Agent Gym benchmark UUID."),
    candidate: str = typer.Argument(help="Candidate callback as 'python.module:function'."),
    name: str = typer.Option(..., help="Leaderboard candidate name."),
    version: str = typer.Option(..., help="Leaderboard candidate version."),
    architecture_description: str | None = typer.Option(
        None, help="Short description of the candidate architecture."
    ),
    revision_id: str | None = typer.Option(None, help="Optional published revision UUID."),
    workdir: Path | None = typer.Option(
        None, help="Directory for materialized inputs and outputs."
    ),
    idempotency_key: str | None = typer.Option(None, help="Stable retry key for this execution."),
    no_wait: bool = typer.Option(False, help="Return after scoring is queued."),
    output_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Run a trusted Python callback and submit its predictions and artifacts."""
    callback = _load_candidate(candidate)
    with AgentGymClient() as client:
        result = client.submit(
            benchmark_id,
            callback,
            name=name,
            version=version,
            architecture_description=architecture_description,
            revision_id=revision_id,
            workdir=workdir,
            idempotency_key=idempotency_key,
            wait=not no_wait,
        )

    run = result.status["run"]
    payload = {
        "submission_id": result.submission_id,
        "revision_id": result.revision_id,
        "run_id": result.run_id,
        "bundle_id": result.bundle_id,
        "status": result.status["status"],
        "scoring_status": run["scoring_status"] if run else None,
        "eligibility_status": run["eligibility_status"] if run else None,
    }
    if output_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    console.print(f"Run [bold]{result.run_id}[/bold] submitted.")
    console.print(f"Submission: {result.status['status']}")
    if run:
        console.print(f"Scoring: {run['scoring_status']}")
        console.print(f"Leaderboard eligibility: {run['eligibility_status']}")
