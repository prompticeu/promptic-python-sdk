# /// script
# requires-python = ">=3.11"
# dependencies = ["promptic-sdk"]
# ///
"""Submit a trusted local Agent Gym candidate and inspect scored evidence.

Run with:
    uv run --no-project --env-file .env examples/agent_gym_external_submission.py

Environment variables:
    PROMPTIC_API_KEY      - AI Application API key
    PROMPTIC_BENCHMARK_ID - Agent Gym benchmark UUID
    PROMPTIC_ENDPOINT     - (optional) defaults to https://promptic.eu
    AGENT_REPOSITORY_URL  - (optional) source repository URL shown on the variant
    AGENT_COMMIT_HASH     - (optional) source revision shown on the variant

The callback runs in this authenticated process. Do not use this pattern for
generated or otherwise untrusted agent code; isolate that code and let a
trusted runner use the lower-level submission session API.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import promptic_sdk
from promptic_sdk import AgentGymCase, AgentGymCaseResult, AgentGymClient, AgentGymOutputArtifact

BENCHMARK_ID = os.environ["PROMPTIC_BENCHMARK_ID"]
REPOSITORY_URL = os.environ.get("AGENT_REPOSITORY_URL")
COMMIT_HASH = os.environ.get("AGENT_COMMIT_HASH")
OUTPUT_ROOT = Path("agent-gym-output")

promptic_sdk.init()


def build_report(case: AgentGymCase) -> AgentGymCaseResult:
    """Build a simple HTML deliverable inside the SDK's case-root trace."""
    report = f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Agent Gym report</title></head>
  <body>
    <h1>{html.escape(case.task["name"])}</h1>
    <p>{html.escape(case.instructions)}</p>
    <pre>{html.escape(repr(case.input))}</pre>
  </body>
</html>
"""
    path = OUTPUT_ROOT / f"case-{case.ordinal:04d}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return AgentGymCaseResult.artifact(
        AgentGymOutputArtifact(path, field_path="report", mime_type="text/html")
    )


with AgentGymClient() as gym:
    submitted = gym.run_and_submit(
        benchmark_id=BENCHMARK_ID,
        executor=build_report,
        name="html-report-agent",
        version="1.0.0",
        architecture_description=(
            "Creates a self-contained HTML report from each immutable case, "
            "records the generation trace, and returns the report as the scored artifact."
        ),
        repository_url=REPOSITORY_URL,
        commit_hash=COMMIT_HASH,
        trace_cases=True,
        trace_policy="best_effort",
    )

    summary = gym.get_run_results(BENCHMARK_ID, submitted.run_id)
    weakest = gym.list_case_results(
        BENCHMARK_ID,
        submitted.run_id,
        sort="score",
        limit=5,
    )
    print("Aggregates:", summary["aggregates"])
    for case_result in weakest["data"]:
        print(case_result["case_id"], case_result["overall_score"])
        print("Judgements:", case_result["judgements"])
        print("Traces:", case_result["traces"])
        if case_result["artifacts"]:
            artifact = case_result["artifacts"][0]
            destination = (
                OUTPUT_ROOT
                / "review"
                / str(case_result["case_id"])
                / (artifact["path"] or "artifact.bin")
            )
            gym.download_prediction_artifact(artifact, destination)
            print("Downloaded:", destination)
