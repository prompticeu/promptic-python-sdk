"""Advanced low-level runner using an optional benchmark-scoped runner key.

Required environment variables:
    PROMPTIC_AGENT_GYM_TOKEN: Benchmark-scoped ``ags_`` runner credential.
    PROMPTIC_BENCHMARK_ID: Benchmark dataset UUID.

Optional environment variables:
    PROMPTIC_ENDPOINT: Platform URL.
    CANDIDATE_VERSION: Candidate bundle version (default: ``local-1``).
"""

from __future__ import annotations

import html
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from opentelemetry import trace

import promptic_sdk
from promptic_sdk import AgentGymClient, ExternalPrediction, ManifestCase


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example")
    return value


def build_report(case: ManifestCase, input_files: list[Path], destination: Path) -> Path:
    """Replace this function with the candidate agent under evaluation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = "".join(f"<li>{html.escape(path.name)}</li>" for path in input_files)
    payload = html.escape(json.dumps(case["input_payload"], indent=2))
    destination.write_text(
        "<!doctype html><html><body>"
        f"<h1>Case {case['ordinal']}</h1>"
        f"<pre>{payload}</pre><ul>{inputs}</ul>"
        "</body></html>",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Execute, upload, finalize, and poll one external benchmark submission."""
    token = _required_env("PROMPTIC_AGENT_GYM_TOKEN")
    benchmark_id = _required_env("PROMPTIC_BENCHMARK_ID")
    endpoint = os.environ.get("PROMPTIC_ENDPOINT", "https://promptic.eu")
    run_key = f"local-{uuid4()}"
    workdir = Path(".promptic-agent-gym") / run_key

    # The scoped runner key can ingest OTLP traces when it has trace:write.
    # Do not call promptic_sdk.artifact() for prediction outputs; use the
    # submission-specific artifact methods below.
    promptic_sdk.init(
        api_key=token,
        endpoint=endpoint,
        service_name="agent-gym-external-runner",
    )
    tracer = trace.get_tracer(__name__)

    with AgentGymClient(submission_token=token, endpoint=endpoint) as client:
        session = client.start_submission(
            benchmark_id,
            idempotency_key=f"{run_key}:create",
        )
        materialized = session.materialize_manifest(workdir / "inputs")
        files_by_case: dict[str, list[Path]] = {}
        for item in materialized.files:
            files_by_case.setdefault(item.case_id, []).append(item.local_path)

        pending: list[tuple[ExternalPrediction, str]] = []
        for case in materialized.manifest["data"]:
            started = datetime.now(UTC)
            started_clock = time.monotonic()
            with tracer.start_as_current_span("agent_gym.case") as span:
                span.set_attribute("agent_gym.revision_case_id", case["case_id"])
                span.set_attribute("agent_gym.case_ordinal", case["ordinal"])
                raw_otel_trace_id = f"{span.get_span_context().trace_id:032x}"

                report_path = build_report(
                    case,
                    files_by_case.get(case["case_id"], []),
                    workdir / "outputs" / case["case_id"] / "report.html",
                )
                artifact = session.upload_artifact_file(
                    report_path,
                    path=f"cases/{case['case_id']}/report.html",
                    mime_type="text/html",
                )

            prediction: ExternalPrediction = {
                # Manifest case_id is the revision_case_id required by finalize.
                "revision_case_id": case["case_id"],
                "status": "succeeded",
                "output": {
                    "kind": "artifact",
                    "value": {"primary_path": artifact["path"]},
                },
                "artifact_ids": [artifact["artifact_id"]],
                "executor_id": "local-python",
                "executor_version": os.environ.get("GIT_COMMIT", "working-tree"),
                "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                "latency_ms": int((time.monotonic() - started_clock) * 1000),
                "started_at": started.isoformat().replace("+00:00", "Z"),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "diagnostics": {"local_report": str(report_path)},
            }
            pending.append((prediction, raw_otel_trace_id))

        # BatchSpanProcessor exports asynchronously. Flush before resolving raw
        # 32-hex OTEL IDs to the trace database UUIDs accepted by finalize.
        force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if callable(force_flush):
            force_flush()

        predictions: list[ExternalPrediction] = []
        for prediction, raw_otel_trace_id in pending:
            prediction["execution_refs"] = {
                "trace_ids": session.wait_for_resolved_traces([raw_otel_trace_id])
            }
            predictions.append(prediction)

        finalized = session.finalize(
            bundle_identity={
                "name": "local-html-report-agent",
                "version": os.environ.get("CANDIDATE_VERSION", "local-1"),
                "architecture_description": (
                    "A local Python runner that consumes frozen manifest cases "
                    "and emits one HTML report artifact per case."
                ),
                "architecture_tags": ["external", "html-report"],
            },
            predictions=predictions,
            idempotency_key=f"{run_key}:finalize",
            metadata={"runner": "examples/agent_gym_external_submission.py"},
        )
        completed = session.wait()
        run = completed["run"]
        print(
            json.dumps(
                {
                    "submission_id": finalized["submission_id"],
                    "run_id": finalized["run_id"],
                    "submission_status": completed["status"],
                    "scoring_status": run["scoring_status"] if run else None,
                    "leaderboard_eligibility": (run["eligibility_status"] if run else None),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
