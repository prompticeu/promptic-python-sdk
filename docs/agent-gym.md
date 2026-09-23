# Agent Gym SDK

`AgentGymClient` and `AsyncAgentGymClient` support the external-runtime-first Agent Gym workflow:

1. Author a benchmark, its scoring/judge contract, and private cases.
2. Explicitly publish an immutable revision once the benchmark draft is ready.
3. Create an idempotent submission session tied to that revision.
4. Page through and safely materialize its manifest and input files.
5. Produce one terminal prediction for every case.
6. Reserve, upload, and verify output artifacts through credential-free signed storage requests.
7. Resolve OpenTelemetry trace IDs and attach execution evidence.
8. Submit the complete prediction set for scoring, poll its status, and inspect aggregate and
   per-case results.

## Benchmark authoring

Authoring is available through the `gym.benchmarks` namespace. The SDK compiles typed schemas and
evaluators into the platform's canonical Agent component contract.

```python
from promptic_sdk import (
    AgentGymClient,
    BenchmarkCase,
    BenchmarkFile,
    EvidenceKind,
    EvidencePolicy,
    ExpectedBehaviorJudge,
    InvestigationBudget,
    VerifierAgent,
    VerifierMetric,
)

with AgentGymClient(ai_application_id="<ai-application-uuid>") as gym:
    benchmark = gym.benchmarks.create(
        name="Document Comparison",
        goal="Compare the supplied documents and produce an auditable report.",
        input_schema={
            "type": "object",
            "properties": {
                "characteristic_id": {"type": "string"},
                "files": {"type": "array", "x-promptic-type": "file"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {"report": {"type": "string"}},
        },
        evaluators=[
            VerifierAgent(
                instructions="Inspect the selected evidence and cite the evidence used.",
                model="<verifier-model>",
                metrics=(
                    VerifierMetric(
                        "correctness",
                        "Correctness",
                        "Check whether facts and values are correct.",
                        weight=2,
                        threshold=0.8,
                    ),
                    VerifierMetric(
                        "completeness", "Completeness", "Check whether required content is present."
                    ),
                ),
                evidence=EvidencePolicy(
                    selected=(
                        EvidenceKind.CASE_INPUT,
                        EvidenceKind.SUBMITTED_OUTPUT,
                        EvidenceKind.EXPECTED_BEHAVIOR,
                        EvidenceKind.EXPECTED_OUTPUT,
                        EvidenceKind.EXECUTION_TRACE,
                    ),
                ),
                budget=InvestigationBudget(max_steps=24),
            ),
            ExpectedBehaviorJudge(name="Expected behavior compliance"),
        ],
    )

    shared = BenchmarkFile("shared-instructions.pdf")
    result = benchmark.cases.add_many([
        BenchmarkCase(
            input={"characteristic_id": "F_26", "files": [shared]},
            output={"report": "Review section A."},
            expected_behavior="Explain all material differences with evidence.",
        ),
        BenchmarkCase(
            input={"characteristic_id": "F_27", "files": [shared]},
            output={"report": "Review section B."},
        ),
    ], batch_size=10)

    print(result["created"], result["draftCaseCount"])
    revision = benchmark.publish()
    print(revision.revision.id, revision.revision.case_count)
```

The verifier has explicit investigation instructions and a fixed contract of one to eight stable
metrics. Metric keys must be unique snake_case identifiers. Set `weight` and an optional passing
`threshold` directly on each `VerifierMetric`. The SDK translates those settings to the platform's
binding map. Verifier evaluators do not accept an evaluator-level weight because every metric is
scored and aggregated independently. The lower-level `metric_bindings` argument remains available
for compatibility, but do not configure the same metric through both interfaces.
`ExpectedBehaviorJudge` is a platform preset with the fixed `behavior_compliance` metric. Beyond
common metadata, callers may select a model and configure that metric through `metric_bindings`;
the server supplies its instructions, evidence, metric definition, and investigation budget.

Local input and reference files are uploaded through the platform's private bulk-import contract.
`add_many()` splits large inputs into confirmed batches, aggregates every per-case response, and
never publishes the draft. Repeated file content is uploaded once and reused across those batches.
Files can also be nested directly in schema-shaped `input` or `output` mappings.

`VerifierAgent` configuration describes the investigation semantically. Its five public evidence
kinds are case inputs, submitted output, expected behavior, expected output, and execution trace.
Submitted output includes submitted artifacts, while expected output includes private reference
files. The default selects every kind except execution trace. The public contract has no required
evidence setting. `InvestigationBudget.max_steps` defaults to 20 and accepts any integer of at least
one; higher values, including values above 20, round-trip unchanged.

The SDK sends instructions, the fixed metrics, selected evidence, an optional model, and a custom
step budget. The platform owns the verifier harness and runtime: shell and image-inspection
capabilities, model-provider setup, credentials, network policy, filesystem layout and paths,
process isolation, and concrete tool implementations are not configurable through
`VerifierAgent`. These semantic SDK settings describe what the verifier should assess and which
evidence it may use; they do not expose platform runtime controls.

Use `on_progress=` to receive a status dictionary after every confirmed batch. If a request ends
without confirmation, `BulkCaseUploadError.result` contains only the batches whose responses were
received. The SDK does not retry or resume an ambiguous request automatically.

`BenchmarkCase` is the in-memory authoring model. `cases.add()` returns a persisted case record
with its server-assigned integer dataset-case `id`; `cases.add_many()` returns the creation summary.

### Benchmark revisions and schema migrations

Published revisions remain immutable. `benchmark.revisions()` returns revision history plus the
active revision's case, execution, and evaluation fingerprints. If a schema change would invalidate
existing cases, `configure()` raises `AgentGymAPIError` with code `schema_migration_required` after
the platform safely persists a migration draft; the active revision remains usable.

Authoring changes always remain in a draft. `publish()` explicitly activates Version 1 or a fully
resolved later revision; adding cases or changing configuration never changes the active revision.

```python
from promptic_sdk import AgentGymAPIError

try:
    benchmark.configure(output_schema=next_output_schema)
except AgentGymAPIError as error:
    if error.code != "schema_migration_required":
        raise

draft = benchmark.revision_draft()
for conflict in draft["conflicts"] if draft else []:
    print(conflict["direction"], conflict["fieldPath"], conflict["affectedCaseCount"])

benchmark.resolve_revision_draft(
    "output.name",
    kind="default_value",
    value="unknown",
)
benchmark.publish()
```

Use `kind="make_optional"` to relax a proposed required field, or `kind="edit_cases"` when cases
will be reviewed explicitly. `abandon_revision_draft()` discards only the proposal and leaves the
active benchmark unchanged. After evaluator-only changes, `gym.reevaluate_run(benchmark_id,
run_id)` queues scoring for an execution-current variant; a run whose execution inputs changed must
be submitted again. The response includes an immutable `evaluation_run_id` for that scoring
operation. Retrying delivery of the same operation remains idempotent, while a deliberate later
`reevaluate_run()` call creates a new operation ID.

The CLI exposes benchmark workflows through `promptic agent-gym apply`, `status`, and `run`, and
the revision lifecycle through `revisions`, `draft`, `resolve-draft`, `publish-draft`,
`abandon-draft`, and `reevaluate`.

The Output schema and evaluators are configured explicitly. The schema defines valid output data;
the field-level evaluator defines which fields contribute and how they are scored:

```python
from promptic_sdk import FieldLevelJudge, FieldScoring

benchmark.configure(
    output_schema={
        "type": "object",
        "properties": {
            "acceptance_criteria_value": {"type": "string"},
            "test_method": {"type": "string"},
            "comment": {"type": "string"},
        },
    },
    evaluators=[
        FieldLevelJudge(
            fields={
                "acceptance_criteria_value": FieldScoring(method="exact"),
                "test_method": FieldScoring(method="embedding"),
                "comment": FieldScoring(method="exact", include=False),
            }
        )
    ],
)
```

`FieldScoring.method` accepts `exact`, `embedding`, `contains`, or `judge` for scalar
fields, and `array_exact`, `array_similarity`, or `array_judge` for array fields.
Use `judge_instructions` with `judge` or `array_judge`. Configure every field with
`method`; `strategy` and `array_strategy` are unsupported.

Cases use only canonical `input`, optional expected `output`, and optional `expected_behavior`.
Place `BenchmarkFile` values directly at file fields declared by the corresponding schema.
Across input and expected output, one case may contain at most 10 files, each no larger than
25,000,000 bytes and no larger than 100,000,000 bytes in total. The SDK validates these limits
before requesting uploads.

The authoring endpoints are currently protected by the platform's alpha admin gate. Both dashboard
sessions and API keys must belong to a platform admin. A non-admin receives `403 Forbidden`, mapped
to `AgentGymAPIError`. This SDK surface should be considered alpha until the platform publishes the
authoring schemas and introduces benchmark-author permissions. It also requires a Promptic
deployment containing the platform `agent-gym` API; those routes are not present on platform
`main` yet.

## Trusted callback API

`run_and_submit()` is the simplest interface for code you trust. The executor runs inside the
calling Python process and therefore has access to its files, network, model credentials, and
Promptic credentials. It is orchestration, not a sandbox.

```python
from pathlib import Path

from promptic_sdk import AgentGymCase, AgentGymCaseResult, AgentGymClient, AgentGymOutputArtifact


def execute_case(case: AgentGymCase) -> AgentGymCaseResult:
    output = Path(f"report-{case.ordinal}.html")
    output.write_text(render_report(case.input))
    return AgentGymCaseResult.artifact(
        AgentGymOutputArtifact(output, field_path="report")
    )


with AgentGymClient() as gym:
    submitted = gym.run_and_submit(
        benchmark_id="<benchmark-uuid>",
        executor=execute_case,
        name="report-agent",
        version="1.0.0",
        architecture_description="Renders and validates a standalone HTML report.",
        repository_url="https://github.com/acme/report-agent",
        commit_hash="6f1ed002ab5595859014ebf0951522d9d5f25a73",
        trace_cases=True,
        trace_policy="best_effort",
    )
```

Executor exceptions are recorded as failed case predictions by default so one broken case does not
discard the rest of the submission. Set `capture_exceptions=False` to fail fast.
After each executor callback returns, the SDK uploads that case's prediction immediately. The
idempotent upload is retried up to three times for transient transport, rate-limit, and server
failures. Session uploads are also split dynamically before either the 1 MiB serialized-body bound
or the API's 500-item bound is reached. A single result larger than that should be stored as an
output artifact rather than embedded in structured output.

An acknowledged upload is durably stored on the canonical run and can be repeated safely while the
session is open. The dashboard can show `submitted / total` progress before scoring begins. If
all upload attempts fail, `run_and_submit()` raises and does not request scoring for partial
coverage. Supply a stable `idempotency_key` when retrying the complete operation. Exactly-once
execution across a hard process or machine failure before acknowledgement requires the low-level
session API plus a durable runner-owned work queue.
`repository_url` and `commit_hash` are optional provenance metadata. When supplied, the platform
shows the source repository and exact revision in the variant's Architecture metadata. They can
also be provided inside `variant_identity`; explicit high-level keyword arguments take precedence.
Repository provenance accepts HTTPS URLs only and rejects embedded credentials.

`wait_for_submission()` returns immediately when the submission reaches `dispatch_failed`. This is
a recoverable delivery outcome rather than a completed score. Inspect `status["dispatch"]`, then
restore scoring on the same run without creating a replacement:

```python
status = gym.wait_for_submission(benchmark_id, submission_id)
if status["status"] == "dispatch_failed" and status["run"] is not None:
    restored = gym.retry_scoring(benchmark_id, status["run"]["id"])
```

The asynchronous client exposes the same operation as `await gym.retry_scoring(...)`. Submission
sessions also provide `submission.retry_scoring()` and reuse their submitted run.

With `trace_cases=True`, the runner creates one independent root span per case, adds benchmark and
case attributes, and links its nonzero trace ID to the prediction. Automatically instrumented work
inside the executor becomes a child of that span. This is opt-in and only operates after
`promptic_sdk.init()` has configured tracing; Agent Gym never initializes tracing implicitly. Raw
trace IDs returned by the executor are merged and deduplicated, so already-running external systems
remain supported.

Trace production and prediction upload are independent. Raw trace IDs are collected while cases
run and resolved once immediately before submission. `trace_policy="best_effort"` is the
default: unresolved IDs emit a warning and are omitted while predictions are still submitted.
`trace_policy="required"` preserves fail-closed behavior, and `trace_policy="disabled"` skips trace
resolution and removes trace evidence from the submitted predictions. The platform does not yet
support attaching trace IDs after a run has been submitted.

## Isolated untrusted execution

Never pass platform or provider credentials into generated, third-party, or otherwise untrusted
agent code. Run that code in a separately isolated environment with only the inputs and capabilities
it needs. A trusted, authenticated runner should use `start_submission()` or `resume_submission()` to
materialize the manifest, collect outputs, upload artifacts, upload prediction batches, and submit
the complete set for scoring.
The high-level callback API does not create this isolation boundary.

```python
with gym.start_submission(
    benchmark_id,
    idempotency_key="audit-agent:1.0.0:session",
    variant_identity={
        "name": "audit-agent",
        "version": "1.0.0",
        "architecture_description": "Evidence-guided audit with independent validation",
        "repository_url": "https://github.com/acme/audit-agent",
        "commit_hash": "6f1ed002ab5595859014ebf0951522d9d5f25a73",
    },
) as submission:
    manifest = submission.materialize_manifest("agent-gym-inputs")

    # Execute these cases elsewhere without exposing the Promptic credential.
    for case_id, result in externally_produced_results(manifest):
        submission.add_prediction(case_id, result)

    submitted = submission.submit(
        idempotency_key="audit-agent:1.0.0:submit",
        trace_policy="best_effort",
    )
```

`add_prediction()` accepts the same `AgentGymCaseResult` representation as the trusted executor. It
validates manifest membership and duplicates, uploads only that prediction's artifacts, and
immediately persists the terminal prediction on the canonical run without waiting for trace
ingestion. `submit()` resolves all pending raw trace IDs once according to `trace_policy`, reuploads
changed trace-linked predictions in dynamically bounded batches, requires exactly one prediction
for every immutable manifest case, and requests scoring. The context manager does not cancel an
unfinished remote session; it remains resumable until its server-side expiry.

Long-running external runtimes can persist progress directly with
`upload_predictions(benchmark_id, submission_id, predictions)`. Each call accepts at most 500
predictions and at most 1 MiB of serialized JSON, and replaces earlier values for the same cases
while the submission remains open. Once coverage is complete, call
`submit_submission(..., idempotency_key=...)`; an incomplete submission returns
`incomplete_submission`, and a dispatch failure can be retried with the same idempotency key.

## Result inspection

```python
with AgentGymClient() as gym:
    summary = gym.get_run_results(benchmark_id, run_id)
    print(summary["score_status_counts"])
    weakest = gym.list_case_results(benchmark_id, run_id, sort="score", limit=5)

    for result in weakest["data"]:
        print(result["case_id"], result["overall_score"])
        print(result["judgements"], result["traces"])

    all_cases = list(gym.iter_case_results(benchmark_id, run_id, page_size=100))
    one_case = gym.get_case_result(benchmark_id, run_id, all_cases[0]["case_id"])
```

Supported sorts are `score` (failures, then weakest first), `score_desc`, `latency` (descending), and
`case`. `compare_runs()` performs a paired comparison only when both single-architecture runs use the
same immutable revision, scorer contract, evaluator snapshot, and case set.

Inspect `score_status_counts` before trusting an aggregate. It separates succeeded scores from
failed, skipped, and insufficient-evidence outcomes. Verifier metrics appear as independent
evaluator results with `source_evaluator_id` and `metric_key`; use those fields rather than parsing
the opaque evaluator-result `id`. Per-field means declare a `succeeded_only` basis and therefore do
not include the zero-filled failures used by official aggregate scores.

## Artifact download safety

`download_prediction_artifact()` reconstructs the authenticated content route from the storage
object UUID instead of trusting a response URL. Authentication is never forwarded to the signed
storage redirect. Downloads are bounded, checked against the advertised exact size, written to a
same-directory temporary file, flushed, and atomically installed. Existing files and symbolic-link
destinations are refused unless explicitly handled by the caller.

The current result API does not expose the SHA-256 stored when external artifacts are submitted. The
SDK verifies a response `sha256` automatically when the platform supplies one in the future; today,
pass a trusted out-of-band digest with `expected_sha256=` when hash verification is required.

See [`examples/agent_gym_external_submission.py`](../examples/agent_gym_external_submission.py) for
an executable HTML-artifact submission with trace linkage, scoring wait, weakest-case inspection,
and artifact download. See
[`examples/agent_gym_author_benchmark.py`](../examples/agent_gym_author_benchmark.py) for an
executable benchmark-authoring and publishing workflow.
