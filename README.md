# Promptic Python SDK

Python SDK and CLI for the [Promptic](https://promptic.eu) platform — tracing, prompt optimization, and experiment management.

## Installation

```bash
pip install promptic-sdk
```

### Optional LLM instrumentation

Install extras to auto-instrument specific providers or agent frameworks:

```bash
# LLM providers
pip install promptic-sdk[openai]         # OpenAI
pip install promptic-sdk[anthropic]      # Anthropic
pip install promptic-sdk[bedrock]        # AWS Bedrock
pip install promptic-sdk[vertexai]       # Google Vertex AI
pip install promptic-sdk[mistralai]      # Mistral

# Agent frameworks
pip install promptic-sdk[langchain]      # LangChain / LangGraph / create_agent / deepagents
pip install promptic-sdk[openai-agents]  # OpenAI Agents SDK
pip install promptic-sdk[claude-agent]   # Claude Agent SDK

pip install promptic-sdk[all]            # Everything above
```

Pydantic AI ships its own OpenTelemetry emitter — enable it with
`Agent(..., instrument=True)`; no extras needed.

## Quick start

### 1. Authenticate

Log in via browser (recommended for local development):

```bash
promptic login
```

This opens your browser for authentication, then auto-selects your AI Application. Credentials are saved to `~/.promptic/config.toml`.

For CI/CD or headless environments, use an API key instead:

```bash
promptic configure
# or set the environment variable:
export PROMPTIC_API_KEY="ptc_..."
```

### 2. Send traces

```python
import promptic_sdk
from openai import OpenAI

# Initialize tracing (auto-instruments installed LLM libraries)
promptic_sdk.init()

client = OpenAI()

# Tag traces with an AI Component name
with promptic_sdk.ai_component("customer-support-agent"):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

### 3. Use the API client

```python
from promptic_sdk import PrompticClient

with PrompticClient() as client:
    # List traces
    traces = client.list_traces(limit=10)

    # Get AI Application info
    ai_application = client.get_ai_application()

    # Manage experiments
    experiment = client.create_experiment(
        ai_component_id="comp_...",
        target_model="gpt-4.1-nano",
        task_type="classification",
        initial_prompt="Classify the following text.",
    )

    # Deploy the best prompt
    client.deploy(component_id="comp_...", experiment_id="exp_...")

    # Fetch a deployed prompt at runtime
    prompt = client.get_deployed_prompt("comp_...")
```

### Tool-selection experiments

Prompt experiments use `create_experiment()` with `classification`, `textGeneration`, or
`structuredOutput`. Tool selection requires additional configuration and uses a typed,
all-or-nothing workflow:

```python
with PrompticClient() as client:
    experiment = client.create_tool_selection_experiment(
        "comp_...",
        tools=[{"name": "get_weather", "description": "Get weather for a city"}],
        test_cases=[{"query": "Weather in Berlin?", "expected_tool": "get_weather"}],
        target_model="gpt-4.1-nano",
        tool_source="manual",
        optimize_system_prompt=True,
    )
    client.start_experiment(experiment["id"])
```

The experiment dataset is managed automatically and deleted with the experiment. Use
`create_dataset()` for a named, reusable dataset shared by evaluations or other workflows.
Each iteration returned by `list_iterations()`, `get_iteration()`, or
`get_best_iteration()` may include `toolDescriptions` (the optimized description keyed by
tool name) and `selectionSystemPrompt` when system-prompt optimization is enabled.

## Tracing

`promptic_sdk.init()` sets up OpenTelemetry to export spans to the Promptic platform.

| Parameter          | Description                                         | Default                      |
| ------------------ | --------------------------------------------------- | ---------------------------- |
| `api_key`          | Promptic API key (falls back to `PROMPTIC_API_KEY`) | —                            |
| `endpoint`         | Platform URL (falls back to `PROMPTIC_ENDPOINT`)    | `https://promptic.eu`    |
| `auto_instrument`  | Auto-detect and instrument LLM client libraries     | `True`                       |
| `service_name`     | OpenTelemetry `service.name` resource attribute      | —                            |

Auto-detected instrumentors: OpenAI, Anthropic, Google Generative AI, Vertex AI,
Bedrock, Mistral, Cohere, LangChain (with LangGraph / deepagents), OpenAI Agents
SDK, Claude Agent SDK. All emit the official OpenTelemetry GenAI semantic
conventions (`gen_ai.*`), so traces work uniformly across frameworks.

### Using other OpenTelemetry instrumentors

Since Promptic uses standard OpenTelemetry under the hood, you can add any OTel-compatible instrumentor alongside the auto-detected ones. Just call `promptic_sdk.init()` first, then instrument manually:

```python
import promptic_sdk
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

promptic_sdk.init()

# Add any OpenTelemetry instrumentor — spans will be exported to Promptic
RequestsInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument(engine=engine)
```

This works with any package from the [opentelemetry-python-contrib](https://github.com/open-telemetry/opentelemetry-python-contrib) ecosystem (HTTP clients, databases, web frameworks, etc.). All spans are exported to the Promptic platform as long as `init()` has been called.

### AI Components

Use `ai_component()` to tag spans with a component name. The platform links traces to the matching AI Component in your AI Application:

```python
with promptic_sdk.ai_component("my-component"):
    # All LLM calls here are tagged
    ...
```

To add traces to a dataset, pass the dataset's immutable UUID. Datasets are
created explicitly in the dashboard, CLI, or API; tracing never creates one
from a display name.

```python
with promptic_sdk.ai_component(
    "my-component",
    dataset_id="550e8400-e29b-41d4-a716-446655440000",
):
    agent.run(test_input)
```

Invalid dataset IDs fail immediately, before spans are created. The SDK emits
the `promptic.dataset.id` OpenTelemetry attribute for server-side linkage.

### Tracing workflows with custom spans

Most users don't need this. With the right `[extras]` installed, auto-instrumentation already
creates spans for every LLM and tool call. Reach for custom spans only when you have meaningful
**non-LLM** workflow logic (retrieval, normalization, business rules, control flow) you want
represented in the trace.

When you do need it, wrap your workflow stages in custom OpenTelemetry spans. Auto-instrumented LLM and tool spans automatically nest under whichever custom span is active.

```python
import json
import promptic_sdk
from opentelemetry import trace

promptic_sdk.init()
tracer = trace.get_tracer(__name__)

with promptic_sdk.ai_component("support-agent"):
    with tracer.start_as_current_span("answer_question") as root:
        root.set_attribute("traceloop.span.kind", "workflow")
        root.set_attribute("traceloop.entity.input", json.dumps(user_input))

        with tracer.start_as_current_span("retrieve_context") as span:
            span.set_attribute("traceloop.span.kind", "task")
            span.set_attribute("traceloop.entity.input", json.dumps(query))
            context = retrieve(query)
            span.set_attribute("traceloop.entity.output", json.dumps(context))

        with tracer.start_as_current_span("generate_answer") as span:
            span.set_attribute("traceloop.span.kind", "task")
            # The auto-instrumented LLM call nests under this task span
            answer = llm_call(context)

        root.set_attribute("traceloop.entity.output", json.dumps(answer))
```

Span attribute conventions:

- `traceloop.span.kind="workflow"` — the top-level run
- `traceloop.span.kind="task"` — an internal pipeline stage
- `traceloop.entity.input` / `traceloop.entity.output` — JSON-serialized stage payloads, surfaced in the Promptic UI

Promptic automatically offloads inline base64 media and large file-like content
into trace artifacts, then keeps lightweight `promptic-artifact://...`
references in span attributes. Local filesystem paths are not read
automatically; attach local files explicitly when you want the bytes available
from the trace:

```python
file_ref = promptic_sdk.artifact("/tmp/report.pdf")
span.set_attribute("retrieval.input_file", file_ref.ref)
```

The artifact's `name` is stored on the record and used as the default download
filename. For local files it defaults to the file's base name; pass `name=` to
override it (also available for bytes and text content), and read it back from
`file_ref.name`:

```python
file_ref = promptic_sdk.artifact(pdf_bytes, name="quarterly-report.pdf")
```

For huge collections, still log a small preview plus a count rather than the
full object:

```python
span.set_attribute(
    "traceloop.entity.output",
    json.dumps({
        "items": items[:5],
        "item_count": len(items),
        "additional_item_count": max(len(items) - 5, 0),
    }),
)
```

See the [Tracing guide](https://promptic.eu/docs/guides/tracing#tracing-workflows-with-custom-spans) for the full pattern.

## Agent Gym

Author a benchmark and atomically add private cases. Publish an immutable revision explicitly once
the configuration is ready:

```python
from promptic_sdk import (
    AgentGymClient,
    BenchmarkCase,
    BenchmarkFile,
    FieldScoring,
    FieldLevelJudge,
)

with AgentGymClient(workspace_id="<workspace-uuid>") as gym:
    benchmark = gym.benchmarks.create(
        name="Document Comparison",
        goal="Compare the supplied documents and produce a review report.",
        input_schema={
            "type": "object",
            "properties": {
                "characteristic_id": {"type": "string"},
                "files": {"type": "array", "x-promptic-type": "file"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "report": {"type": "string"},
                "confidence": {"type": "number"},
            },
        },
        evaluators=[
            FieldLevelJudge(
                fields={
                    "report": FieldScoring(method="judge", judge_instructions="Check cited evidence."),
                    "confidence": FieldScoring(method="exact"),
                },
            )
        ],
    )
    shared = BenchmarkFile("shared-instructions.pdf")
    benchmark.cases.add_many([
        BenchmarkCase(
            input={"characteristic_id": "F_26", "files": [shared]},
            output={"report": "Review section A.", "confidence": 1.0},
        ),
        BenchmarkCase(
            input={"characteristic_id": "F_27", "files": [shared]},
            output={"report": "Review section B.", "confidence": 1.0},
        ),
    ])
    published = benchmark.publish()
    revision_id = published.revision.id
```

Benchmark authoring currently follows the platform's alpha API and therefore requires an admin
account. The SDK exposes typed schemas and evaluator models, imports cases in confirmed batches,
and leaves every authoring change unpublished until `publish()` is called.

The Output schema defines the expected shape. Explicit evaluators define how that output is scored;
the field-level evaluator can mix deterministic and judge-based strategies per field.

Run a trusted local candidate against an immutable Agent Gym benchmark, upload its output artifacts
and traces, wait for scoring, and inspect the weakest cases:

```python
from pathlib import Path

from promptic_sdk import AgentGymCase, AgentGymCaseResult, AgentGymClient, AgentGymOutputArtifact


def execute_case(case: AgentGymCase) -> AgentGymCaseResult:
    report = Path(f"report-{case.ordinal}.html")
    report.write_text(build_report(case.input))
    return AgentGymCaseResult.artifact(
        AgentGymOutputArtifact(report, field_path="report")
    )


with AgentGymClient() as gym:
    submitted = gym.run_and_submit(
        benchmark_id="<benchmark-uuid>",
        executor=execute_case,
        name="report-agent",
        version="1.0.0",
        architecture_description="Builds and validates a standalone HTML report.",
        repository_url="https://github.com/acme/report-agent",
        commit_hash="6f1ed002ab5595859014ebf0951522d9d5f25a73",
        trace_cases=True,
        trace_policy="best_effort",
    )
    summary = gym.get_run_results("<benchmark-uuid>", submitted.run_id)
    weakest = gym.list_case_results(
        "<benchmark-uuid>", submitted.run_id, sort="score", limit=5
    )
```

`run_and_submit()` executes the callback inside your authenticated Python process. Use it only for
code you trust. It does **not** sandbox generated or untrusted agents. `trace_cases=True` creates and
links one independent root trace per case when Promptic tracing is already configured; it never
initializes tracing implicitly. Trace resolution happens once before submission and defaults to
`trace_policy="best_effort"`, so delayed or failed trace ingestion cannot discard valid predictions.
Use `"required"` when trace evidence must be present or `"disabled"` to omit it entirely.
Each completed case prediction is uploaded immediately. Prediction uploads are idempotent, retry
transient transport, rate-limit, and server failures up to three times, and are bounded to 1 MiB
per request. The lower-level session helper also splits queued predictions dynamically by both
serialized size and the API's 500-item limit. Store larger outputs as artifacts.

After an upload is acknowledged, rerunning it safely replaces the same case rather than creating a
duplicate. If all upload attempts fail, `run_and_submit()` raises and never submits partial
coverage for scoring. Use a stable `idempotency_key` when retrying a whole run. A hard process or
machine failure before the upload acknowledgement cannot provide exactly-once local execution; use
the low-level session API with your own durable work queue when that guarantee is required.
Optional `repository_url` and `commit_hash` values preserve the source revision as variant
Architecture metadata. Low-level external runtimes can provide the same keys in
`submit(identity=...)`. Repository URLs must use HTTPS and cannot contain credentials.

If `wait_for_submission()` returns the recoverable `dispatch_failed` state, call
`retry_scoring(benchmark_id, run_id)` to restore scoring delivery for that same run. Sync, async,
and submission-session clients expose this operation; it never creates a replacement run.

Run untrusted code in a separately isolated, credentialless environment, then let a trusted runner
use `start_submission(variant_identity=...)`, `add_prediction()`, and `submit()` to persist its predictions
and request scoring.

Sync and async clients expose the same lifecycle and inspection surface: submission creation and
resume, manifest paging/materialization, artifact upload and verification, trace resolution,
submission and polling, aggregate results, cursor-paginated case results, paired comparisons, and
bounded atomic artifact downloads. See the [Agent Gym guide](docs/agent-gym.md) and the
[authoring](examples/agent_gym_author_benchmark.py) and
[submission](examples/agent_gym_external_submission.py) examples.

## API client

Both a sync (`PrompticClient`) and async (`AsyncPrompticClient`) client are available. They share the same method signatures and return types.

```python
from promptic_sdk import PrompticClient

with PrompticClient() as client:
    traces = client.list_traces(limit=10)
    models = client.models.list()
    judge_model_ids = [model["id"] for model in models["data"] if model["judgeEligible"]]
```

`models.list()` requires a platform deployment that exposes `GET /api/v1/models`.
When explicitly configuring a benchmark judge evaluator, use only models with
`judgeEligible == True` (the OpenAI group). Omit the judge model to use the
platform default. An explicit, non-eligible model receives a 400 configuration
error once platform validation is deployed.

```python
from promptic_sdk import AsyncPrompticClient

async with AsyncPrompticClient() as client:
    traces = await client.list_traces(limit=10)
    models = await client.models.list()
```

Both clients provide typed methods for the full Promptic REST API:

| Resource       | Methods                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| AI Application | `get_ai_application`                                                     |
| Models         | `models.list` (AI Application-scoped available models)                   |
| Traces         | `list_traces`, `get_trace`, `list_trace_artifacts`, `get_artifact`, `get_artifact_content`, `download_artifact`, `get_stats` |
| Components     | `list_components`, `get_component`, `create_component`, `delete_component` |
| Experiments    | `list_experiments`, `get_experiment`, `create_experiment`, `update_experiment`, `delete_experiment`, `start_experiment` |
| Dataset cases | `list_dataset_cases`, `get_dataset_case`, `create_dataset_cases`, `update_dataset_case`, `delete_dataset_case` |
| Evaluators     | `list_evaluators`, `create_evaluators`, `update_evaluator`, `delete_evaluator` |
| Iterations     | `list_iterations`, `get_iteration`, `get_best_iteration`                |
| Deployments    | `get_deployment`, `deploy`, `undeploy`, `get_deployed_prompt`           |

The client reads `PROMPTIC_API_KEY` and `PROMPTIC_ENDPOINT` from the environment, or accepts them as constructor arguments.

## CLI

The `promptic` CLI mirrors the API client and supports both human-readable tables and `--json` output.

```
promptic [command] [subcommand] [options]
```

### Commands

| Command                                | Description                            |
| -------------------------------------- | -------------------------------------- |
| `promptic login`                       | Authenticate via browser (device flow) |
| `promptic logout`                      | Clear saved credentials                |
| `promptic configure`                   | Save API key and endpoint (CI/CD)      |
| `promptic ai-application list`         | List accessible AI Applications        |
| `promptic ai-application select <id>`  | Select an AI Application                |
| `promptic ai-application info`         | Show AI Application info                |
| `promptic models list`                 | List available models                   |
| `promptic traces list`                 | List recent traces                     |
| `promptic traces get <id>`             | Get a trace with spans                 |
| `promptic traces artifacts <id>`       | List artifacts for a trace             |
| `promptic artifacts get <id> -o file`  | Download an artifact                   |
| `promptic traces stats`               | Show aggregated tracing stats          |
| `promptic components list`             | List AI components                     |
| `promptic components create`           | Create a component                     |
| `promptic components get <id>`         | Get component details                  |
| `promptic components delete <id>`      | Delete a component                     |
| `promptic experiments list`            | List experiments                       |
| `promptic experiments create`          | Create/configure an experiment         |
| `promptic experiments create-tool-selection` | Create a tool-selection experiment |
| `promptic experiments get <id>`        | Get experiment details                 |
| `promptic experiments update <id>`     | Update an experiment                   |
| `promptic experiments delete <id>`     | Delete an experiment                   |
| `promptic experiments start <id>`      | Start an experiment                    |
| `promptic evaluators list`             | List evaluators for an experiment      |
| `promptic evaluators add`              | Add an evaluator                       |
| `promptic evaluators delete <id>`      | Delete an evaluator                    |
| `promptic iterations list`             | List iterations for an experiment      |
| `promptic iterations get <id>`         | Get iteration details                  |
| `promptic iterations best`             | Get the best iteration                 |
| `promptic deployments status <id>`     | Show deployment for a component        |
| `promptic deployments deploy`          | Deploy an experiment                   |
| `promptic deployments prompt <id>`     | Show the deployed prompt               |
| `promptic deployments undeploy <id>`   | Remove a deployment                    |
| `promptic datasets create`             | Create a dataset                       |
| `promptic datasets list`               | List datasets                          |
| `promptic datasets get <id>`           | Get dataset details                    |
| `promptic datasets delete <id>`        | Delete a dataset                       |
| `promptic datasets cases list <id>`    | List canonical dataset cases           |
| `promptic datasets cases get <id> <case-id>` | Get a canonical dataset case    |
| `promptic datasets cases add <id> --file cases.json` | Upload canonical cases     |
| `promptic datasets cases update <id> <case-id> --file case.json` | Update a case |
| `promptic datasets cases delete <id> <case-id>` | Delete a canonical case       |

All list commands support `--json` for machine-readable output.

`promptic experiments create` accepts `--hyperparameters <file>` and
`--output-schema <file>` JSON documents. Use `--start` to schedule the experiment immediately.

## Configuration

The SDK and CLI resolve configuration in this order:

1. Explicit arguments (`api_key=`, `endpoint=`)
2. Environment variables (`PROMPTIC_API_KEY`, `PROMPTIC_ENDPOINT`)
3. Config file (`~/.promptic/config.toml`, written by `promptic login` or `promptic configure`)

| Variable            | Description                  | Default                   |
| ------------------- | ---------------------------- | ------------------------- |
| `PROMPTIC_API_KEY`  | API key (for tracing & CI/CD)| —                         |
| `PROMPTIC_ENDPOINT` | Platform URL                 | `https://promptic.eu`     |

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .
uv run ruff format .
```

## License

MIT — see [LICENSE](LICENSE) for details.
