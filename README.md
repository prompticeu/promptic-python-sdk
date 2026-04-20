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

This opens your browser for authentication, then auto-selects your workspace. Credentials are saved to `~/.promptic/config.toml`.

For CI/CD or headless environments, use an API key instead:

```bash
promptic configure
# or set the environment variable:
export PROMPTIC_API_KEY="pk_..."
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

    # Get workspace info
    workspace = client.get_workspace()

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

Use `ai_component()` to tag spans with a component name. The platform links traces to the matching AI Component in your workspace:

```python
with promptic_sdk.ai_component("my-component"):
    # All LLM calls here are tagged
    ...
```

## API client

Both a sync (`PrompticClient`) and async (`AsyncPrompticClient`) client are available. They share the same method signatures and return types.

```python
from promptic_sdk import PrompticClient

with PrompticClient() as client:
    traces = client.list_traces(limit=10)
```

```python
from promptic_sdk import AsyncPrompticClient

async with AsyncPrompticClient() as client:
    traces = await client.list_traces(limit=10)
```

Both clients provide typed methods for the full Promptic REST API:

| Resource       | Methods                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| Workspace      | `get_workspace`                                                         |
| Traces         | `list_traces`, `get_trace`, `get_stats`                                 |
| Components     | `list_components`, `get_component`, `create_component`, `delete_component` |
| Experiments    | `list_experiments`, `get_experiment`, `create_experiment`, `update_experiment`, `delete_experiment`, `start_experiment` |
| Observations   | `list_observations`, `create_observations`, `update_observation`, `delete_observation` |
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
| `promptic workspace list`              | List accessible workspaces             |
| `promptic workspace select <id>`       | Select a workspace                     |
| `promptic workspace info`              | Show workspace info                    |
| `promptic traces list`                 | List recent traces                     |
| `promptic traces get <id>`             | Get a trace with spans                 |
| `promptic traces stats`               | Show aggregated tracing stats          |
| `promptic components list`             | List AI components                     |
| `promptic components create`           | Create a component                     |
| `promptic components get <id>`         | Get component details                  |
| `promptic components delete <id>`      | Delete a component                     |
| `promptic experiments list`            | List experiments                       |
| `promptic experiments create`          | Create an experiment (interactive)     |
| `promptic experiments get <id>`        | Get experiment details                 |
| `promptic experiments update <id>`     | Update an experiment                   |
| `promptic experiments delete <id>`     | Delete an experiment                   |
| `promptic experiments start <id>`      | Start an experiment                    |
| `promptic observations list`           | List observations for an experiment    |
| `promptic observations add`            | Add an observation                     |
| `promptic observations delete <id>`    | Delete an observation                  |
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
| `promptic runs create`                 | Create a run                           |
| `promptic runs list`                   | List runs                              |
| `promptic runs get <id>`               | Get run details                        |
| `promptic runs delete <id>`            | Delete a run                           |
| `promptic annotations create`          | Create an annotation                   |
| `promptic annotations list`            | List annotations                       |
| `promptic annotations delete <id>`     | Delete an annotation                   |
| `promptic evaluations run`             | Run an evaluation                      |
| `promptic evaluations list`            | List evaluations                       |
| `promptic evaluations get <id>`        | Get evaluation details                 |

All list commands support `--json` for machine-readable output.

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
