# Changelog

## v0.24.0 (2026-09-18)

## Features

- Expose available models through the SDK and CLI.


## v0.23.0 (2026-09-18)

## Features

- Added a prompt optimization workflow to the CLI, bringing it to feature parity with the SDK.

## Improvements

- Unified Agent Gym field-scoring methods in the Python SDK.


## v0.22.0 (2026-09-04)

## Features

- Added an Agent Gym SDK for submitting external evaluations, retrieving results, and authoring tasks.


## Unreleased

## Features

- Added complete canonical dataset-case CLI CRUD through `promptic datasets cases`.
- Added CLI experiment hyperparameter files, structured-output schemas, and optional immediate
  start support.
- Added typed verifier-agent evidence selection, required grounding evidence, sandbox tool
  capabilities, and investigation budgets to Agent Gym benchmark authoring.
- Added typed sync and async Agent Gym benchmark authoring for definitions, structured field
  scoring, LLM/agent judges, private case and reference files, bulk cases, draft status, and
  immutable revision publishing against the platform's current alpha admin API.
- Added explicit Input and Output schemas plus typed field maps on
  `AgentEvaluator.field_level_judge()`, keeping data contracts independent from evaluator choice.
- Added modular typed sync and async Agent Gym clients for external submission sessions, safe
  manifest materialization, output artifacts, trace linkage, trusted callback execution, and scoring
  polling.
- Added authenticated aggregate and case result inspection, paired run comparison, and bounded
  atomic prediction artifact downloads with exact-size and optional SHA-256 verification.
- Added `run_and_submit(executor=...)` with opt-in automatic per-case root traces.
- Persisted each completed callback prediction immediately with bounded transient retries, while
  dynamically limiting session upload batches by both serialized size and API item count.
- Added sync and async session prediction builders that upload result artifacts, resolve trace
  references, validate exact manifest coverage, upload prediction batches, and request scoring.
- Decoupled canonical prediction uploads from trace ingestion with submission-wide `required`, `best_effort`,
  and `disabled` trace policies. Trace IDs are now resolved once at submission, and best-effort
  failures no longer prevent valid predictions from being submitted.
- Added optional `repository_url` and `commit_hash` variant provenance to typed low-level
  retry-safe prediction uploads and trusted sync/async Agent Gym submission APIs.
- Aligned immutable manifest and result case identifiers with canonical integer dataset-case IDs,
  and added typed schema-migration draft inspection/resolution plus run re-evaluation APIs.
- Added Agent Gym CLI commands for revision history, migration drafts, conflict resolution, draft
  abandonment, and evaluator-only re-evaluation.
- Aligned benchmark bulk authoring and CLI apply with automatic ready-revision activation, so one
  bulk import cannot expose intermediate per-case revisions.

## Documentation

- Added an executable benchmark-authoring example covering artifact judge evidence, private case
  files, reference evidence, and publishing.
- Added an Agent Gym guide and executable HTML artifact example that distinguish trusted in-process
  callbacks from isolated untrusted agent execution.

## Bug Fixes

- Ensured an explicit `PROMPTIC_API_KEY` is not shadowed by a stale access token in the saved CLI
  configuration.

## v0.21.1 (2026-09-03)

## Improvements

- Removed the legacy Agent Evaluation API.


## v0.21.0 (2026-09-03)

## Features

- Adopted “AI Application” terminology across the public client API.

## Bug Fixes

- Exposed tool-selection iteration outputs from models.


## v0.20.0 (2026-08-27)

## Features

- Added a typed tool-selection workflow for experiments.


## v0.19.0 (2026-08-06)

## Features

- Added support for specifying an artifact name in tracing.

## Bug Fixes

- Improved cleanup handling for superseded releases.

## v0.18.0 (2026-07-31)

## Features

- Use canonical dataset IDs for tracing.

## v0.17.0 (2026-07-31)

## Features

- Added `JudgeResult` types and a client method for listing judge results.
- Added `avgPredictionLatencyMs` to `Iteration`.

## Bug Fixes

- Updated API key examples to use the `ptc_` prefix.
- Fixed changelog generation to use OpenAI.

## Improvements

- Added a gated production release workflow.


## v0.16.0 (2026-05-28)

## Features

- Added support for instrumentor selection in tracing.

## Improvements

- Updated client documentation to note 402 billing gate on start_experiment and create_evaluation.

## v0.15.0 (2026-05-23)

Features

- Added support for uploading trace artifacts directly, making tracing easier and more efficient.

## v0.14.4 (2026-05-20)

## Bug Fixes

- Improved tracing support to handle the current OTLP exporter signature, ensuring better compatibility and reliability when exporting traces.

## v0.14.3 (2026-05-20)

## Bug Fixes

- Improved tracing to correctly preserve OpenAI image inputs.

## v0.14.2 (2026-04-30)

## Bug Fixes

- Improved tracing reliability by splitting oversized OTLP batches when receiving HTTP 413 errors.

## Improvements

- Added documentation for custom workflow span tracing patterns.

## Unreleased

## Bug Fixes

- Tracing: oversized OTLP span batches (chatty agent traces with large message attributes) were being rejected by the Promptic ingest with HTTP 413 and silently dropped. The SDK now wraps the OTLP exporter in a bisecting wrapper that, on a 413 response, halves the batch and retries each half recursively. Single spans that are too large are dropped with a clear error log instead of breaking the export pipeline.

## v0.14.1 (2026-04-27)

## Bug Fixes
- Updated EvaluatorType handling after judge module split for improved accuracy in evaluators.
- Synced experiment response schema with recent removal of initialPrompt field to ensure compatibility.

## Improvements
- Documentation: Added evalNormalizedScore attribute to Iteration type definition for enhanced clarity.

## v0.14.0 (2026-04-25)

## Features

- Added duplicate and continue commands to the CLI for experiments.



## Unreleased

## Features

- Added `experiments duplicate` and `experiments continue` CLI commands (and `PrompticClient.duplicate_experiment` / `AsyncPrompticClient.duplicate_experiment`) to clone an experiment with its observations and evaluators. `continue` seeds the new experiment from the source's best optimized prompt; both commands accept `--start` to enqueue the new run immediately.

## v0.13.0 (2026-04-22)

## Features

- Added tracing: When LANGSMITH_TRACING=true, the SDK now warns users if it overrides LangChain callbacks.

## v0.12.0 (2026-04-22)

## Features

- Migrated tracing to OpenLLMetry for improved observability.
- Removed LangSmith auto-bridge integration to streamline tracing.

## v0.11.3 (2026-04-17)

## Bug Fixes

- Normalize observation variables to ensure consistent handling within the SDK.

## v0.11.2 (2026-04-09)

## Bug Fixes

- Improved the deployments prompt in CLI for better user experience.
- Added support for multi-message format in CLI deployments.

## v0.11.1 (2026-04-09)

## Bug Fixes

- CLI now gracefully handles missing prompt fields in the deployments prompt command.

## v0.11.0 (2026-04-08)

## Features
- CLI now requires the --run flag for evaluations, ensuring clearer and more explicit command usage.

## v0.10.1 (2026-03-18)

## Bug Fixes

- Release assets now only include .whl and .tar.gz files, reducing unnecessary uploads.

## v0.10.0 (2026-03-18)

## Features

- Added workflow_dispatch trigger to support manual testing during release.

## Bug Fixes

- Fixed version detection mechanism.
- Resolved cascading loop issue in the release process.
- Reset changelog to avoid errors.
