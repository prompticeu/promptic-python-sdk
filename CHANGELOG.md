# Changelog

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
