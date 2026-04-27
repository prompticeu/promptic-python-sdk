# Changelog

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
