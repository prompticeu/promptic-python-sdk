# Changelog

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
