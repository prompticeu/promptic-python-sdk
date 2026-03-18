# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## v0.2.0 (2026-03-18)

### Feat

- **release**: add automated release workflow with PyPI publishing
- **release**: add automated release workflow with PyPI publishing
- **sdk**: add deep agent example, LangSmith OTel bridge, and use PyPI installs
- **sdk**: add dataset-level annotation listing
- **sdk**: add runs, annotations, evaluations CLI and prefer session auth
- **cli**: auto-select workspace after login and update README
- **sdk**: improve tracing, CLI login, and fix default endpoint
- add AsyncPromenticClient and comprehensive README
- add typed models, CLI commands, and deployed prompt support
- setup python sdk repository

### Fix

- rename PromenticClient to PrompticClient and remove unused CLI params
- **tracing**: catch ModuleNotFoundError from find_spec in auto_instrument
- mock config file in default endpoint test
- pin Python to 3.13 to fix CI segfault
- pin Python to 3.11 to fix CI segfault
