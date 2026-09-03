"""Deprecated workspace commands — kept as an alias for ``ai-application``.

The customer-facing name for this resource is now **AI Application**. This module
re-exports the AI Application command group so the legacy ``promptic workspace``
subcommand keeps working.
"""

from __future__ import annotations

from promptic_sdk.cli.commands.ai_application import ai_application_app

# Backward-compatible alias.
workspace_app = ai_application_app

__all__ = ["workspace_app"]
