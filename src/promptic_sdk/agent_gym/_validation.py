"""Readable validation errors shared by authoring, HTTP clients, and the CLI."""

from __future__ import annotations

from typing import Any


def _path_label(path: Any) -> str:
    if isinstance(path, str):
        return path
    if not isinstance(path, list):
        return ""
    result = ""
    for segment in path:
        if isinstance(segment, int):
            result += f"[{segment}]"
        elif isinstance(segment, str):
            result += f".{segment}" if result else segment
    return result


def validation_message(message: str, details: dict[str, Any] | None) -> str:
    """Render issue paths and legacy flattened validation messages without dumping payloads."""
    if not details:
        return message
    messages: list[str] = []
    issues = details.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict) or not isinstance(issue.get("message"), str):
                continue
            path = _path_label(issue.get("path", issue.get("fieldPath")))
            messages.append(f"{path}: {issue['message']}" if path else issue["message"])
    field_errors = details.get("fieldErrors")
    if isinstance(field_errors, dict):
        for path, errors in field_errors.items():
            if isinstance(errors, list):
                messages.extend(f"{path}: {error}" for error in errors if isinstance(error, str))
    form_errors = details.get("formErrors")
    if isinstance(form_errors, list):
        messages.extend(error for error in form_errors if isinstance(error, str))
    if not messages:
        return message
    generic = {"Invalid body", "Invalid draft update", "Invalid evaluator configuration."}
    if message not in generic and not any(
        item == message or item.endswith(f": {message}") for item in messages
    ):
        messages.insert(0, message)
    return "\n".join(dict.fromkeys(messages))


class EvaluatorConfigurationError(ValueError):
    """Invalid local evaluator collection, with structured correction details."""

    code = "duplicate_evaluator_kind"

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        """Retain issue paths and the first error code for CLI diagnostics."""
        if issues:
            self.code = issues[0]["code"]
        self.details = {"issues": issues}
        self.message = validation_message("Invalid evaluator configuration.", self.details)
        super().__init__(self.message)
