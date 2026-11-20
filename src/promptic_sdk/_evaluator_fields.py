"""Validation shared by Prompt Optimization and Agent Gym targeted evaluators."""

from __future__ import annotations

TARGETED_EVALUATOR_TYPES = ("exactMatch", "containsText", "semanticSimilarity", "aiJudge")
"""Prompt Optimization evaluator types that score one output field or the whole response."""


def validate_target_field(value: str | None) -> str | None:
    """Return a stripped dotted output path, or ``None`` to score the whole response.

    Mirrors the server's schema-independent checks: the path must be non-empty, use dots
    between non-empty segments, and must not point into list items.

    Raises:
        ValueError: If the path is malformed.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "field must be a non-empty dotted output path; omit it to score the whole response"
        )
    path = value.strip()
    if "[" in path or "]" in path:
        raise ValueError(
            f"field '{path}' points into list items; target the list itself "
            "(for example 'people' instead of 'people[].name')"
        )
    if any(not segment.strip() for segment in path.split(".")):
        raise ValueError(f"field '{path}' must be a dotted path such as 'answer' or 'user.name'")
    return path
