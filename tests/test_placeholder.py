"""Verify package version and public API."""

from promptic_sdk import PrompticClient, __version__, ai_component, init


def test_version():
    """Verify package version is set."""
    assert __version__ == "0.1.0"


def test_public_api():
    """Verify public API exports."""
    assert callable(init)
    assert callable(PrompticClient)
    assert callable(ai_component)
