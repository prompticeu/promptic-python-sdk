"""Minimal Agent Gym submission using normal Promptic authentication."""

from pathlib import Path

from promptic_sdk import (
    AgentGymCase,
    AgentGymCaseResult,
    AgentGymClient,
)


def my_agent(case: AgentGymCase) -> AgentGymCaseResult:
    """Replace this body with the candidate being benchmarked."""
    report = Path("outputs") / case.id / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"<h1>{case.input['title']}</h1>", encoding="utf-8")
    return AgentGymCaseResult.artifact(report)


with AgentGymClient() as gym:
    result = gym.submit(
        benchmark_id="11111111-1111-4111-8111-111111111111",
        candidate=my_agent,
        name="html-report-agent",
        version="1.0.0",
    )

print(result.run_id)
