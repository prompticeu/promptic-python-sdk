# /// script
# requires-python = ">=3.11"
# dependencies = ["promptic-sdk"]
# ///
"""Author and publish an artifact-judged Agent Gym benchmark.

Run with:
    uv run --no-project --env-file .env examples/agent_gym_author_benchmark.py

Environment variables:
    PROMPTIC_API_KEY          - API key owned by a platform admin
    PROMPTIC_AI_APPLICATION_ID - AI Application UUID (needed for login tokens;
                                optional with an AI Application-scoped API key)
    AGENT_GYM_INPUT_FILE      - Local PDF, document, image, or text input
    AGENT_GYM_REFERENCE_FILE  - Local private reference output

The platform's benchmark-authoring routes are currently alpha/admin-gated.
"""

from __future__ import annotations

import os
from pathlib import Path

from promptic_sdk import (
    AgentGymClient,
    BenchmarkFile,
    EvidenceKind,
    EvidencePolicy,
    ExpectedBehaviorJudge,
    InvestigationBudget,
    VerifierAgent,
    VerifierMetric,
)

AI_APPLICATION_ID = os.environ.get("PROMPTIC_AI_APPLICATION_ID")
INPUT_FILE = Path(os.environ["AGENT_GYM_INPUT_FILE"])
REFERENCE_FILE = Path(os.environ["AGENT_GYM_REFERENCE_FILE"])

with AgentGymClient(ai_application_id=AI_APPLICATION_ID) as gym:
    benchmark = gym.benchmarks.create(
        name="Document Comparison",
        goal=(
            "Compare the supplied source documents and produce a self-contained review report "
            "with evidence for every material conclusion."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "source_document": {"type": "string", "x-promptic-type": "file"},
            },
            "required": ["task", "source_document"],
        },
        output_schema={
            "type": "object",
            "properties": {"reference_report": {"type": "string", "x-promptic-type": "file"}},
            "required": ["reference_report"],
        },
        evaluators=[
            VerifierAgent(
                instructions="Assess the generated comparison against the selected evidence.",
                metrics=(
                    VerifierMetric(
                        "correctness",
                        "Correctness",
                        "Check facts and material differences.",
                        weight=2,
                        threshold=0.8,
                    ),
                    VerifierMetric(
                        "usability", "Reviewer usability", "Check clarity and actionable detail."
                    ),
                ),
                evidence=EvidencePolicy(
                    selected=(
                        EvidenceKind.CASE_INPUT,
                        EvidenceKind.SUBMITTED_OUTPUT,
                        EvidenceKind.EXPECTED_BEHAVIOR,
                        EvidenceKind.EXPECTED_OUTPUT,
                    ),
                ),
                budget=InvestigationBudget(max_steps=24),
            ),
            ExpectedBehaviorJudge(name="Expected behavior compliance"),
        ],
    )

    benchmark.cases.add(
        input={
            "task": "Identify and explain every material difference.",
            "source_document": BenchmarkFile(INPUT_FILE),
        },
        output={"reference_report": BenchmarkFile(REFERENCE_FILE)},
        expected_behavior="Identify every material difference and explain its impact.",
    )

    benchmark.publish()
    benchmark.refresh()
    print("Benchmark:", benchmark.id)
    print("Active revision:", benchmark.data.get("activeRevisionId"))
