# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "promptic-sdk[langchain]",
#     "langchain>=0.4",
#     "langchain-openai>=0.4",
# ]
# ///
"""Example: Run an agent with dataset tagging for evaluation.

This script shows how to:
1. Tag agent runs with a dataset name for automatic dataset creation
2. Run multiple test inputs to build an evaluation dataset
3. Use the CLI to analyze agent performance

Run with:
    uv run --no-project --env-file .env examples/agent_evaluation.py

After running, evaluate via CLI:
    promptic evaluations run <COMPONENT_ID> --dataset <DATASET_ID> --json

Environment variables:
    OPENAI_API_KEY      - Your OpenAI API key
    PROMPTIC_API_KEY    - Your Promptic API key (from workspace settings)
    PROMPTIC_ENDPOINT   - (optional) defaults to https://promptic.eu
"""

import os

import promptic_sdk

promptic_sdk.init()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

COMPONENT_NAME = os.environ.get("PROMPTIC_COMPONENT_NAME", "Agent")
DATASET_NAME = os.environ.get("PROMPTIC_DATASET_NAME", "baseline")
RUN_NAME = os.environ.get("PROMPTIC_RUN_NAME", None)
MODEL_NAME = os.environ.get("PROMPTIC_MODEL_NAME", "openai:gpt-4.1-nano")
SYSTEM_PROMPT = os.environ.get(
    "PROMPTIC_SYSTEM_PROMPT",
    "You are a weather assistant. Use the weather tool only when it is needed to answer "
    "the user. Call each tool at most once per city, then answer directly and concisely.",
)


# 2. Define tools for the agent.
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22°C in {city}."


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "vienna": "1.9 million",
        "prague": "1.3 million",
        "berlin": "3.7 million",
    }
    return populations.get(city.lower(), f"Unknown population for {city}")


# 3. Create a LangChain agent with an OpenAI model.
agent = create_agent(MODEL_NAME, system_prompt=SYSTEM_PROMPT, tools=[get_weather, get_population])

# 4. Test inputs to evaluate the agent on.
test_queries = [
    "What's the weather in Vienna?",
    "What's the weather in Prague?",
    "Compare the weather in Berlin and Vienna.",
    "What's the weather in Prague?",
]


#    All traces are automatically linked to the "eval-round-1" dataset.
#    Option A: dataset parameter on ai_component (simplest)

with promptic_sdk.ai_component(COMPONENT_NAME, dataset=DATASET_NAME, run=RUN_NAME):
    for query in test_queries:
        result = agent.invoke({"messages": [("user", query)]})
        print(f"Q: {query}")
        print(f"A: {result['messages'][-1].content}\n")
