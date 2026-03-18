# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "promptic-sdk[langchain]",
#     "langchain>=0.4",
#     "langchain-openai>=0.4",
# ]
# ///
"""Minimal example: LangChain agent traced with Promptic.

Run with:
    uv run --no-project --env-file .env examples/langchain_agent.py

Environment variables:
    OPENAI_API_KEY      - Your OpenAI API key
    PROMPTIC_API_KEY    - Your Promptic API key (from workspace settings)
    PROMPTIC_ENDPOINT   - (optional) defaults to https://promptic.eu
"""

import promptic_sdk

# 1. Initialize Promptic tracing — must be called before any LLM usage.
#    This sets up OpenTelemetry and auto-instruments LangChain + OpenAI.
promptic_sdk.init()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


# 2. Define a simple tool for the agent.
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22°C in {city}."


# 3. Create a LangChain agent with an OpenAI model.
agent = create_agent("openai:gpt-4.1-nano", tools=[get_weather])

# 4. Run the agent — all LLM calls are automatically traced to Promptic.
#    Use ai_component() to link traces to a specific AI Component in your workspace.
with promptic_sdk.ai_component("weather-agent"):
    result = agent.invoke({"messages": [("user", "What's the weather in Vienna?")]})

print(result["messages"][-1].content)
