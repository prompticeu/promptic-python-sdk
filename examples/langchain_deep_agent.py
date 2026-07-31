# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "promptic-sdk[langchain]",
#     "deepagents",
#     "langchain>=0.4",
#     "langchain-openai>=0.4",
# ]
# ///
"""Example: Deep agent with subagents, traced with Promptic.

This script shows how to:
1. Define specialist subagents with their own tools
2. Use `create_deep_agent` so the main agent can spawn subagents via the `task()` tool
3. Trace the full multi-agent interaction with Promptic

Scenario:
  A user asks a travel planning question. The deep agent delegates research
  to a "travel-researcher" subagent (weather, flights, hotels) and budget
  calculations to a "budget-analyst" subagent.

Run with:
    uv run --no-project --env-file .env examples/langchain_deep_agent.py

Environment variables:
    OPENAI_API_KEY      - Your OpenAI API key
    PROMPTIC_API_KEY    - Your Promptic API key (from workspace settings)
    PROMPTIC_ENDPOINT   - (optional) defaults to https://promptic.eu
"""

import json

import promptic_sdk

# 1. Initialize Promptic tracing — must be called before any LLM usage.
promptic_sdk.init()

from deepagents import SubAgent, create_deep_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Define tools for the travel-researcher subagent.
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a city."""
    forecasts = {
        "paris": "Partly cloudy, 18°C, light breeze from the west.",
        "tokyo": "Sunny, 24°C, clear skies expected all week.",
        "new york": "Overcast, 12°C, chance of rain in the afternoon.",
        "london": "Rainy, 10°C, heavy showers until evening.",
    }
    return forecasts.get(city.lower(), f"Weather data unavailable for {city}.")


@tool
def search_flights(origin: str, destination: str) -> str:
    """Search for available flights between two cities."""
    flights = [
        {"airline": "SkyLine", "departure": "08:30", "arrival": "11:45", "price": 320},
        {"airline": "AeroConnect", "departure": "14:00", "arrival": "17:20", "price": 275},
    ]
    return json.dumps({"route": f"{origin} → {destination}", "flights": flights})


@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """Search for available hotels in a city for given dates."""
    hotels = [
        {"name": "Grand Plaza", "rating": 4.5, "price_per_night": 180},
        {"name": "City Comfort Inn", "rating": 4.0, "price_per_night": 95},
    ]
    return json.dumps({"city": city, "dates": f"{checkin} to {checkout}", "hotels": hotels})


# ---------------------------------------------------------------------------
# 3. Define tools for the budget-analyst subagent.
# ---------------------------------------------------------------------------


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies."""
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.27,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()), 1.0)
    converted = round(amount * rate, 2)
    return f"{amount} {from_currency} = {converted} {to_currency}"


@tool
def calculate_trip_budget(
    flight_cost: float, hotel_per_night: float, nights: int, daily_expenses: float
) -> str:
    """Calculate total estimated trip budget."""
    total = flight_cost + (hotel_per_night * nights) + (daily_expenses * nights)
    return json.dumps(
        {
            "flight": flight_cost,
            "hotel": hotel_per_night * nights,
            "daily_expenses": daily_expenses * nights,
            "total": total,
        }
    )


# ---------------------------------------------------------------------------
# 4. Define the subagents.
# ---------------------------------------------------------------------------

travel_researcher: SubAgent = {
    "name": "travel-researcher",
    "description": (
        "Researches travel destinations: looks up weather forecasts, "
        "searches for flights and hotels. Use for any destination research."
    ),
    "system_prompt": (
        "You are a travel research assistant. Use your tools to gather "
        "weather, flight, and hotel information, then return a concise "
        "research summary with specific data points."
    ),
    "tools": [get_weather, search_flights, search_hotels],
    "model": "openai:gpt-4.1-nano",
}

budget_analyst: SubAgent = {
    "name": "budget-analyst",
    "description": (
        "Calculates trip budgets and converts currencies. Use after "
        "gathering travel research to estimate total costs."
    ),
    "system_prompt": (
        "You are a budget analyst for travel planning. Use your tools to "
        "calculate trip costs and convert currencies. Return a clear "
        "budget breakdown."
    ),
    "tools": [convert_currency, calculate_trip_budget],
    "model": "openai:gpt-4.1-nano",
}

# ---------------------------------------------------------------------------
# 5. Create the deep agent with subagents.
#    The main agent can spawn subagents via the built-in `task()` tool.
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    "openai:gpt-4.1-mini",
    system_prompt=(
        "You are a travel planning assistant. For each user request:\n"
        "1. Delegate destination research to the travel-researcher subagent\n"
        "2. Once you have research results, delegate budget estimation to the budget-analyst subagent\n"
        "3. Combine both results into a helpful travel recommendation"
    ),
    subagents=[travel_researcher, budget_analyst],
)

# ---------------------------------------------------------------------------
# 6. Run the deep agent — all LLM calls (main agent + subagents) are
#    automatically traced to Promptic.
# ---------------------------------------------------------------------------

test_queries = [
    "I want to plan a 3-night trip to Tokyo from New York. What's the weather like and how much will it cost?",
    "Plan me a weekend in Paris flying from London. I'd budget about $100/day for food and activities.",
]

with promptic_sdk.ai_component("travel-planner", dataset="travel-queries", run="optimized"):
    for query in test_queries:
        print(f"User: {query}")
        result = agent.invoke({"messages": [("user", query)]})
        print(f"Assistant: {result['messages'][-1].content}")
        print("-" * 80)
