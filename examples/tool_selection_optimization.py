"""Create and start a complete tool-selection optimization experiment."""

from promptic_sdk import PrompticClient, ToolSelectionTestCase, ToolSelectionTool

tools: list[ToolSelectionTool] = [
    {
        "name": "get_weather",
        "description": "Return the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
test_cases: list[ToolSelectionTestCase] = [
    {"query": "What is the weather in Berlin?", "expected_tool": "get_weather"},
    {"query": "Write a haiku.", "expected_tool": "none"},
]

with PrompticClient() as client:
    experiment = client.create_tool_selection_experiment(
        "YOUR_AI_COMPONENT_ID",
        tools=tools,
        test_cases=test_cases,
        target_model="gpt-4.1-nano",
        tool_source="manual",
        system_prompt="Select a tool only when it is needed.",
        optimize_system_prompt=True,
    )
    client.start_experiment(experiment["id"])
