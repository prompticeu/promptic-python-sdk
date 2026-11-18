## Optimization guidance

Requires a Promptic SDK and server release with optimization-guidance support. Both
`PrompticClient` and `AsyncPrompticClient` expose the same methods (await async calls).

Use experiment rules for instructions such as "Focus on confusion between labels X
and Y; preserve performance on other labels." Rules guide prompt generation and
critique. They do not change evaluator weights or guarantee that the critique
sample contains the requested cases. Context supplied as a rule is not automatically
inserted into the target prompt.

```python
from promptic_sdk import PrompticClient, OptimizationBrief

brief: OptimizationBrief = {
    "rules": ["Focus on confusion between **X** and **Y**. Preserve other labels."],
}
with PrompticClient() as client:
    experiment = client.create_experiment(
        "<component-id>",
        "<available-model-id>",
        task_type="classification",
        initial_prompt="Return one label. Classify {input}",
        optimization_guidance=brief,
    )
    # Editing guidance replaces the experiment's local rules.
    client.update_experiment(experiment["id"], optimization_guidance=brief)
    # Pass {} to remove local rules; inherited component rules still apply.
    client.update_experiment(experiment["id"], optimization_guidance={})
```

`optimization_guidance` is also accepted by `create_tool_selection_experiment`.
To optimize the tool-selection system prompt, enable `optimize_system_prompt`.

### Rules across experiments

```python
with PrompticClient() as client:
    component = client.get_component("<component-id>")
    current = component.get("optimizationRules")
    updated = client.update_component_rules(
        component["id"],
        {"rules": ["Keep the existing label names unchanged."]},
        expected_version=current["version"] if current else 0,
    )
```

Each successful component update increments `optimizationRules.version`. A stale
version returns HTTP 409: fetch the latest component and reconcile edits before
retrying. Pass an empty brief to clear shared rules. Component rules are captured
when scheduling a run; completed runs retain their saved rules. Experiment rules
apply alongside inherited component rules and cannot override them.

### Guidance types and limits

The exported types are `OptimizationBrief`, `ComponentOptimizationRules`, and
`OptimizationGuidance`. Fields inside these dictionaries use API camelCase.

Each scope permits up to 50 nonempty rules of 12,000 characters each. To change
only one part of a brief, retain the other local rules.
`Experiment.optimizationGuidance` includes the saved `contractVersion` and
`componentRules`; those fields are server-managed response metadata, not client
overrides. `Component.optimizationRules` contains the current shared rules
(or null when unset).

### CLI

Save the brief as a JSON object, for example `guidance.json`:

```json
{"rules": ["Focus on confusion between labels X and Y."]}
```

```bash
promptic experiments create -c <component-id> -m <available-model-id> \
  --initial-prompt 'Classify {input}' --guidance guidance.json
promptic experiments update <experiment-id> --guidance guidance.json
promptic experiments create-tool-selection -c <component-id> \
  --tools tools.json --test-cases cases.json --guidance guidance.json
promptic components set-rules <component-id> --guidance guidance.json --version 0
```

Use the component's current rules version for `--version` (zero when unset).
The CLI `--guidance` option replaces the complete brief; a file containing `{}`
clears it. Use `--json` on get commands to inspect effective guidance and versions.
