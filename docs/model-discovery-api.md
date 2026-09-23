# Model discovery for external coding agents

The SDK and CLI model-discovery methods require the platform endpoint described
below. Earlier platform releases have no model-discovery endpoint. The dashboard's
benchmark evaluator Judge model picker uses a session-backed model selector;
an external agent with an API key cannot reliably reconstruct its choices from
a static list or a provider's catalog.

## Platform prerequisite

Deploy the read-only `GET /api/v1/models` endpoint and its OpenAPI schema.
Authenticate with the same bearer API key used by other SDK endpoints;
resolve the AI Application from that key. Session-token callers should use the
existing `X-AI-Application-Id` scope. Do not accept an arbitrary application ID
from an API-key caller.

Return the effective available models for that AI Application, not every model
known to the platform. Apply its configured provider access, disabled models,
and other availability checks. Each model must also indicate whether it can be
selected in the benchmark Evaluation → Edit evaluator → Judge model picker.
Derive that flag from the selector's server-side eligibility logic, including
its OpenAI-only group restriction.

Response:

```json
{
  "data": [
    {
      "id": "<canonical model ID>",
      "name": "<display name>",
      "provider": "<provider display name>",
      "group": "openai",
      "region": "eu",
      "judgeEligible": true
    }
  ]
}
```

`data` contains only models available to the AI Application. IDs must be the
exact values accepted in model configuration. `group` is one of `openai`,
`platform`, `google`, `custom`, or `openrouter`. `region` is optional.
`judgeEligible` identifies the subset accepted when configuring a benchmark
judge evaluator. External agents should call `client.models.list()` and use
only records with `judgeEligible == True` for an explicit benchmark judge
model. If the judge model is omitted, the platform uses its default. An
explicit model outside that subset receives a 400 configuration error once
platform validation is deployed. The endpoint does not return the default
judge model.

Return `401` for missing/invalid credentials and `404` when the scoped AI
Application is not found. Follow the REST API's normal error response shape.

The platform should test that this endpoint and the dashboard selector agree
for (at least) disabled models and missing provider configuration. The SDK
provides sync/async `client.models.list()` and `promptic models list --json`.
These calls will return a platform API error until the platform route is
deployed. The SDK does not cache or hardcode model IDs.
