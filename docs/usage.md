# Usage

## Graph input

`masge validate --graph graph.json` accepts UTF-8 JSON with exactly these fields:

| Object | Required fields |
| --- | --- |
| Root | `domain` (nonempty string), `directed` (Boolean), `nodes` (nonempty array), `edges` (array) |
| Node | `id` (integer), `kind` (nonempty string), `name` (string), `text` (string) |
| Edge | `id` (integer), `source` (node ID), `target` (node ID), `text` (string) |

Node IDs must be unique and dense from zero through N minus one; node array order
does not matter. Edge IDs must be unique nonnegative signed 64-bit integers and
are retained as original record identifiers. Edge endpoints must exist. Extra
fields, including gold answers and task labels, are rejected. Empty text and an
empty edge array are allowed. Node names support literal identity lookup; put
semantic evidence in `text`.

Each edge is stored once with its source and target, including on undirected
graphs. Parallel edges and self-loops are retained. The `directed` field describes
the source graph to the Coordinator; the accepted Plan explicitly chooses
directed or undirected traversal. Core extraction uses weak adjacency.

The exact input bytes receive a SHA256 provenance fingerprint. Edge references
inside Inspector evidence (`edge:0`, etc.) use the edge's zero-based array index.
Final `edge_ids` and assignment `relation_edge_ids` restore the supplied original
IDs. Preserve the original input file when retaining a run for audit.

## Provider configuration

The bundled native adapters accept these explicit configurations:

| Model identifier in this implementation | `--effort` | Credential |
| --- | --- | --- |
| `gpt-5.6-terra`, `gpt-5.6-luna` | `high` or `none` | `OPENAI_API_KEY` |
| `deepseek-v4-flash`, `deepseek-v4-pro` | `high` or `none` | `DEEPSEEK_API_KEY` |
| `google/gemini-3.8-flash` | `low` | `OPENROUTER_API_KEY` |

These are the adapter's supported identifiers, not an assertion of model access
for every account. The OpenRouter adapter pins `google-ai-studio/flex` and disables
route fallback. Native response model and effort are checked against the request.
The same selected model serves both Coordinator and Inspector.

`.env` is loaded from the working directory, with `override=False`. No credentials
are read from a package installation directory. The provider uses its designated
key; it does not substitute another provider's credential.

## Run and limits

```sh
masge run --graph graph.json --query "YOUR QUERY" --anchor 0 --anchor 2 --model deepseek-v4-pro --effort none --output artifacts/run-001 --allow-api
```

At least one public anchor is required. Repeated anchor IDs are deduplicated by
the CLI. The output directory must not already exist. Exit status is zero for
`complete`, one for a recorded `unresolved` result, and two for CLI validation or
file errors.

Use `--limits limits.json` to override any subset of the following defaults:

```json
{
  "max_calls": 160,
  "max_tokens": 400000,
  "provider_seconds": 180,
  "query_seconds": 1800,
  "graph_seconds": 180,
  "max_assignments": 100000,
  "max_revisions": 1,
  "page_size": 40,
  "max_actions": 120,
  "max_output_tokens": 10000
}
```

`max_calls`, `max_tokens`, and `max_actions` may be `null` to disable that
cumulative bound. Timeouts and structural limits remain active. Other numeric
limits must be positive, except `max_revisions`, which can be zero. The cumulative
token bound stops subsequent requests; it cannot cap tokens already consumed by
an in-flight request.

## Result files

| File | Contents |
| --- | --- |
| `run_configuration.json` | Requested model/effort, graph metadata and hash, execution limits |
| `public_input.json` | Query, public anchors, graph schema, limits |
| `result.json` | Completion status, answer structures or failure, execution audit |
| `workspace.json` | Accepted Plan, assignments, shared atoms, receipts, candidate bounds; written once a workspace exists |
| `events.json` | Runtime actions and feedback |
| `reasoning_memory.json` | Query-local notes, latest actions and diagnostics |
| `context_routes.json` | Context modules activated per delegation |
| `calls/` | Per-call structured request, raw response when received, accounting |
| `provider_usage.json` | Aggregate call and reported-token accounting |

Inspect `status` before using `answers`. An empty complete answer and an unresolved
execution have different meanings. The audit's protocol and completion flags do
not measure semantic accuracy. Run artifacts contain the supplied question and
graph evidence; `artifacts/` is excluded from Git.

The Python `Solver.run` method returns the result dictionary and saves runtime
traces when an output directory is supplied. The CLI additionally writes
`run_configuration.json` and `result.json`.

## Offline replay

`masge.agentic.replay.ReplayProvider` implements the same provider protocol. Pass
a recorded run directory, the original model/effort, and the original limits,
then run a fresh Solver with the same graph, query, anchors, and schema.
Each requested call must match its recorded role, input, schema, model, native
route, and response accounting. Check that all recorded calls were consumed and
compare final results excluding wall-clock duration. An incomplete transcript
cannot supply a missing response, and replay does not establish semantic accuracy.
