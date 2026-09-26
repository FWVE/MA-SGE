# MA-SGE

A framework for answering natural-language queries over text-rich graphs.

- **Coordinator** interprets the query and decides what to inspect.
- **Inspector** evaluates semantic conditions against node and edge text.
- **Runtime** finds structural candidates, reuses shared judgments, and returns the matching graphs.

Supports pattern matching, shortest paths, and anchored core extraction.

## Setup

Python 3.11:

```sh
python -m pip install .
```

Copy `.env.example` to `.env` and set your provider key, for example `DEEPSEEK_API_KEY`.

## Usage

Provide a graph as JSON:

```json
{
  "domain": "my_graph",
  "directed": true,
  "nodes": [
    {"id": 0, "kind": "entity", "name": "A", "text": "Node text"},
    {"id": 1, "kind": "entity", "name": "B", "text": "Node text"}
  ],
  "edges": [{"id": 0, "source": 0, "target": 1, "text": "Edge text"}]
}
```

Node IDs must be consecutive integers starting at zero; edge IDs must be unique.

```sh
masge run --graph graph.json --query "Your question" --anchor 0 --model deepseek-v4-pro --effort none --output artifacts/run-001 --allow-api
```

Results are saved to `artifacts/run-001/result.json`. Use a new output directory
for each run. Run `masge run --help` for options.

Core implementation: [`src/masge/agentic`](src/masge/agentic).
