# Framework architecture

## Components and state

| Component | Responsibility | Implementation |
| --- | --- | --- |
| Coordinator | Interpret the query, create or revise a Plan, choose inspection order, maintain investigation notes | `agentic/service.py`, `agentic/prompts.py` |
| Inspector | Judge the assigned semantic conditions against supplied text and return evidence | `agentic/prompts.py`, `agentic/models.py` |
| Graph engine | Enumerate structural assignments and preserve original graph records | `agentic/graph.py` |
| Workspace | Ground formulas, index shared dependencies, validate receipts, propagate judgments, assemble output | `agentic/workspace.py` |
| Context manager | Assemble role-specific messages, compact feedback, bounded notes, and paginated views | `agentic/context.py` |
| Provider | Enforce native output contracts, record usage, apply request and query limits | `agentic/provider.py`, `agentic/deepseek_strict.py`, `agentic/openrouter.py` |
| Public input and CLI | Validate user graph JSON and run a single query independently of a dataset | `io.py`, `cli.py` |

Paths in the table are relative to `src/masge/`.

The Coordinator's persistent state is explicit query-local state, not an
ever-growing provider conversation. Every request is assembled from the current
question, execution stage, selected notes, and relevant feedback. Every Inspector
delegation receives a fresh context; previous units' conversations are not
carried into it. Accepted receipts remain in the Runtime workspace.

## Plan and structural enumeration

A Plan specifies a graph operation, typed roles, relations, anchor role, fixed
bindings, symmetric role groups, semantic conditions and their positive Boolean
formula, and whether all qualifying assignments must be returned.

| Operation | Structural domain |
| --- | --- |
| `pattern` | Connected, non-induced, injective role bindings, respecting direction, node kinds, fixed nodes, and declared symmetry |
| `shortest_paths` | Every tied globally shortest unweighted path between an anchor and the fixed target, before semantic filtering |
| `anchored_core` | Anchor-containing connected components of a fixed k-core, or the anchor's innermost core, on the weak simple projection |

Parallel edges remain distinct records. Core degree computation excludes
self-loops and collapses parallel adjacency, then restores the original internal
records of the selected component for evidence and output. Pattern roles are
distinct nodes; a pattern relation cannot be a self-loop.

Enumeration must finish before its candidate domain is accepted. Time or
assignment limits produce an explicit failure, never a truncated domain
represented as complete. Name lookup resolves public identity by literal name
matching; it does not establish semantic eligibility.

## Semantic grounding and sharing

A condition has a natural-language phrase, a target (node roles, relation roles,
or a whole path/component), an `each` or `joint` judgment mode, an `all`, `any`,
or `at_least` quantifier, and optional context roles. Conditions are combined
using `condition`, `all`, `any`, and `at_least` formula nodes.

Each structural assignment induces a formula over grounded semantic atoms.
Equivalent output structures can have multiple assignments; a qualifying
assignment can establish membership, while requests for all assignments create
additional output obligations.

An atom's sharing identity includes the condition, ordered bound subjects,
necessary context, source field, and graph hash within the current Plan version.
The same object under a different binding or context need not share a judgment.
Compatible atoms are grouped into immutable inspection units. Reverse dependency
indexes connect units and atoms to candidate membership and output obligations.

For an atom with value `true`, `false`, or `unknown`, the lower/upper truth bounds
are respectively `(1, 1)`, `(0, 0)`, and `(0, 1)`. For a threshold formula with
threshold k, its lower bound is true if at least k children have true lower
bounds; its upper bound is true if at least k children have true upper bounds.
Conjunction and disjunction are the corresponding threshold cases.

The workspace therefore maintains a confirmed set C-minus and a possible set
C-plus. The frontier contains unknown judgments that still affect an unresolved
candidate or a required output assignment, including judgments whose usefulness
depends on later evidence. It does not require immediate candidate reduction.

## Investigation and evidence

The Coordinator chooses an ordered list of units. The Runtime checks relevance
again before each delegation, so a judgment accepted earlier in the same stage
may make a later unit unnecessary. The Inspector receives the unit's full supplied
source text, endpoints, bindings, conditions, and required context.

| Inspector status | Runtime interpretation |
| --- | --- |
| `established` | Positive support; requires a literal source quotation |
| `explicitly_false` | Explicit negative evidence; requires a literal source quotation |
| `not_established` | Reviewed text does not establish the condition; treated as negative support |
| `uncertain` | Unresolved; may block completion |

The Runtime validates atom coverage, object references, and quote spans before
committing the receipt. A malformed receipt cannot partly mutate truth values.
Validation checks evidence provenance and shape, not whether a quote logically
supports the Inspector's interpretation.

The Coordinator can inspect paginated frontier, unit, dependency, receipt,
completion, history, memory, Plan, and guide views. Relevant blocked units can
receive one explicit clarification. Genuine interpretation errors can trigger a
bounded Plan revision; accepting a new Plan clears old semantic state and notes.
Repeated blocked actions without progress eventually return `unresolved`.

## Completion and output

The Runtime completes automatically when:

1. The structural domain has been completely enumerated.
2. Confirmed and possible candidate sets agree.
3. Required output assignments are resolved.
4. No relevant blocking conflict remains.

The result contains every eligible candidate, node IDs, original edge IDs, and
assignment witnesses. A complete empty set has `outcome: "no_solution"`.
`unresolved` is separate and has no valid final answer set.

The result's `certificate` field is a conditional execution-completion record.
It records structural and output completeness relative to the accepted Plan and
receipts; it is not a proof of correct natural-language interpretation.

## Data and execution boundaries

The public interface supplies only graph text, topology, node identity/type,
query, and anchors. Gold answers, expected paths/costs, task labels, and evaluator
conclusions are not framework inputs. There is no query-specific prompt or
label-based operation selection.

Provider calls and reported tokens are counted across Coordinator and Inspector
requests. Token limits are checked before starting another call; the final call
can exceed the cumulative threshold. Each response also has an output-token
limit. Requests have no automatic transport retry; schema repair is an explicit,
bounded additional model call. Timeouts and limits can leave a query unresolved.

`batch.py` and `campaign.py` provide optional bounded-concurrency and persistent
execution-accounting utilities. The public CLI runs one query. `replay.py`
replays recorded native calls against the exact supplied inputs and output
schemas without contacting a provider; it is an auditing utility, not an evaluator.
