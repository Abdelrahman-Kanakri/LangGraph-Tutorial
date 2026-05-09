# Module 4 — Controllability

Tools for **shaping how a graph executes**: running nodes side by side, composing graphs out of smaller graphs, and merging concurrent writes safely. The throughline is *reducers* — the rule that lets multiple writers share a state key without losing data.

---

## 📌 Reducers — the cross-cutting concept of this module

*A reducer controls how a new value is merged into the existing value for a key. You need one when you either want accumulation, or when multiple nodes write the same key concurrently and you can't afford to lose any write.*

| Default behavior | When | What happens |
|------------------|------|--------------|
| Overwrite (last write wins) | Single writer per super-step | Fine — no reducer needed |
| **Hard error** (`INVALID_CONCURRENT_GRAPH_UPDATE`) | Multiple writers in the *same* super-step | Reducer required to define a merge |

Attach with `Annotated[<value_type>, <reducer_fn>]` inside your `TypedDict`. Common choices: `operator.add` (list concat), `add_messages` (append + id-overwrite + `RemoveMessage`), or any custom 2-arg `(existing, update) -> merged` function.

---

## 01 · parallelization

**Notebook:** [parallelization.ipynb](../module-4/01-parallelization/parallelization.ipynb) · **Walkthrough:** [walkthrough.md](../module-4/01-parallelization/walkthrough.md)

**What it teaches**
**Fan-out** (one node → many in parallel) and **fan-in** (many → one). Why concurrent writes to a shared key require a reducer, and how to control the order of those writes.

**Key APIs**

- `builder.add_edge("a", "b")` + `builder.add_edge("a", "c")` — fan-out from `a`
- `builder.add_edge(["b", "c"], "d")` — fan-in barrier; `d` waits for both
- `Annotated[list, operator.add]` — concatenation reducer for parallel list writes
- Custom reducer pattern: `def reducer(left, right) -> merged: ...` attached via `Annotated`
- Returning `{"key": [single_value]}` from each parallel writer so the reducer concatenates contributions

**Critical concept**
A **super-step** is one parallel batch. Within a super-step, write order is deterministic but not user-controlled. Use a custom reducer (e.g., `sorting_reducer`) or a sink node if order matters.

**The error you'll see without a reducer**

```
At key 'state': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

This is `INVALID_CONCURRENT_GRAPH_UPDATE` — a *feature*, not a bug. LangGraph refuses to silently drop one writer's data.

**When to use**
Independent I/O (web search + Wikipedia + RAG retrieval), parallel calls to multiple analysts/critics, anything where wall-clock latency = slowest branch instead of sum of branches.

---

## 02 · sub-graph

**Notebook:** [sub-graph.ipynb](../module-4/02-sub-graph/sub-graph.ipynb) · **Walkthrough:** [walkthrough.md](../module-4/02-sub-graph/walkthrough.md)

**What it teaches**
Compose graphs out of **other compiled graphs**. Each sub-graph can carry its own state schema and its own internal nodes; communication with the parent happens through **overlapping state keys**.

**Key APIs**

- `parent.add_node("name", sub_builder.compile())` — drop a compiled sub-graph in as a node
- `StateGraph(state_schema=..., output_state=...)` — sub-graph internal state vs. what leaks up to the parent
- `Annotated[List[int], add]` on a parent key written by **multiple sub-graphs in parallel** — required reducer
- `graph.get_graph(xray=1).draw_mermaid_png()` — render with sub-graphs expanded

**Communication pattern**

```
parent state:    raw_logs, cleaned_logs, fa_summary, report, processed_logs
                                ↓ shared
sub-graph A:     cleaned_logs, failures, fa_summary, processed_logs
sub-graph B:     cleaned_logs, qs_summary, report, processed_logs
```

Keys with the same name in both schemas are shared. Anything else is private to the sub-graph.

**The reducer in this notebook**
`processed_logs` is written by **both** sub-graphs in parallel — this is the textbook concurrent-write case. Without `Annotated[List[int], add]` on the parent state, the run crashes with `INVALID_CONCURRENT_GRAPH_UPDATE`.

**Two ways to avoid duplicate-merge headaches**

1. **Output schema per sub-graph** — sub-graph only "publishes" the keys you actually want shared upward. Keeps the parent's reducer surface small.
2. **Reducer on every shared key** — works, but concatenates duplicates when sub-graphs pass through unchanged values.

**When to use**
Multi-agent systems where each agent has its own scratchpad; reusable workflows that drop into multiple parents; pipelines you want to unit-test in isolation.

---

## 03 · map-reduce

**Notebook:** [map-reduce.ipynb](../module-4/03-map-reduce/map-reduce.ipynb) · **Walkthrough:** [walkthrough.md](../module-4/03-map-reduce/walkthrough.md)

**What it teaches**
**Dynamic** fan-out with `Send()`: spawn one parallel branch per item in a list whose length isn't known until runtime, then reduce all their outputs back into a single value. The textbook concurrent-write pattern.

**Key APIs**

- `from langgraph.types import Send` — the dispatch primitive
- `Send(target_node, payload)` — one parallel invocation of `target_node`, with a custom per-branch payload
- `add_conditional_edges(source, fn, [possible_targets])` — the third arg is **required** when `fn` returns `Send` objects; LangGraph can't otherwise validate the targets
- `Annotated[list, operator.add]` on the key the map branches write — the reducer that merges N concurrent writes
- `with_structured_output(PydanticModel)` — used here to constrain the LLM's "list of subjects" and "best joke index" outputs

**The map-reduce shape**

```
generate_topics  ──→  Send → generate_joke(s_1) ─┐
                      Send → generate_joke(s_2) ─┤── jokes (REDUCER)  ──→  best_joke
                      Send → generate_joke(s_N) ─┘
```

**The reducer in this notebook**
`jokes: Annotated[list, operator.add]`. N parallel `generate_joke` branches all write `jokes` in the same super-step — without `operator.add`, the run hard-errors and no joke is kept.

**Send() vs static fan-out**

| | Static fan-out (01) | `Send()` (03) |
|---|---|---|
| Branch count | Compile-time, fixed | Runtime, data-driven |
| Per-branch payload | Same state for all | Custom payload each |
| Branch schema | Parent state schema | Can be a private sub-schema |

**When to use**
Whenever the number of parallel calls comes from data the LLM (or upstream code) just produced: one critic per analyst, one summary per chunk, one tool call per query, etc.

---

## Module 4 quick decision tree

```
Need to do work in parallel?
   ├── independent nodes inside one graph, count fixed → fan-out / fan-in   (notebook 01)
   ├── self-contained pipelines with their own state   → sub-graphs         (notebook 02)
   └── same operation over a runtime list of N items   → Send() map-reduce  (notebook 03)

About to add a key to your state schema?
   ├── one writer per step, replace is fine             → no reducer
   ├── one writer per step, value should accumulate     → reducer (operator.add / add_messages / custom)
   └── multiple writers in the same step                → reducer REQUIRED
```

If the second column says "REQUIRED" and you don't add one, the graph will hard-error at runtime — not silently drop data. Treat that error as a signpost telling you which key is missing its merge rule.
