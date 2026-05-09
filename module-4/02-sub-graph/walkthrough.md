# Walkthrough: Sub-graphs

> **Notebook:** `sub-graph.ipynb`
> **Goal:** Compose graphs out of **smaller graphs**. Give each piece its own state, run them in parallel as nodes of a parent graph, and use overlapping state keys as the communication channel.

---

## The Big Picture

A **sub-graph** is just a compiled `StateGraph` that you plug into another graph as if it were a single node:

```python
parent.add_node("question_summarization", qs_builder.compile())
parent.add_node("failure_analysis",       fa_builder.compile())
```

Why bother? Three reasons:

1. **Encapsulation** — each team / agent / sub-task gets *its own* state schema and *its own* internal logic, isolated from the rest.
2. **Reusability** — a sub-graph compiled once can be dropped into many parents.
3. **Parallelism** — sub-graphs can run side by side as parallel nodes (super-step rules from notebook 01 still apply).

The mental rule for communication is short:

> **Sub-graphs talk to their parent through overlapping state keys.**
> If both schemas declare a key, it's shared. Anything else is private to the sub-graph.

```
parent state:    raw_logs, cleaned_logs, fa_summary, report, processed_logs
                                ↓ shared with both
sub-graph 1:     cleaned_logs, failures, fa_summary, processed_logs
sub-graph 2:     cleaned_logs, qs_summary, report, processed_logs
```

`cleaned_logs` flows **from** parent **into** each sub-graph. `fa_summary` and `report` flow **back up**. `processed_logs` is written by **both sub-graphs in parallel** — and that's exactly the situation that requires a reducer.

> ### 📌 Reducers — the rule of this module
>
> *A reducer controls how a new value is merged into the existing value for a key. You need one when you either want accumulation, or when multiple nodes write the same key concurrently and you can't afford to lose any write.*
>
> In this notebook the reducer that *must* exist is `processed_logs: Annotated[List[int], add]` on the parent state. Both sub-graphs run in parallel and both write `processed_logs`. Without `add`, you'd hit `INVALID_CONCURRENT_GRAPH_UPDATE` and lose one sub-graph's work.

---

## 1. The Toy Scenario

The notebook models a small log-processing pipeline:

- A system receives `raw_logs`.
- One sub-graph performs **failure analysis** — find logs with a failing grade and summarize them.
- A second sub-graph performs **question summarization** — summarize what users were asking about, then ship a "report" to Slack.
- Both run **in parallel** after a shared `clean_logs` step.

We start by defining the log shape:

```python
from typing import List, Optional
from typing_extensions import TypedDict

class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[str]
    grader: Optional[str]
    feedback: Optional[str]
```

Optional fields = "may or may not be present". Failure analysis distinguishes failures from non-failures by checking whether `grade` is set.

---

## 2. Sub-graph #1: Failure Analysis

Two state schemas — one for the **internal** state, one for the **output** the parent sees:

```python
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]
```

The split matters: `failures` is a scratchpad — useful inside the sub-graph, irrelevant to the parent. By giving the builder an explicit `output_state`, only `fa_summary` and `processed_logs` leak upward.

```python
def get_failures(state):
    """Get logs that contain a failure."""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """Generate a summary of the failures."""
    failures = state["failures"]
    fa_summary = "Poor Quality retrieval of chroma documentation."   # placeholder for an LLM call
    return {
        "fa_summary": fa_summary,
        "processed_logs": [f"failure-analysis-on-log-{f['id']}" for f in failures],
    }

fa_builder = StateGraph(state_schema=FailureAnalysisState,
                        output_state=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)
```

The sub-graph is a *normal* LangGraph — START, nodes, END. Nothing about the API changes when it ends up nested inside another graph.

---

## 3. Sub-graph #2: Question Summarization

Same shape, different work. Two nodes, plus an output schema that drops the internal scratchpad (`qs_summary`):

```python
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = "Questions focused on usage of ChatOllama and Chroma VectorDB"   # placeholder
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs],
    }

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    report = "foo bar baz"   # placeholder
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState,
                        output_schema=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)
```

`qs_summary` is the in-flight summary the second node consumes. It never makes it to the parent because the output schema doesn't list it.

---

## 4. The Parent Graph — First (Naive) Attempt

```python
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add]   # ← do we really need a reducer here?
    fa_summary: str
    report: str
    processed_logs: Annotated[List[int], add]
```

This is the *first* version the notebook shows, with `cleaned_logs` annotated with the `add` reducer. The question the notebook then asks is great:

> Why does `cleaned_logs` have a reducer if it only goes **into** each sub-graph as input? It is not modified.

**The answer** is the subtle gotcha of nested graphs. Even when a sub-graph "doesn't change" a shared key, by default its output state still **includes that key**. So when the two sub-graphs run in parallel and both pass `cleaned_logs` back up — even unmodified — the parent sees two concurrent writes to the same key. That triggers `INVALID_CONCURRENT_GRAPH_UPDATE` unless `cleaned_logs` has a reducer.

So the naive fix is to slap `add` on `cleaned_logs` too — which technically works, at the cost of duplicating the data (the reducer concatenates two identical copies).

The cleaner fix is the second version below.

---

## 5. The Parent Graph — Cleaner Version

We've already designed each sub-graph with an **explicit output schema** that drops `cleaned_logs`. So the sub-graphs no longer publish `cleaned_logs` at all, and the parent doesn't have to reduce it:

```python
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]                          # no reducer — single writer (`clean_logs`)
    fa_summary: str                                  # only failure-analysis writes this
    report: str                                      # only question-summarization writes this
    processed_logs: Annotated[List[int], add]        # both sub-graphs write — REDUCER REQUIRED

def clean_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs           # placeholder data-cleaning step
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis",       fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()
```

`add_node("question_summarization", qs_builder.compile())` — the compiled sub-graph drops in **as a node**. From the parent's perspective there is no difference between a regular function-node and a sub-graph-node.

### Reducer audit on this state

| Key | Writers | Reducer needed? | Why |
|-----|---------|-----------------|-----|
| `raw_logs` | the caller, once | No | Single writer at `invoke()` |
| `cleaned_logs` | `clean_logs` only | No | Single writer per step |
| `fa_summary` | `failure_analysis` only | No | Only one sub-graph writes it |
| `report` | `question_summarization` only | No | Only one sub-graph writes it |
| `processed_logs` | **both** sub-graphs in parallel | **Yes** | Concurrent writes → reducer required |

This is exactly the rule from the callout: *a reducer is needed when the value should accumulate, or when multiple nodes write the same key concurrently*. `processed_logs` hits the second trigger. Everything else has at most one writer per step.

### Visualizing nested graphs

```python
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

`xray=1` expands sub-graphs in the diagram instead of showing them as opaque blocks. Useful for debugging.

---

## 6. Running It End-to-End

```python
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)
question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

graph.invoke({"raw_logs": [question_answer, question_answer_feedback]})
```

Result (trimmed):

```python
{
  'raw_logs': [...],
  'cleaned_logs': [...],
  'fa_summary': 'Poor Quality retrieval of chroma documentation.',
  'report': 'foo bar baz',
  'processed_logs': [
      'failure-analysis-on-log-2',     # from failure-analysis sub-graph
      'summary-on-log-1',              # from question-summarization sub-graph
      'summary-on-log-2',              # from question-summarization sub-graph
  ],
}
```

The `processed_logs` list is the **merge** of both sub-graphs' outputs — exactly what the `add` reducer was there for. If you delete that reducer, you'll see the run crash with the concurrent-update error from notebook 01.

---

## 7. Communication Patterns Between Parent and Sub-graph

There are two ways to compose state across nested graphs. The notebook uses pattern A.

### A. Shared schemas (overlapping keys)

What the notebook does. Each schema independently declares the keys it cares about. LangGraph matches them by **name**.

- Pros: no boilerplate adapter code; each sub-graph stays usable on its own.
- Cons: keys must match exactly across schemas; renaming requires touching every level.

### B. Wrapper node that translates state

When the sub-graph's schema doesn't naturally line up with the parent's, wrap the call in a node:

```python
def call_sub_graph(state: ParentState) -> dict:
    sub_input = {"some_key": state["different_name"]}
    sub_output = sub_graph.invoke(sub_input)
    return {"different_name": sub_output["some_key"]}
```

- Pros: full decoupling — sub-graph internals can change without rippling.
- Cons: extra adapter code to maintain.

Pattern A is the default; reach for B when reusing a sub-graph across very different parents.

---

## When to Reach for Sub-graphs

| Situation | Why a sub-graph helps |
|-----------|----------------------|
| Multi-agent system, each agent has its own scratchpad | Agent state stays private to its sub-graph |
| The same workflow is reused in two different contexts | Compile once, drop in as a node anywhere |
| A complex pipeline you want to test in isolation | Sub-graph runs standalone — easy unit testing |
| Heterogeneous parallel work (different schemas per branch) | Each branch owns its schema; parent only sees the overlap |

If your "sub-graph" has the **same schema** as the parent and isn't reused, a regular function-node is simpler.

---

## Key Takeaways

1. **A sub-graph is a compiled `StateGraph` used as a node.** API-wise, nothing else changes.
2. **Communication is via overlapping state keys** — declare the same key name in both schemas and it's shared.
3. **Each sub-graph can have its own state schema and its own output schema.** The output schema controls *which* keys leak back up to the parent.
4. **Sub-graphs run in parallel just like any other parallel nodes** — super-step rules from notebook 01 apply unchanged.
5. **Concurrent writes to the same parent key require a reducer** — this notebook needs `Annotated[List[int], add]` on `processed_logs` because both sub-graphs publish it.
6. **Use an output schema to *avoid* needing a reducer on a key.** If a sub-graph doesn't publish a key, the parent never sees a concurrent write to it (cleaner than reducing duplicate writes away).
7. **`graph.get_graph(xray=1)`** expands nested sub-graphs in the diagram so you can see the full topology.
