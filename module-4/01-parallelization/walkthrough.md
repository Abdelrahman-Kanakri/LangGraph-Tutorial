# Walkthrough: Parallelization (Fan-out / Fan-in)

> **Notebook:** `parallelization.ipynb`
> **Goal:** Run multiple nodes **in the same step** by fanning out from one node to several, then fanning back in. Learn why this *requires* a reducer, and how to control the order of concurrent writes.

---

## The Big Picture

So far every graph in this tutorial has been **linear** — one node finishes, the next starts. That's wasteful when independent work could run side by side: hitting Wikipedia and a web search in parallel, calling N analysts at once, mapping a batch of items.

LangGraph's answer is **fan-out**: from one node, draw an edge to multiple downstream nodes. They will execute **concurrently in the same super-step**. To merge their results back together you draw multiple edges into a single fan-in node.

```
        ┌── b ──┐
   a ──┤        ├── d
        └── c ──┘
```

The catch lives at the merge point: when `b` and `c` both write the same state key in the same step, LangGraph needs a rule for **how to combine** their writes. That rule is a **reducer**.

> ### 📌 Reducers — the rule of this module
>
> *A reducer controls how a new value is merged into the existing value for a key. You need one when you either want accumulation, or when multiple nodes write the same key concurrently and you can't afford to lose any write.*
>
> Without a reducer, two concurrent writes to the same key raise `INVALID_CONCURRENT_GRAPH_UPDATE`. The reducer turns that error into a defined merge.

---

## 1. Setup: A Linear Baseline

The notebook builds up incrementally. The starting point is a four-node linear graph that **overwrites** state at every step:

```python
from typing import Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # Note: no reducer. Default behavior is overwrite.
    state: List[str]

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}     # ← returns a list of length 1

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```

`ReturnNodeValue` is a tiny callable class so we can stamp four different node "secrets" without writing four near-identical functions.

**Run it:**

```python
graph.invoke({"state": []})
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm B"]   ← B was overwritten
# Adding I'm D to ["I'm C"]
# {'state': ["I'm D"]}
```

Because there is **no reducer** on `state`, every node's `return {"state": [...]}` *replaces* the previous list. Each node sees only what the immediately preceding node wrote. The final value is whatever `d` wrote: `["I'm D"]`. This is the default LangGraph behavior — **last write wins**.

---

## 2. Fan-out Without a Reducer → Error

Now we change the topology so `b` and `c` both run **after** `a`, and both feed into `d`:

```python
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
```

```
        ┌── b ──┐
   a ──┤        ├── d
        └── c ──┘
```

Same `State` (no reducer). Run it:

```python
from langgraph.errors import InvalidUpdateError
try:
    graph.invoke({"state": []})
except InvalidUpdateError as e:
    print(f"An error occurred: {e}")
```

```
Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
An error occurred: At key 'state': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

This is the **`INVALID_CONCURRENT_GRAPH_UPDATE`** error. `b` and `c` ran in parallel — both wrote `state` in the same super-step — and LangGraph refuses to silently pick a winner.

> **The mental model:** *one writer per step, no reducer needed. Many writers per step, reducer required.* The error is a feature: it stops your graph from silently dropping writes you cared about.

---

## 3. Adding `operator.add` as a Reducer

The fix is to declare how parallel writes merge. For lists, the simplest reducer is **`operator.add`**, which is plain list concatenation:

```python
import operator
from typing import Annotated

class State(TypedDict):
    state: Annotated[list, operator.add]    # ← reducer attached
```

Same graph, same nodes — only the schema changed. Re-run:

```python
graph.invoke({"state": []})
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm A"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C"]
# {'state': ["I'm A", "I'm B", "I'm C", "I'm D"]}
```

Now both parallel writes are *kept* and concatenated. Notice `d` sees the merged list — `["I'm A", "I'm B", "I'm C"]` — before adding its own value.

**Two things changed by attaching the reducer:**

1. The error went away. Concurrent writes now have a defined merge rule.
2. Single-writer behavior also flipped — `a → state = ["I'm A"]`, then `b → state = ["I'm A", "I'm B"]`. The reducer applies to *every* write to that key, not just concurrent ones.

`Annotated[list, operator.add]` is the smallest possible reducer pattern. The full signature is `Annotated[<value_type>, <merge_fn>]` — the merge function takes `(existing_value, new_value)` and returns the combined value.

---

## 4. Waiting for Slower Branches (Super-step Synchronization)

What if one fork has more work than the other?

```python
builder.add_edge(START, "a")
builder.add_edge("a", "b")     # short branch:
builder.add_edge("a", "c")     # long branch:
builder.add_edge("b", "b2")    #   b → b2
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
```

```
        ┌── b ── b2 ──┐
   a ──┤              ├── d
        └─────── c ───┘
```

`add_edge(["b2", "c"], "d")` — the list form means "wait for *both* `b2` and `c` to finish before running `d`". This is a **fan-in barrier**.

Run it:

```python
graph.invoke({"state": []})
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm A"]
# Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C", "I'm B2"]
```

**What just happened:**

| Super-step | Concurrent writers |
|------------|--------------------|
| 1 | `a` |
| 2 | `b`, `c` (parallel) |
| 3 | `b2` (only — `c` already done) |
| 4 | `d` |

Inside super-step 2, `b` and `c` ran together. `b2` runs *after* its parent `b`, in super-step 3. Super-step 4 only fires once **everything** that feeds `d` is done — that's how `add_edge([...], "d")` enforces "wait for all".

> **Super-step rule:** every super-step is one parallel batch. The graph waits for the entire batch to finish before starting the next.

---

## 5. Custom Reducer to Order Concurrent Writes

Inside a single super-step the order of writes is **deterministic but not user-controlled** — LangGraph picks an order based on graph topology that you can't directly influence. That's why above, `c` came before `b2` even though intuitively `b2` is "later".

If the order matters (say you want sorted output), write a custom reducer:

```python
def sorting_reducer(left, right):
    """Combines and sorts the values in a list."""
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right, reverse=False)

class State(TypedDict):
    state: Annotated[list, sorting_reducer]
```

Same fan-out/fan-in graph as section 4, just a different reducer:

```python
graph.invoke({"state": []})
# {'state': ["I'm A", "I'm B", "I'm B2", "I'm C", "I'm D"]}
```

Now the final list is alphabetically sorted because the reducer **always sorts** the merged list, regardless of which write arrived first.

**Three other patterns the docs mention** when sorting-globally is too crude:

1. Have parallel nodes write to a **separate scratch key** (so they don't collide with downstream writes).
2. Use a dedicated **sink node** after the parallel step to combine and order outputs deterministically.
3. **Clear the temporary field** in the sink so it doesn't leak.

---

## 6. A Realistic Example: Parallel Web + Wikipedia Lookup

Now the toy graph is replaced with something useful. Goal: answer a question by gathering context from **Wikipedia and a web search in parallel**, then handing both to an LLM.

### State

```python
from typing_extensions import TypedDict
from typing import Annotated
import operator

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]    # accumulates from both sources
```

`question` and `answer` have no reducer (single writer each, overwrite is fine). `context` *must* have one — Wikipedia and web search both push into it concurrently.

### Nodes

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-medium", temperature=0)

def search_web(state):
    """Retrieve docs from web search."""
    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state['question']})
    search_docs = data.get("results", data)
    formatted = "\n\n---\n\n".join(
        [f'<Document href="{d["url"]}">\n{d["content"]}\n</Document>' for d in search_docs]
    )
    return {"context": [formatted]}

def search_wikipedia(state):
    """Retrieve docs from wikipedia."""
    search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()
    formatted = "\n\n---\n\n".join(
        [f'<Document source="{d.metadata["source"]}" page="{d.metadata.get("page", "")}">\n{d.page_content}\n</Document>' for d in search_docs]
    )
    return {"context": [formatted]}

def generate_answer(state):
    """Node to answer a question."""
    template = "Answer the question {question} using this context: {context}"
    instructions = template.format(question=state["question"], context=state["context"])
    answer = llm.invoke([SystemMessage(content=instructions),
                         HumanMessage(content="Answer the question.")])
    return {"answer": answer}
```

Each retrieval node returns `{"context": [formatted]}` — a **list of length 1**. `operator.add` then concatenates them into a 2-element list at the merge point. This is the standard pattern: *each parallel writer returns a list with its own contribution, the reducer merges them*.

### Graph

```python
builder = StateGraph(State)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()
```

```
            ┌── search_web ──────┐
   START ──┤                     ├── generate_answer ── END
            └── search_wikipedia ─┘
```

`generate_answer` runs only after **both** retrievals finish (super-step barrier), and it sees `context` as a 2-element list — one entry per source.

### Run

```python
result = graph.invoke({"question": "How were Nvidia's Q2 2025 earnings"})
result['answer'].content
```

The model gets richer context than either source alone could provide, and the round-trip is roughly the time of the *slower* lookup, not their sum.

---

## When You Need a Reducer — Quick Decision Table

| Scenario | Reducer? | Why |
|----------|----------|-----|
| Single writer per step, value is fine to replace | No | Default overwrite is what you want |
| Single writer per step, value should accumulate | **Yes** | Need to merge with existing |
| Multiple writers in same step (fan-out) | **Yes (required)** | Otherwise `INVALID_CONCURRENT_GRAPH_UPDATE` |
| `Send()` (map-reduce) writing back to one key | **Yes (required)** | Same as fan-out — many writes per step |

---

## Key Takeaways

1. **Fan-out** = one node has multiple downstream edges. They run **concurrently in the same super-step**.
2. **Fan-in** = `add_edge([n1, n2, ...], "next")` — `next` waits for *all* of them to finish.
3. Without a reducer, two writes to the same key in one step **error out** with `INVALID_CONCURRENT_GRAPH_UPDATE`. This is a feature — silent drops are worse than loud errors.
4. **`operator.add` on a list** is the simplest reducer: it concatenates. Each parallel writer returns `[its_value]`, the reducer merges them into a list of contributions.
5. **Order within a super-step is deterministic but not user-controlled.** If order matters, use a custom reducer (e.g., the `sorting_reducer`) or a dedicated sink node.
6. The reducer applies to **every** write to that key — sequential writes also start accumulating, not just concurrent ones.
7. Real use case: gather context from **multiple sources in parallel** and merge into a single `context` list before the LLM sees it. Latency = slowest source, not sum.
