# Walkthrough: Simple Graph

> **Notebook:** `simple_graph.ipynb`
> **Goal:** Build your first LangGraph graph with 3 nodes, normal edges, and a conditional edge.

---

## The Big Picture

This is the simplest possible LangGraph graph. It teaches three core concepts:
1. **State** — the data that flows through the graph
2. **Nodes** — Python functions that read/write state
3. **Edges** — connections between nodes (fixed or conditional)

---

## 1. Define the State

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

**Line-by-line:**
- `TypedDict` — a Python class that defines a dictionary with typed keys. Think of it as a blueprint for the data your graph passes around.
- `graph_state: str` — our state has a single key called `graph_state` that holds a string.

**Why `TypedDict`?** LangGraph needs to know the shape of your state. `TypedDict` is the simplest way to define it. We'll see `Pydantic` and `dataclass` alternatives in Module 2.

> Every node in the graph receives this state as input and returns updates to it.

---

## 2. Define the Nodes

```python
def node1(state: State) -> State:
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}

def node2(state: State) -> State:
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " Happy!"}

def node3(state: State) -> State:
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " Sad!"}
```

**Line-by-line:**
- Each node is a **plain Python function** — no special class or decorator needed.
- The function signature `(state: State) -> State` means: receives the current state, returns an update.
- `state["graph_state"]` — reads the current value from state (dict-style access because we used `TypedDict`).
- `return {"graph_state": ...}` — returns a dict with the key to update.

**Critical concept: Overwrite by default.** The returned dict **replaces** the value of `graph_state`. It does NOT append. So after `node1`, the state becomes `"Hi There, I'm Abood I am"` — the original input plus `" I am"`.

---

## 3. Define a Conditional Edge

```python
import random
from typing import Literal

def decide_node(state: State) -> Literal["node2", "node3"]:
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "node2"
    else:
        return "node3"
```

**Line-by-line:**
- `Literal["node2", "node3"]` — type hint that says this function can only return one of these two strings. These must match the node names registered in the graph.
- `state["graph_state"]` — we *could* use the state to make a decision (e.g., check if the user said something specific). Here we just do a random 50/50 split.
- The returned string tells LangGraph **which node to run next**.

**Two types of edges in LangGraph:**

| Type | When to use | Example |
|------|-------------|---------|
| **Normal edge** | Always go to the same next node | `node1 → node2` |
| **Conditional edge** | Choose the next node based on logic | `node1 → node2 OR node3` |

---

## 4. Build the Graph

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
```

- `StateGraph(State)` — creates a new graph builder and tells it to use our `State` schema.

### Add Nodes

```python
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)
```

- `add_node("node1", node1)` — registers the function `node1` under the name `"node1"`. The name is what edges reference.

### Add Edges

```python
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_node)
builder.add_edge("node2", END)
builder.add_edge("node3", END)
```

- `START` — a special built-in node representing the entry point. User input goes here first.
- `add_edge(START, "node1")` — always go from START to node1.
- `add_conditional_edges("node1", decide_node)` — after node1, call `decide_node()` to pick the next node.
- `add_edge("node2", END)` / `add_edge("node3", END)` — both node2 and node3 terminate the graph.
- `END` — a special built-in node representing the exit point.

### Compile

```python
graph = builder.compile()
```

- `.compile()` validates the graph structure (checks for disconnected nodes, missing edges, etc.) and returns a runnable graph.

### Visualize

```python
display(Image(graph.get_graph().draw_mermaid_png()))
```

- Renders the graph as a visual diagram using Mermaid. Useful for debugging and understanding the flow.

---

## 5. Run the Graph

```python
graph.invoke({"graph_state": "Hi There, I'm Abood"})
```

**What happens step by step:**

1. `invoke()` receives the input dict → sets initial state: `{"graph_state": "Hi There, I'm Abood"}`
2. `START` → routes to `node1`
3. `node1` reads state, appends `" I am"` → state: `"Hi There, I'm Abood I am"`
4. Conditional edge calls `decide_node()` → randomly picks `"node2"` or `"node3"`
5. If `node2`: appends `" Happy!"` → final state: `"Hi There, I'm Abood I am Happy!"`
6. If `node3`: appends `" Sad!"` → final state: `"Hi There, I'm Abood I am Sad!"`
7. `END` → graph returns the final state dict

**Output:**
```python
{'graph_state': "Hi There, I'm Abood I am Sad!"}
```

---

## Key Takeaways

1. **State is a dict** — defined with `TypedDict`, shared across all nodes
2. **Nodes are functions** — they receive state, return updates (dict)
3. **Default behavior is overwrite** — returning `{"graph_state": "new"}` replaces the old value
4. **Conditional edges** — a function that returns the name of the next node
5. **`START` and `END`** — special nodes for entry and exit
6. **`.compile()`** — validates and makes the graph runnable
7. **`.invoke()`** — runs the graph synchronously from START to END
