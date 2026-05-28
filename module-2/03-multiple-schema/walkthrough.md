# Walkthrough: Multiple Schemas

> **Notebook:** `multiple-schema.ipynb`
> **Goal:** Learn how to use **private state** between nodes and separate **input/output schemas** to control what goes in and out of your graph.

---

## The Big Picture

So far, every node read and wrote to the same state schema. But sometimes you need:
- **Private state** — intermediate data that nodes pass to each other but should NOT appear in the graph's final output
- **Input/Output schemas** — restrict what the user can send in and what they get back

---

## 1. Private State

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int
```

**Two schemas:**
- `OverallState` — the graph's "public" state. This is what the user sends and receives.
- `PrivateState` — internal-only. Contains `baz`, which is never exposed to the user.

### The Nodes

```python
def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}
```

**Line-by-line:**
- `node_1` takes `OverallState` (reads `foo`), returns `PrivateState` (writes `baz`)
- `node_2` takes `PrivateState` (reads `baz`), returns `OverallState` (writes `foo`)
- The type hints on the function signatures tell LangGraph which schema each node uses

### Build and Run

```python
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

graph.invoke({"foo": 1})
# {'foo': 3}
```

**What happened:**
1. Input: `foo = 1`
2. `node_1`: reads `foo` (1), returns `baz = 2`
3. `node_2`: reads `baz` (2), returns `foo = 3`
4. Output: `{'foo': 3}` — notice `baz` is **not** in the output!

`baz` was used internally between nodes but excluded from the final output because it's not in `OverallState`.

**Use case:** Private state is useful for intermediate processing — for example, a node that cleans/transforms data before passing it to the next node, without exposing those internals to the user.

---

## 2. Input / Output Schema

### The Problem: Too Much in the Output

By default, the graph returns **all keys** from the state:

```python
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Abood"}

def answer_node(state: OverallState):
    return {"answer": "bye Abood"}

graph.invoke({"question": "What is my name?"})
# {'question': 'What is my name?', 'answer': 'bye Abood', 'notes': '... his name is Abood'}
```

The output includes `question`, `answer`, AND `notes`. But the user probably only needs `answer`.

### The Solution: Separate Input and Output Schemas

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str
```

**Three schemas:**
- `InputState` — what the user is allowed to send (only `question`)
- `OutputState` — what the user receives back (only `answer`)
- `OverallState` — the full internal state with all keys

### The Nodes with Type Hints

```python
def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}
```

- `thinking_node(state: InputState)` — this node only sees `question` (not `answer` or `notes`)
- `answer_node(state: OverallState) -> OutputState` — reads the full state, but the `-> OutputState` hint tells LangGraph to filter the output

### Build with Input/Output Schemas

```python
graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()
graph.invoke({"question": "What is my name?"})
# {'answer': 'bye Lance'}
```

**Key line:** `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)`

Now the output is **filtered** — only `answer` is returned. `question` and `notes` are excluded.

---

## Private State vs. Input/Output Schemas

| Feature | Private State | Input/Output Schemas |
|---------|--------------|---------------------|
| **Purpose** | Hide intermediate data between nodes | Control what users send/receive |
| **Mechanism** | Separate `TypedDict` for internal nodes | `input_schema` / `output_schema` params |
| **Who sees it** | Nodes (via type hints) | User (via graph input/output) |
| **Use case** | Data cleaning, intermediate transforms | API design, clean interfaces |

---

## Key Takeaways

1. **Private state** lets nodes pass data internally without exposing it in the graph output
2. **Input/Output schemas** act as filters on what goes in and out of the graph
3. **Type hints** on node functions tell LangGraph which schema each node uses
4. `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)` — the three-schema pattern
5. The internal state can have many keys; the user only sees what you expose
