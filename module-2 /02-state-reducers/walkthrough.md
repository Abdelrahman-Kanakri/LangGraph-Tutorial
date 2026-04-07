# Walkthrough: State Reducers

> **Notebook:** `state-reducers.ipynb`
> **Goal:** Understand **how** state updates work — the mechanism behind overwriting, appending, custom logic, and the `add_messages` reducer.

---

## The Big Picture

By default, when a node returns a value for a state key, it **overwrites** the previous value. This works for simple cases but breaks when:
- Multiple nodes run **in parallel** and both update the same key
- You want to **accumulate** values (like a list of messages)

**Reducers** solve this. They define **how** updates are combined.

---

## 1. Default Behavior: Overwrite

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    foo: int

def node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

builder = StateGraph(State)
builder.add_node("node 1", node_1)
builder.add_edge(START, "node 1")
builder.add_edge("node 1", END)
graph = builder.compile()

graph.invoke({"foo": 1})
# {'foo': 2}
```

**What happened:**
- Input: `foo = 1`
- `node_1` returns `{"foo": 1 + 1}` = `{"foo": 2}`
- The old value (1) is **replaced** by 2
- This is the default: no reducer = overwrite

---

## 2. The Problem: Parallel Nodes

```python
class State(TypedDict):
    foo: int

def node_1(state):
    return {"foo": state['foo'] + 1}

def node_2(state):
    return {"foo": state['foo'] + 1}

def node_3(state):
    return {"foo": state['foo'] + 1}

# node_1 branches to BOTH node_2 AND node_3
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
```

```python
graph.invoke({"foo": 1})
# InvalidUpdateError: Can receive only one value per step.
# Use an Annotated key to handle multiple values.
```

**Why it fails:**
- `node_1` runs, then `node_2` and `node_3` run **in parallel** (same step)
- Both return `{"foo": 3}` (because both saw `foo = 2` from node_1)
- LangGraph gets TWO values for `foo` in the same step — which one should it keep?
- Without a reducer, this is **ambiguous**, so LangGraph raises an error

---

## 3. The Solution: `Annotated` + Reducer

```python
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]
```

**Line-by-line:**
- `Annotated[type, reducer]` — attaches a reducer function to a type
- `list[int]` — `foo` is now a **list** of integers (not a single int)
- `add` — Python's `operator.add`, which **concatenates** lists when applied to them

**How it works:**

```python
# operator.add on lists = concatenation
[1, 2] + [3] = [1, 2, 3]
```

### Single Node

```python
def node_1(state):
    return {"foo": [state['foo'][0] + 1]}

graph.invoke({"foo": [1]})
# {'foo': [1, 2]}
```

- Input: `[1]`
- `node_1` returns `[2]`
- Reducer: `[1] + [2]` = `[1, 2]` (appended, not overwritten!)

### Parallel Nodes (Now Works!)

```python
def node_1(state):
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    return {"foo": [state['foo'][-1] + 1]}

# node_1 → node_2 AND node_3 (parallel)
graph.invoke({"foo": [1]})
# {'foo': [1, 2, 3, 3]}
```

**Step by step:**
1. Input: `[1]`
2. `node_1` returns `[2]` → reducer: `[1] + [2]` = `[1, 2]`
3. `node_2` returns `[3]` AND `node_3` returns `[3]` (both see `[-1]` = 2)
4. Reducer: `[1, 2] + [3] + [3]` = `[1, 2, 3, 3]`

No more error! The reducer knows how to combine multiple updates.

---

## 4. The `None` Problem

```python
graph.invoke({"foo": None})
# TypeError: can only concatenate list (not "NoneType") to list
```

`operator.add` can't handle `None`. If the initial value might be `None`, you need a custom reducer.

---

## 5. Custom Reducers

```python
def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling None inputs."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class CustomReducerState(TypedDict):
    foo: Annotated[list, reduce_list]
```

**Line-by-line:**
- `reduce_list` is a function that takes **two arguments**: the current value (`left`) and the new value (`right`)
- It handles `None` by converting to empty list `[]`
- Then concatenates as usual

```python
# With the default reducer (operator.add):
graph.invoke({"foo": None})  # TypeError!

# With the custom reducer:
graph.invoke({"foo": None})  # {'foo': [2]} — works!
```

**The reducer function signature is always:** `(current_value, new_value) -> merged_value`

---

## 6. The `add_messages` Reducer

This is the most important reducer in LangGraph — purpose-built for message lists.

### Equivalent Definitions

```python
from typing import Annotated
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# Option A: Verbose (custom TypedDict)
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str

# Option B: Concise (extends MessagesState)
class ExtendedMessagesState(MessagesState):
    added_key_1: str
    added_key_2: str
```

Both are **identical** in behavior. `MessagesState` just saves you from typing the `Annotated` line.

### Superpower 1: Append

```python
from langchain_core.messages import AIMessage, HumanMessage

initial = [AIMessage(content="Hello!", name="Model"),
           HumanMessage(content="Looking for info on marine biology.", name="Abood")]

new = AIMessage(content="Sure! What specific info?", name="Model")

add_messages(initial, new)
# [AIMessage("Hello!"), HumanMessage("Looking for..."), AIMessage("Sure!")]
```

New messages are **appended** to the list.

### Superpower 2: Overwrite by ID

```python
initial = [AIMessage(content="Hello!", name="Model", id="1"),
           HumanMessage(content="I'm looking for info on marine biology.", name="Abood", id="2")]

# Same ID "2" — this REPLACES the original
new = HumanMessage(content="I'm looking for info on whales, specifically", name="Abood", id="2")

add_messages(initial, new)
# [AIMessage("Hello!"), HumanMessage("I'm looking for info on whales, specifically")]
```

If a new message has the **same `id`** as an existing one, it **replaces** it. Useful for editing/correcting messages.

### Superpower 3: Remove by ID

```python
from langchain_core.messages import RemoveMessage

messages = [AIMessage("Hi.", name="Bot", id="1"),
            HumanMessage("Hi.", name="Lance", id="2"),
            AIMessage("So you were researching ocean mammals?", name="Bot", id="3"),
            HumanMessage("Yes, what others should I learn about?", name="Lance", id="4")]

# Remove the first two messages
to_delete = [RemoveMessage(id=m.id) for m in messages[:-2]]
# [RemoveMessage(id="1"), RemoveMessage(id="2")]

add_messages(messages, to_delete)
# [AIMessage("So you were researching ocean mammals?"),
#  HumanMessage("Yes, what others should I learn about?")]
```

`RemoveMessage` is a special message type that **deletes** the message with the matching ID. This is useful for trimming conversation history to save tokens.

---

## Summary: Reducer Cheat Sheet

| Reducer | Type | Behavior |
|---------|------|----------|
| None (default) | Any | **Overwrite** — new value replaces old |
| `operator.add` | `list` | **Concatenate** — append new list to old |
| Custom function | Any | **Your logic** — full control over merge |
| `add_messages` | `list[AnyMessage]` | **Smart merge** — append, overwrite by ID, or remove by ID |

---

## Key Takeaways

1. **Default = overwrite** — no reducer means the new value replaces the old
2. **Parallel nodes need reducers** — otherwise LangGraph can't resolve conflicting updates
3. **`Annotated[type, reducer]`** attaches a reducer to a state key
4. **`operator.add`** works for simple list concatenation
5. **Custom reducers** handle edge cases like `None` inputs
6. **`add_messages`** is the star — it appends, overwrites by ID, and removes by ID
7. **`MessagesState`** = `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]` pre-built
