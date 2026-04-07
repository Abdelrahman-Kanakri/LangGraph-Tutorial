# Walkthrough: State Schema

> **Notebook:** `state-schema.ipynb`
> **Goal:** Compare three ways to define graph state — **TypedDict**, **Dataclass**, and **Pydantic** — and understand when to use each.

---

## The Big Picture

In Module 1, we always used `TypedDict` or `MessagesState` for state. But LangGraph supports multiple schema approaches, each with different trade-offs around validation, syntax, and safety.

---

## 1. TypedDict

```python
from typing_extensions import TypedDict
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]
```

**Line-by-line:**
- `TypedDict` — a Python standard library class that defines a typed dictionary
- `name: str` — a key called `name` that should hold a string
- `mood: Literal["happy", "sad"]` — a key that should only be `"happy"` or `"sad"`

**Important: These are just hints!** Python does NOT enforce them at runtime:

```python
# This works even though "mad" is not in Literal["happy", "sad"]!
state = TypedDictState(name="Alice", mood="mad")  # No error
```

Static type checkers (like mypy or your IDE) will warn you, but the code runs fine.

### Using TypedDict in a Graph

```python
def node_1(state):
    print("---Node 1---")
    return {"name": state['name'] + " is ... "}

def node_2(state):
    print("---Node 2---")
    return {"mood": "happy"}

def node_3(state):
    print("---Node 3---")
    return {"mood": "sad"}
```

- Access with **dict subscript**: `state['name']`
- Return a **dict** with the keys to update
- Each key is a "channel" in the graph — nodes update individual channels

```python
builder = StateGraph(TypedDictState)
# ... add nodes and edges ...
graph = builder.compile()

graph.invoke({"name": "Alice"})
# {'name': 'Alice is ... ', 'mood': 'happy'}
```

- Invoke with a **plain dict**
- Only provide the keys you want to set initially (the rest get populated by nodes)

---

## 2. Dataclass

```python
from dataclasses import dataclass

@dataclass
class DataClassState:
    name: str
    mood: Literal["Happy", "Sad"]
```

**Line-by-line:**
- `@dataclass` — a decorator that auto-generates `__init__`, `__repr__`, etc.
- Fields are defined the same way, but you get a **class instance** instead of a dict

### Key Difference: Attribute Access

```python
def node_1(state):
    print("---Node 1---")
    return {"name": state.name + " is ... "}  # state.name NOT state["name"]
```

- Access with **dot notation**: `state.name` instead of `state["name"]`
- But nodes still **return a dict** for updates — LangGraph stores each key separately

### Invoking with a Dataclass

```python
graph.invoke(DataClassState(name="Abood", mood="sad"))
# {'name': 'Abood is ... ', 'mood': 'sad'}
```

- You pass an **instance** of the dataclass, not a plain dict
- The output is still a dict (LangGraph's internal representation)

### No Runtime Validation (Same as TypedDict)

```python
dataclass_instance = DataClassState(name="Lance", mood="mad")
# No error! "mad" is accepted even though the hint says "Happy" or "Sad"
```

---

## 3. Pydantic

```python
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str  # "happy" or "sad"

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value):
        if value not in ["happy", "sad"]:
            raise ValueError("Mood must be either 'happy' or 'sad'")
        return value
```

**Line-by-line:**
- `BaseModel` — Pydantic's base class. Unlike `TypedDict`/`dataclass`, it **validates at runtime**.
- `@field_validator("mood")` — a decorator that runs custom validation when `mood` is set.
- `@classmethod` — required by Pydantic v2 for validators.
- `raise ValueError(...)` — this actually **stops execution** with a clear error message.

### Runtime Validation in Action

```python
# Valid input — works fine
state = PydanticState(name="Mustafa", mood="happy")

# Invalid input — raises an error!
try:
    state = PydanticState(name="Lance", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)
# Output: Mood must be either 'happy' or 'sad'
```

### Using Pydantic in a Graph

```python
builder = StateGraph(PydanticState)
# ... same nodes and edges ...
graph = builder.compile()

# Valid — works
graph.invoke(PydanticState(name="Mustafa", mood="happy"))
# {'name': 'Mustafa is ... ', 'mood': 'happy'}

# Invalid — crashes immediately
graph.invoke(PydanticState(name="Mustafa", mood="happsy"))
# ValidationError: Mood must be either 'happy' or 'sad'
```

The validation happens **before** the graph even starts running. Bad data is caught at the door.

---

## Comparison Table

| Feature | TypedDict | Dataclass | Pydantic |
|---------|-----------|-----------|----------|
| **Runtime validation** | No | No | **Yes** |
| **Type hints** | Yes | Yes | Yes |
| **State access syntax** | `state["key"]` | `state.key` | `state.key` |
| **Invoke with** | `dict` | Class instance | Class instance |
| **Node return type** | `dict` | `dict` | `dict` |
| **Invalid data** | Silently accepted | Silently accepted | **Raises `ValidationError`** |
| **Best for** | Quick prototyping | Cleaner syntax | Production apps |
| **Boilerplate** | Minimal | Minimal | More (validators) |

---

## When to Use What

| Scenario | Recommendation |
|----------|---------------|
| Learning / tutorials / prototyping | **TypedDict** — simplest, least ceremony |
| You prefer dot-notation access | **Dataclass** — same as TypedDict but `state.name` |
| Production app, user-facing inputs | **Pydantic** — catches invalid data before it causes problems |
| Working with messages | **`MessagesState`** — pre-built, includes `add_messages` reducer |

---

## Key Takeaways

1. **TypedDict** = type hints only, dict access (`state["key"]`), no runtime checks
2. **Dataclass** = type hints only, attribute access (`state.key`), no runtime checks
3. **Pydantic** = **runtime validation**, attribute access, raises errors on invalid data
4. All three return **dicts** from nodes — LangGraph stores state keys independently
5. The graph structure (nodes, edges) is **identical** regardless of which schema you use
6. Choose based on your validation needs: prototyping → `TypedDict`, production → `Pydantic`
