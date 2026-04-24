# Walkthrough: Dynamic Breakpoints

> **Notebook:** `dynamic-breakpoints.ipynb`
> **Goal:** Learn how a graph can **interrupt itself** from inside a node — based on runtime conditions — using `NodeInterrupt`.

---

## The Big Picture

The breakpoints from the previous notebook are **static** — you declare them at compile time with `interrupt_before=[...]`. They always fire. Always. Regardless of state.

But often you want to pause only **when something is wrong**:

- Input is too long
- A tool returned a suspicious value
- The agent is about to spend money above a threshold
- A field failed validation

For these, LangGraph provides **`NodeInterrupt`** — an exception you `raise` from inside a node. The graph catches it, pauses execution, and surfaces the interrupt message to the caller.

| Static breakpoints | Dynamic breakpoints (`NodeInterrupt`) |
|---|---|
| Declared at `compile()` time | Raised at runtime, from inside a node |
| Always fire at that node | Only fire when your condition is true |
| No context passed to user | You can attach a message explaining why |

---

## 1. Setup: A Simple 3-Step Graph

No LLM this time — just three passthrough steps, with a conditional interrupt in the middle:

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    # Dynamically interrupt if input is too long
    if len(state["input"]) > 5:
        raise NodeInterrupt(
            f"Received input that is longer than 5 characters: {state['input']}"
        )
    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---step 3---")
    return state
```

**The key line:**

```python
raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
```

- `NodeInterrupt` is imported from `langgraph.errors`
- The string you pass is attached to the interrupt — the caller can read it to know **why** execution paused
- Raising it halts the node and saves the current checkpoint

Compile with a checkpointer (required — same as static breakpoints):

```python
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

graph = builder.compile(checkpointer=MemorySaver())
```

> **Best practice:** use `NodeInterrupt` for conditions the **developer** can't know at compile time. For unconditional pauses (e.g., always approve before tool execution), use `interrupt_before=[...]` — it's simpler.

---

## 2. Triggering the Interrupt

Run with an input longer than 5 characters:

```python
initial_input = {"input": "hello world"}
thread_config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)
```

**Output:**
```
{'input': 'hello world'}
---Step 1---
{'input': 'hello world'}
```

`step_1` ran. Control moved to `step_2`, which raised `NodeInterrupt` — so the stream ended. `step_3` never ran.

---

## 3. Inspecting the Interrupt

Two things to check:

**a. Which node is paused:**

```python
state = graph.get_state(thread_config)
state.next
# ('step_2',)
```

The graph is paused *at* `step_2` — that's the one that raised.

**b. The interrupt details:**

```python
print(state.tasks)
```

**Output:**
```python
(PregelTask(
    id='...',
    name='step_2',
    interrupts=(Interrupt(value='Received input that is longer than 5 characters: hello world', ...),),
    ...
 ),)
```

`state.tasks` contains the pending tasks, and each has an `interrupts` tuple listing everything raised. This is how a UI reads the interrupt message and shows it to the user.

---

## 4. The Gotcha: Resuming Doesn't Work Yet

Try to resume as you would with a static breakpoint:

```python
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

**Output:**
```
{'input': 'hello world'}
```

Nothing happened. Why?

```python
state = graph.get_state(thread_config)
print(state.next)
# ('step_2',)
```

Still paused at `step_2`. **The condition that caused the interrupt is still true** — `state["input"]` is still `"hello world"`. LangGraph re-runs the node, the same `if len(...) > 5` triggers, `NodeInterrupt` raises again.

**This is the critical property of dynamic breakpoints:** they re-fire until the underlying state changes.

```
static interrupt_before     → resume with None → runs the node
dynamic NodeInterrupt       → resume with None → runs the node → re-raises → stuck
                              fix: update state → resume with None → now it passes
```

---

## 5. Fixing the State and Resuming

Change the input so the condition no longer triggers:

```python
graph.update_state(thread_config, {"input": "hi"})
```

Then resume:

```python
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

**Output:**
```
{'input': 'hi'}
---Step 2---
{'input': 'hi'}
---step 3---
{'input': 'hi'}
```

`step_2` now passes its check, prints, and moves on. `step_3` runs. The graph reaches `END`.

**The full recovery pattern:**

```
stream(input) → NodeInterrupt fires → inspect state.tasks[0].interrupts
                                   ↓
                          update_state(thread, fixed_patch)
                                   ↓
                          stream(None) → node re-runs → passes → continues
```

---

## When to Use `NodeInterrupt`

| Scenario | Use |
|----------|-----|
| Input validation failed inside a node | `NodeInterrupt("input missing required field X")` |
| Budget / cost threshold exceeded | `NodeInterrupt("run cost exceeded $5, approve?")` |
| Tool returned a suspicious result | `NodeInterrupt("tool output looks malformed: ...")` |
| Always pause before tool execution | `interrupt_before=["tools"]` — static is simpler here |
| Pause every node for debugging | `debug=True` on compile, not `NodeInterrupt` |

---

## Static vs Dynamic Breakpoints — Side by Side

```
STATIC                                  DYNAMIC
────────                                ──────────
compile(interrupt_before=["tools"])     raise NodeInterrupt("reason")
│                                       │
├─ fires: every time                    ├─ fires: only when you raise
├─ resume with None: works directly     ├─ resume with None: re-triggers
├─ message to user: none                ├─ message to user: the NodeInterrupt arg
└─ use: unconditional gates             └─ use: conditional guards
```

---

## Key Takeaways

1. **`NodeInterrupt`** (from `langgraph.errors`) is an exception you `raise` inside a node to pause the graph dynamically
2. The string passed to `NodeInterrupt(...)` is surfaced via `state.tasks[*].interrupts` — use it to tell the user *why* you paused
3. **A checkpointer is still required** — dynamic breakpoints save state the same way as static ones
4. **Resuming with `None` is not enough by itself** — the node re-runs, the condition re-triggers, and the interrupt fires again
5. You must **change the state** (via `graph.update_state`) to clear the condition, then resume
6. Use `NodeInterrupt` for runtime validation / guards; use `interrupt_before=[...]` for unconditional pauses
7. This is the mechanism that makes **self-validating agents** possible — the agent stops itself when it detects a problem
