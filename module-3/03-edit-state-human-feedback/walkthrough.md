# Walkthrough: Editing State & Human Feedback

> **Notebook:** `edit-state-human-feedback.ipynb`
> **Goal:** Learn how to **modify graph state at a breakpoint** — either directly via `graph.update_state()`, or by collecting **human feedback** through a dedicated placeholder node.

---

## The Big Picture

Breakpoints (previous notebook) let us **pause** the graph. But pausing is only half the story — once paused, we often want to **change what happens next**. Three common scenarios:

1. **Approval** — accept or reject an action (covered in breakpoints)
2. **Debugging** — inspect state and resume
3. **Editing** — *modify* the state, then resume

This notebook covers editing. Two patterns:

| Pattern | When to use |
|---------|-------------|
| Direct `update_state()` | Developer tooling, tests, debugging |
| `human_feedback` placeholder node + `as_node=` | Production human-in-the-loop UIs |

---

## 1. Setup: The Arithmetic Agent (Again)

Same ReAct agent as the breakpoints notebook — `multiply`, `add`, `divide` tools bound to a Mistral model:

```python
from langchain_mistralai import ChatMistralAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

tools = [add, multiply, divide]
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

**The key difference:** this time we interrupt **before `assistant`**, not before `tools`:

```python
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)
```

Why? We want to **edit the input** to the LLM *before* it sees it.

---

## 2. Running Until the Breakpoint

```python
initial_input = {"messages": "Multiply 2 and 3"}
thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

**Output:**
```
Human Message: Multiply 2 and 3
```

The graph enters, saves the HumanMessage to state, then **stops before `assistant`** runs. The LLM has not been called yet.

Inspect the paused state:

```python
state = graph.get_state(thread)
state.next  # ('assistant',)
```

---

## 3. Direct State Edit with `update_state()`

The core API:

```python
graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually is 3 and 3!")]},
)
```

**What this does:**
- Loads the current checkpoint for `thread`
- Applies the update through the state reducers (so `messages` goes through `add_messages`)
- Saves a new checkpoint

**Behavior of `add_messages` — two modes:**

| Case | Behavior |
|------|----------|
| Message **without** `id` | **Appends** to the list |
| Message **with** existing `id` | **Overwrites** the message with that id |

Here we append — now state has both messages:

```python
new_state = graph.get_state(thread).values
for m in new_state["messages"]:
    m.pretty_print()
```

**Output:**
```
Human Message: Multiply 2 and 3
Human Message: No, actually is 3 and 3!
```

---

## 4. Resuming After the Edit

Pass `None` to resume from the checkpoint:

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

The graph runs `assistant` (which now sees the corrected request), gets a tool call, runs `tools`, then loops back to — **the `assistant` breakpoint again**. Pass `None` a second time to finish:

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

**Final answer:** 3 × 3 = 9.

> **Why two `None` calls?** Because `interrupt_before=["assistant"]` stops **every time** control returns to that node. The ReAct loop hits `assistant` twice (once for the tool call, once for the final answer), so you interrupt twice.

---

## 5. Pattern 2: A `human_feedback` Placeholder Node

For real UIs, baking human input into the graph topology is cleaner. Add a **no-op node** whose only purpose is to be interrupted on:

```python
# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    pass

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "human_feedback")   # tool results re-enter via feedback

graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
```

**Graph topology:**

```
START → human_feedback 🛑 → assistant → [tools_condition]
                ↑                ├─→ tools ──┐
                └────────────────┼───────────┘
                                 └─→ END
```

The `human_feedback` node is just `pass` — it does nothing. Its purpose is to be a **named anchor** for interrupting and for applying state updates.

---

## 6. `as_node` — Applying Updates as if from a Specific Node

The magic parameter:

```python
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")
```

**What `as_node="human_feedback"` does:**
- Tells the checkpointer "pretend this update came from the `human_feedback` node"
- The graph then proceeds along whatever edge leaves `human_feedback` (here: `→ assistant`)

Without `as_node`, the update is applied but the graph doesn't know it has "executed" that node, so `.next` still points to `human_feedback` and you'd get stuck in a loop.

**Full pattern:**

```python
initial_input = {"messages": "Multiply 2 and 3"}
thread = {"configurable": {"thread_id": "5"}}

# 1. Run until the breakpoint at human_feedback
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 2. Collect input from the human
user_input = input("Tell me how you want to update the state: ")

# 3. Apply it as if the human_feedback node produced it
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

# 4. Continue — now the graph proceeds from human_feedback → assistant
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

---

## `update_state()` vs `as_node`: When to Use Which

```
plain update_state(thread, patch)
    → patch applied, .next unchanged
    → use for: edit-then-resume at same breakpoint

update_state(thread, patch, as_node="X")
    → patch applied AND graph advances from node X
    → use for: human_feedback-style placeholder nodes
```

---

## Direct Edit vs Placeholder Node — Comparison

| | Direct `update_state()` | `human_feedback` node |
|---|---|---|
| Code change | None — just call the API | Add a placeholder node + edge |
| Where feedback lives | Outside the graph | As part of graph topology |
| Visible in graph diagram | No | Yes — the node shows up |
| Best for | Ad-hoc debugging, tests | Production UIs, LangGraph Studio |
| Needs `as_node=` | No | Yes |

---

## Key Takeaways

1. **`graph.update_state(thread, patch)`** modifies state at a checkpoint — applied through the normal reducers
2. **`add_messages` reducer:** no `id` → appends; matching `id` → **overwrites**
3. **Passing `None` to `.stream()`** resumes from the last checkpoint — works after `update_state` too
4. **A no-op `human_feedback` node** is a clean way to bake human-in-the-loop into the graph topology
5. **`as_node="human_feedback"`** tells the graph the update *came from* that node, so execution advances along its outgoing edges
6. Without `as_node`, an update does not advance `.next` — the graph would pause at the same breakpoint again
7. Combine with a UI (Studio, a web app) to let end-users correct, approve, or augment the agent's state mid-run
