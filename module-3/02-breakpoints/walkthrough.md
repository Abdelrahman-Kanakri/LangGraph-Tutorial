# Walkthrough: Breakpoints

> **Notebook:** `breakpoints.ipynb`
> **Goal:** Learn how to **interrupt** a running graph at specific nodes to enable **human-in-the-loop** workflows — approval, debugging, and editing.

---

## The Big Picture

So far, graphs run from START to END without stopping. But many real-world scenarios require human involvement:

1. **Approval** — Pause before the agent calls a tool; let the user approve
2. **Debugging** — Pause, inspect state, rewind to reproduce an issue
3. **Editing** — Pause, modify the state, then continue

LangGraph solves this with **breakpoints** — `interrupt_before` and `interrupt_after` parameters on `.compile()` that stop execution at chosen nodes.

---

## 1. Setup: The Arithmetic Agent

Same ReAct agent from Module 1, with multiple tools:

```python
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [multiply, add, divide]
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

> **Best practice:** note the list form — `[llm.invoke(...)]`. Always wrap messages in a list when returning to state. See [Module 1 / Chain walkthrough](../../module-1/02-chain/walkthrough.md#7-the-chain-graph) for full reasoning.

---

## 2. The Key Line: `interrupt_before`

```python
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
```

**The magic:**
- `interrupt_before=["tools"]` — the graph will **pause** just before the `tools` node executes
- `checkpointer=memory` — **required** for breakpoints to work (state needs to be saved before pausing)

**Why `interrupt_before=["tools"]`?** Tool execution is the place where agents do things in the real world (send emails, write to DBs, make API calls). Pausing here lets a human approve.

You can also use `interrupt_after=[...]` to pause AFTER a node runs.

---

## 3. Running Until the Breakpoint

```python
initial_input = {"messages": [HumanMessage(content="What is 5 * 3?")]}
thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

**Output:**
```
Human Message: What is 5 * 3?
Ai Message:
  Tool Calls: multiply(a=5, b=3) [Call ID: 9ZbbzAStu]
```

The graph ran through `assistant` (which returned a tool call), then **stopped** before executing the `tools` node. No tool was actually called yet — the AI message is just the request.

---

## 4. Inspecting the Paused State

```python
state = graph.get_state(thread)
state.next
# ('tools',)
```

**Line-by-line:**
- `graph.get_state(thread)` — retrieves the current checkpoint for this thread
- `state.next` — tells you which node(s) would run next if we resumed

This is invaluable for debugging: you can inspect the full state (messages, tool calls, etc.) before the agent takes action.

---

## 5. Resuming Execution

**The trick:** passing `None` as input resumes from the last checkpoint.

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

**Output:**
```
Ai Message:
  Tool Calls: multiply(a=5, b=3)         ← re-emitted current state
Tool Message (multiply): 15              ← tool actually ran!
Ai Message: 5 * 3 equals **15**.         ← final response
```

**What happens:**
1. LangGraph loads the checkpoint
2. Re-emits the current state (the AI tool call)
3. Proceeds from the paused point: runs `tools` → returns to `assistant` → final answer

---

## 6. Human Approval Pattern

Combining breakpoints with user input creates an approval workflow:

```python
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}
thread = {"configurable": {"thread_id": "2"}}

# Step 1: Run until the breakpoint
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# Step 2: Inspect what the agent wants to do
lst_message = graph.get_state(thread).values["messages"][-1]
tool_name = lst_message.tool_calls[0]["name"]

# Step 3: Ask the user
user_approval = input(f"Do you want to call the {tool_name} tool? (yes/no): ")

# Step 4: Continue or abort
if user_approval.lower() == "yes":
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
else:
    print("Operation cancelled by user")
```

**The pattern:**

```
stream(input) → breakpoint → inspect state → ask user → stream(None) to continue
                                                        OR abort
```

---

## How Breakpoints Work Under the Hood

```
Normal graph:   START → assistant → [tools_condition] → tools → assistant → END

With breakpoint: START → assistant → [tools_condition] → 🛑 PAUSE (checkpoint saved)
                                                           ↓
                                                    [user decides]
                                                           ↓
                                    stream(None) → tools → assistant → END
                                       OR abort (don't call again)
```

The checkpointer is critical — without it, there's no saved state to resume from.

---

## Breakpoint Use Cases

| Use case | `interrupt_before` | `interrupt_after` |
|----------|-------------------|-------------------|
| **Approval** (before risky action) | The action node (e.g., `tools`, `send_email`) | — |
| **Debugging** | Any node you want to inspect | Any node you want to verify output |
| **Editing** | Node whose input you want to modify | Node whose output you want to modify |
| **Audit logging** | — | Every node for a complete trace |

---

## Key Takeaways

1. **`interrupt_before=["node_name"]`** pauses the graph before that node executes
2. **`interrupt_after=["node_name"]`** pauses after a node executes
3. A **checkpointer is required** — breakpoints store state to resume from
4. **`graph.get_state(thread).next`** tells you which node would run next
5. **Passing `None` to `.stream()`** resumes from the last checkpoint
6. This is the foundation for **human-in-the-loop** patterns: approval, debugging, editing
7. Combined with `input()` or a UI, you get full control over agent execution
