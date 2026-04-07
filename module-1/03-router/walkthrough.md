# Walkthrough: Router

> **Notebook:** `router.ipynb`
> **Goal:** Extend the chain so the LLM can **route** between executing a tool and responding directly.

---

## The Big Picture

In the Chain notebook, the LLM could **decide** to call a tool — but the graph ended immediately. The tool was never executed. Now we fix that by adding:

1. **`ToolNode`** — a pre-built node that executes tool calls
2. **`tools_condition`** — a pre-built conditional edge that routes based on whether the LLM called a tool

This creates a **router**: the LLM decides to either respond directly (→ END) or call a tool (→ tools node).

---

## 1. Set Up the Tool and LLM

```python
from langchain_mistralai import ChatMistralAI

def multiply(a: int, b: int) -> int:
    """
    Args:
        a (int): first number
        b (int): second number

    Returns:
        int: the product of a and b
    """
    return a * b

llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0)
llm_with_tools = llm.bind_tools([multiply])
```

Same pattern as the Chain notebook:
- Define a function with **type hints and a docstring** (the LLM reads these to understand the tool)
- `bind_tools([multiply])` makes the LLM aware of the tool

---

## 2. Pre-built Components: `ToolNode` and `tools_condition`

```python
from langgraph.prebuilt import ToolNode, tools_condition
```

These are LangGraph's built-in helpers:

| Component | What it does |
|-----------|-------------|
| `ToolNode([multiply])` | A node that reads the `tool_calls` from the latest `AIMessage` and executes the matching function. Returns a `ToolMessage` with the result. |
| `tools_condition` | A conditional edge function that checks the latest `AIMessage`. If it contains `tool_calls` → route to `"tools"`. If not → route to `END`. |

**Why pre-built?** This pattern (check for tool calls → execute them) is so common that LangGraph ships it out of the box. You *could* write these yourself, but there's no need.

---

## 3. The Node Function

```python
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

- Reads all messages from state
- Sends them to the LLM (which may or may not return a tool call)
- Wraps the `AIMessage` response in a list and returns it
- The `add_messages` reducer (built into `MessagesState`) appends it to the conversation

---

## 4. Build the Router Graph

```python
from langgraph.graph import StateGraph, MessagesState, START, END

builder = StateGraph(MessagesState)
```

### Add Nodes

```python
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
```

- `"tool_calling_llm"` — our custom node that calls the LLM
- `"tools"` — the pre-built `ToolNode` that executes tool calls

### Add Edges

```python
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)
```

**The flow:**

```
START → tool_calling_llm → [tools_condition] → tools → END
                                ↓
                               END (if no tool call)
```

- `START → tool_calling_llm` — always starts with the LLM
- `tools_condition` checks the LLM's output:
  - **Has `tool_calls`?** → route to `"tools"` node
  - **No `tool_calls`?** → route to `END`
- `tools → END` — after executing the tool, the graph ends

### Compile

```python
graph = builder.compile()
```

---

## 5. Run It

**With a tool-triggering question:**

```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="What is the product of 5 and 10?")]
messages = graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()
```

**Output:**
```
Human Message: What is the product of 5 and 10?
Ai Message:
  Tool Calls: multiply(a=5, b=10) [Call ID: l29w6bHhQ]
Tool Message (name=multiply): 50
```

**What happened step by step:**
1. `START` → `tool_calling_llm`: LLM sees the question, decides to call `multiply(5, 10)`
2. `tools_condition`: detects `tool_calls` in the `AIMessage` → routes to `"tools"`
3. `tools` (`ToolNode`): executes `multiply(5, 10)` → returns `ToolMessage(content="50")`
4. `tools → END`: graph terminates

**With a regular greeting (no tool needed):**

```python
messages = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
# Human: Hello!
# AI: Hello! How can I assist you today?
```

Here, the LLM responds directly. `tools_condition` sees no `tool_calls` → routes to `END`.

---

## Router vs. Chain — What Changed?

| Feature | Chain (previous) | Router (this) |
|---------|-----------------|---------------|
| Nodes | 1 (`tool_calling_llm`) | 2 (`tool_calling_llm` + `tools`) |
| Tool execution | No (just returns the call) | Yes (`ToolNode` runs it) |
| Routing | None (always → END) | Conditional (`tools_condition`) |
| Output | Raw `AIMessage` with tool call | Actual `ToolMessage` with result |

---

## Limitation

The tool result (`ToolMessage`) goes to `END` — it's returned to the user as-is. The LLM **never sees** the tool result, so it can't give a natural language answer like "The product of 5 and 10 is 50."

The **Agent** (next notebook) fixes this by looping the tool result back to the LLM.

---

## Key Takeaways

1. **`ToolNode`** executes tool calls from the LLM's `AIMessage` — no custom code needed
2. **`tools_condition`** routes between tool execution and direct response — also built-in
3. The router pattern: LLM decides → conditional edge checks → tool runs or graph ends
4. **Limitation**: the tool result doesn't go back to the LLM — the Agent pattern solves this
