# Walkthrough: Agent

> **Notebook:** `agent.ipynb`
> **Goal:** Build a full **ReAct agent** — an LLM that can call tools in a loop until it has the final answer.

---

## The Big Picture

The Router could execute a tool, but the result went straight to `END`. The LLM never saw it. Now we create a **loop**: tools → assistant → tools → assistant → ... until the LLM decides to respond directly.

This is the **ReAct** (Reason + Act) pattern:
1. **Act** — the model calls a tool
2. **Observe** — the tool result is passed back to the model
3. **Reason** — the model decides: call another tool, or respond to the user

---

## 1. Define Multiple Tools

```python
from langchain_mistralai import ChatMistralAI

def multiply(a: int, b: int) -> int:
    """
    args:
        a: int, b: int
    returns:
        int: the product of a and b
    """
    return a * b

def add(a: int, b: int) -> int:
    """
    args:
        a: int, b: int
    returns:
        int: the sum of a and b
    """
    return a + b

def divide(a: int, b: int) -> int:
    """
    args:
        a: int, b: int
    returns:
        int: the quotient of a and b
    """
    return a / b

tools = [multiply, add, divide]
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

**What's new:**
- **Three tools** instead of one — the LLM can now add, multiply, and divide
- All tools are passed together to `bind_tools(tools)` — the LLM sees all three and picks the right one per step

---

## 2. The Assistant Node with System Message

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState) -> MessagesState:
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}
```

**Line-by-line:**
- `SystemMessage` — instructions that guide the model's behavior. Here: "you do arithmetic."
- `[sys_msg] + state["messages"]` — **prepends** the system message to the conversation every time. 

**Critical detail: the system message is NOT stored in state.**

```python
# What happens each call:
llm_with_tools.invoke([sys_msg, human_msg1, ai_msg1, tool_msg1, ...])
#                       ↑ prepended fresh each time, never in state
```

Why? If you put `sys_msg` in state, the `add_messages` reducer would duplicate it on every call. By prepending it at invoke time, it's always there exactly once.

---

## 3. Build the Agent Graph — THE KEY CHANGE

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")  # <-- THE LOOP!

react_graph = builder.compile()
```

**The one line that changes everything:**

```python
builder.add_edge("tools", "assistant")
```

In the Router, this was `builder.add_edge("tools", END)`. Now the tool result goes **back to the assistant**, creating a loop.

**The flow:**

```
START → assistant → [tools_condition] → tools → assistant → [tools_condition] → ...
                         ↓                                        ↓
                        END (no tool call)                       END (no tool call)
```

The loop continues until the LLM responds **without** a tool call, at which point `tools_condition` routes to `END`.

---

## 4. Multi-Step Execution

```python
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()
```

**Output — step by step:**

```
Human Message: Add 3 and 4. Multiply the output by 2. Divide the output by 5

Ai Message:
  Tool Calls: add(a=3, b=4) [Call ID: NQDuGjhvj]

Tool Message (name=add): 7

Ai Message:
  Tool Calls: multiply(a=7, b=2) [Call ID: Mwb5YT5sA]

Tool Message (name=multiply): 14

Ai Message:
  Tool Calls: divide(a=14, b=5) [Call ID: qvWfFhsHL]

Tool Message (name=divide): 2.8

Ai Message: The final result is **2.8**.
```

**What happened:**

| Step | Node | Action | State after |
|------|------|--------|-------------|
| 1 | `assistant` | LLM calls `add(3, 4)` | `[human, ai(tool_call)]` |
| 2 | `tools` | Executes → returns `7` | `[..., tool(7)]` |
| 3 | `assistant` | LLM sees 7, calls `multiply(7, 2)` | `[..., ai(tool_call)]` |
| 4 | `tools` | Executes → returns `14` | `[..., tool(14)]` |
| 5 | `assistant` | LLM sees 14, calls `divide(14, 5)` | `[..., ai(tool_call)]` |
| 6 | `tools` | Executes → returns `2.8` | `[..., tool(2.8)]` |
| 7 | `assistant` | LLM sees 2.8, responds naturally | `[..., ai("2.8")]` |
| 8 | `tools_condition` | No tool call → END | Final state |

The agent went through the loop **3 times** (one for each operation), then exited.

---

## Router vs. Agent — What Changed?

| Feature | Router | Agent |
|---------|--------|-------|
| `tools` edge | `tools → END` | `tools → assistant` (loop!) |
| Tool result | Returned to user as-is | Fed back to LLM |
| Multi-step | No (one tool call max) | Yes (unlimited loop) |
| Final response | `ToolMessage` | Natural language from LLM |

The **only structural change** is one edge: `tools → assistant` instead of `tools → END`. That single change transforms a router into a full agent.

---

## Key Takeaways

1. **ReAct = Reason + Act** — the LLM reasons about results and decides next actions
2. **The loop** (`tools → assistant`) is what makes an agent an agent
3. **System messages** are prepended at invoke time, not stored in state
4. The agent can chain **multiple tool calls** across multiple loop iterations
5. The loop terminates when the LLM responds without a tool call
6. **Multiple tools** — the LLM picks the right one based on the task
