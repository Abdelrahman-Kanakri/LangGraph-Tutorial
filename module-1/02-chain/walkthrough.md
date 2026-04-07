# Walkthrough: Chain

> **Notebook:** `chain.ipynb`
> **Goal:** Combine messages as state, chat models, tool binding, and the `MessagesState` reducer into a working LLM chain.

---

## The Big Picture

The simple graph used a plain string as state. Real LLM apps need **conversation history** — a list of messages. This notebook introduces:
1. **Message types** — `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`
2. **Chat models in nodes** — using an LLM inside a graph node
3. **Tool binding** — giving the LLM access to functions
4. **Reducers** — the mechanism that makes messages **append** instead of overwrite
5. **`MessagesState`** — a pre-built shortcut for message-based state

---

## 1. Messages

```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content="So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content="Yes, that's right.", name="Abood"))
messages.append(AIMessage(content="Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content="I want to learn about the best place to see Orcas in the US.", name="Abood"))

for m in messages:
    m.pretty_print()
```

**Line-by-line:**
- `AIMessage` — represents the **model's** response
- `HumanMessage` — represents the **user's** input
- `name` — optional metadata to identify who said what
- `.pretty_print()` — formats the message with a role header like `Human Message` or `Ai Message`

**Message types in LangChain:**

| Type | Role | When used |
|------|------|-----------|
| `HumanMessage` | User | What the user says |
| `AIMessage` | Assistant | What the model responds |
| `SystemMessage` | System | Instructions for the model's behavior |
| `ToolMessage` | Tool | Result of a tool call |

---

## 2. Chat Models

```python
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)
result = llm.invoke(messages)
```

- `ChatMistralAI(...)` — initializes the model. Replace with `ChatOpenAI(model="gpt-4o")` for OpenAI.
- `.invoke(messages)` — sends the entire message list to the model. The model sees the full conversation.
- Returns an `AIMessage` with `content` (the reply) and `response_metadata` (token usage, finish reason, etc.).

```python
result.response_metadata
# {'token_usage': {'prompt_tokens': 49, 'completion_tokens': 1510, ...},
#  'model_name': 'mistral-medium-latest',
#  'finish_reason': 'stop'}
```

---

## 3. Tool Binding

```python
def multiply(x: int, y: int) -> int:
    """Multiply a and b.

    Args:
        x: first int
        y: second int
        return: the product of x and y
    """
    return x * y

llm_with_tools = llm.bind_tools([multiply])
```

**Line-by-line:**
- `multiply` is a plain Python function. The **docstring and type hints** are critical — LangChain uses them to generate the tool schema that the LLM sees.
- `llm.bind_tools([multiply])` — creates a new model instance that is **aware** of the `multiply` tool. The model can now choose to call it.

**Testing it:**
```python
tool_call = llm_with_tools.invoke([HumanMessage(content="What is 5 multiplied by 10?")])
tool_call.tool_calls
# [{'name': 'multiply', 'args': {'x': 5, 'y': 10}, 'id': 'YvzU1L78o', 'type': 'tool_call'}]
```

- The model doesn't execute the function — it returns a **structured request** to call it.
- `tool_calls` is a list because the model could request multiple tool calls at once.

---

## 4. Messages as State — The Problem

```python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessageState(TypedDict):
    messages: list[AnyMessage]
```

This looks right, but there's a **critical flaw**: by default, LangGraph **overwrites** state keys. If node A returns `{"messages": [msg1]}` and node B returns `{"messages": [msg2]}`, you end up with only `[msg2]` — the conversation history is lost!

---

## 5. The Solution: Reducers

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**Line-by-line:**
- `Annotated[type, reducer]` — Python's `Annotated` lets you attach metadata to a type.
- `add_messages` — a built-in reducer function that **appends** new messages instead of overwriting.
- Now when a node returns `{"messages": [new_msg]}`, it gets **added** to the existing list.

**How `add_messages` works in isolation:**
```python
initial_state = [AIMessage(content="Hello!"), HumanMessage(content="Hi!")]
new_messages = [AIMessage(content="How can I help?")]
add_messages(initial_state, new_messages)
# -> [AIMessage("Hello!"), HumanMessage("Hi!"), AIMessage("How can I help?")]
```

---

## 6. The Shortcut: `MessagesState`

```python
from langgraph.graph import MessagesState

class MessageState(MessagesState):
    pass
```

**`MessagesState` is exactly equivalent to the custom `TypedDict` above.** It comes with:
- A pre-built `messages` key
- The `add_messages` reducer already attached
- You can extend it by adding more keys if needed

**Why use it?** Less boilerplate. The three approaches compared:

| Approach | Code | Behavior |
|----------|------|----------|
| Plain `TypedDict` | `messages: list[AnyMessage]` | **Overwrites** (broken!) |
| `TypedDict` + reducer | `messages: Annotated[list[AnyMessage], add_messages]` | **Appends** (correct) |
| `MessagesState` | `class MyState(MessagesState): pass` | **Appends** (correct, less code) |

---

## 7. The Chain Graph

```python
from langgraph.graph import StateGraph, START, END

def tool_calling_llm(state: MessageState) -> MessageState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()
```

**Line-by-line:**
- `tool_calling_llm(state)` — the node function. It:
  1. Reads all messages from state: `state["messages"]`
  2. Sends them to the LLM: `llm_with_tools.invoke(...)`
  3. Wraps the response in a list and returns it: `{"messages": [result]}`
  4. The `add_messages` reducer appends this to the existing messages
- The graph is linear: `START → tool_calling_llm → END`

**Testing with a greeting:**
```python
graph.invoke({"messages": [HumanMessage(content="Hello!")]})
# Human: Hello!
# AI: Hello! How can I assist you today?
```

**Testing with a tool-triggering question:**
```python
graph.invoke({"messages": [HumanMessage(content="what is 5 multiplied by 10?")]})
# Human: what is 5 multiplied by 10?
# AI: [Tool Call: multiply(x=5, y=10)]
```

The model returns a tool call — but the graph ends here. The tool is **not executed**. That's what the Router (next notebook) solves.

---

## Key Takeaways

1. **Messages** are typed objects with roles — the foundation of LLM conversations
2. **Tool binding** gives the LLM awareness of functions it can call (via docstrings and type hints)
3. **Default state behavior is overwrite** — this destroys conversation history
4. **Reducers** (via `Annotated`) control how state updates happen — `add_messages` appends
5. **`MessagesState`** is the recommended shortcut — pre-built `messages` key with `add_messages`
6. This chain can **decide** to call a tool, but can't **execute** it yet
