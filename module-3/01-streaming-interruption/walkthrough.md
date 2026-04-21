# Walkthrough: Streaming

> **Notebook:** `streaming-interruption.ipynb`
> **Goal:** Learn the streaming methods LangGraph provides — streaming **full state**, **state updates**, and **individual tokens** from chat models.

---

## The Big Picture

Until now, we've used `.invoke()` — which runs the graph to completion and returns the final state. But real apps need to show progress as it happens: typing indicators, partial responses, streaming tokens.

LangGraph supports streaming at multiple levels:

| Stream mode | What you get |
|-------------|-------------|
| `values` | Full state snapshot after each node |
| `updates` | Only the delta (what each node changed) |
| `astream_events` | Low-level events including individual LLM tokens |
| `messages` (API only) | High-level event types for message streaming |

---

## 1. Setup: The Summarization Chatbot

The notebook rebuilds the chatbot from Module 2 (summarization + checkpointer):

```python
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState

model = ChatMistralAI(model="mistral-medium", temperature=0)

class State(MessagesState):
    summary: str

def call_model(state: State, config: RunnableConfig):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}  # list form
```

**One new detail:** `call_model(state: State, config: RunnableConfig)` — accepting `RunnableConfig` is needed for token-wise streaming on Python < 3.11 (e.g., Colab). It propagates callbacks so LangChain can emit token events.

> **Best practice reminder:** return the LLM response in a list — `{"messages": [response]}`. The original notebook uses the raw form (`{"messages": response}`) which works with `add_messages`, but the list form is the recommended convention. See [Module 1 / Chain walkthrough](../../module-1/02-chain/walkthrough.md#7-the-chain-graph) for full reasoning.

The rest (summarize_conversation, should_continue, graph construction with `MemorySaver`) is identical to Module 2.

---

## 2. Stream Mode: `updates`

```python
config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"messages": [HumanMessage(content="Hi am Abood!")]},
    config=config,
    stream_mode="updates"
):
    print(chunk)
```

**Output:**
```python
{'conversation': {'messages': AIMessage(content='Marhaba, Abood!...')}}
```

**What you get:**
- One chunk per node execution
- The chunk is a dict keyed by the **node name** (e.g., `"conversation"`)
- The value is the dict that the node **returned** — just the delta, not the full state

**Pretty-printing:**

```python
for chunk in graph.stream({...}, config=config, stream_mode="updates"):
    chunk["conversation"]["messages"].pretty_print()
```

Access the message via `chunk[node_name][state_key]`.

---

## 3. Stream Mode: `values`

```python
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---" * 25)
```

**What you get:**
- The **full state** after each node execution
- The chunk is the entire state dict (e.g., `{'messages': [...], 'summary': '...'}`)
- Each chunk includes ALL previous messages, not just the new ones

**`values` vs `updates` — visual:**

```
After node N:
  updates: {node_N_name: {key: delta_value}}     ← just what changed
  values:  {key1: full_val, key2: full_val, ...} ← entire state
```

Use `updates` for efficient progress logs. Use `values` when you need the full picture at each step.

---

## 4. Streaming Tokens with `astream_events`

For token-by-token streaming of LLM output, use `astream_events`:

```python
config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="CS:GO Video Game")

async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    print(f"Node: {event['metadata'].get('langgraph_node','')}. "
          f"Type: {event['event']}. "
          f"Name: {event['name']}")
```

**Each event is a dict with:**

| Key | Meaning |
|-----|---------|
| `event` | Event type (e.g., `on_chat_model_stream`) |
| `name` | Event name (e.g., `ChatMistralAI`) |
| `data` | Event payload (tokens, state, etc.) |
| `metadata` | Contains `langgraph_node` — which node emitted the event |
| `version="v2"` | API version for event streaming |

**Event types you'll see:**
- `on_chain_start` / `on_chain_end` — graph or node start/end
- `on_chat_model_start` / `on_chat_model_end` — LLM call lifecycle
- `on_chat_model_stream` — **individual tokens** as they arrive (this is what we want!)

---

## 5. Filtering Token Events from a Specific Node

```python
node_to_stream = 'conversation'
config = {"configurable": {"thread_id": "4"}}

input_message = HumanMessage(content="CS:GO Video Game")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        print(event["data"])
```

**Line-by-line:**
- `event["event"] == "on_chat_model_stream"` — only token events
- `event['metadata'].get('langgraph_node','') == node_to_stream` — only from the `conversation` node (ignore tokens from the `summarize_conversation` node)
- `event["data"]` contains `{'chunk': AIMessageChunk(content='...')}`

**Printing just the token content (live typing effect):**

```python
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        print(event["data"]["chunk"].content, end="|")
```

The `|` separator reveals each individual token:
```
|The| **San Francisco| 49|ers** are| one of the most| stor|ied|...
```

This is how you build streaming chat UIs.

---

## 6. Streaming via the LangGraph API (SDK)

When the graph is deployed via `langgraph dev`, you stream via the SDK:

```python
from langgraph_sdk import get_client

URL = "http://127.0.0.1:2024"
client = get_client(url=URL)
assistants = await client.assistants.search()

thread = await client.threads.create()
input_message = HumanMessage(content="Multiply 2 and 3")

async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [input_message]},
    stream_mode="values"
):
    print(event)
```

**API-specific objects:**
- `StreamPart(event=..., data=...)` — each streamed chunk
- Messages come as dicts; use `convert_to_messages(messages)` to get LangChain message objects back

---

## 7. The `messages` Stream Mode (API Only)

A higher-level mode specifically designed for message streaming:

```python
async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [input_message]},
    stream_mode="messages"
):
    print(event.event)
```

**Event types:**
- `metadata` — run metadata (run_id, etc.)
- `messages/metadata` — message-level metadata
- `messages/partial` — token-by-token LLM output
- `messages/complete` — fully formed message

This is the cleanest mode for building a chat UI — you get a clear signal for partial tokens vs. complete messages.

---

## Stream Mode Cheat Sheet

| Mode | When to use |
|------|-------------|
| `values` | Show full state at each step (debug, logs) |
| `updates` | Show only what each node changed (efficient progress UI) |
| `astream_events` | Token-by-token streaming, fine-grained LLM events |
| `messages` (API) | High-level event types for chat UIs |

---

## Key Takeaways

1. **`stream_mode="values"`** gives the full state snapshot after each node
2. **`stream_mode="updates"`** gives only the delta each node produced
3. **`astream_events`** unlocks token-level streaming via `on_chat_model_stream` events
4. Filter events by `event['metadata']['langgraph_node']` to get tokens from a specific node only
5. When deployed via `langgraph dev`, the SDK has a built-in `"messages"` stream mode with typed events
6. `RunnableConfig` in node signatures is needed for token streaming on Python < 3.11
7. Streaming is the foundation for **human-in-the-loop** (next notebook: breakpoints)
