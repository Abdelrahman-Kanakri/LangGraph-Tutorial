# LangGraph Tutorial - From Basics to Agents

A hands-on, notebook-by-notebook walkthrough of [LangGraph](https://github.com/langchain-ai/langgraph) - the framework for building stateful, multi-step AI agent workflows. This tutorial covers everything from simple graphs to full ReAct agents with memory and deployment.

> **Note:** This tutorial uses **Mistral AI** models (`mistral-small`, `mistral-medium`) instead of OpenAI. You can swap in any LangChain-compatible chat model.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Module 0 - Basics](#module-0---basics)
- [Module 1 - Simple Graphs to Agents](#module-1---simple-graphs-to-agents)
  - [1.1 Simple Graph](#11-simple-graph)
  - [1.2 Chain](#12-chain)
  - [1.3 Router](#13-router)
  - [1.4 Agent](#14-agent)
  - [1.5 Agent with Memory](#15-agent-with-memory)
  - [1.6 Deployment](#16-deployment)
- [Module 2 - State and Memory (In Progress)](#module-2---state-and-memory-in-progress)
  - [2.1 State Schema](#21-state-schema)
  - [2.2 State Reducers](#22-state-reducers)
- [Key Concepts at a Glance](#key-concepts-at-a-glance)
- [Running LangGraph Studio](#running-langgraph-studio)
- [Resources](#resources)

---

## Prerequisites

- Python 3.11+
- [Docker](https://www.docker.com/) (for LangGraph Studio)
- API keys:
  - `MISTRAL_API_KEY` (or `OPENAI_API_KEY` if you swap models)
  - `TAVILY_API_KEY` (for search tools - used in later modules)
  - `LANGSMITH_API_KEY` (optional, for tracing & deployment)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/LangGraph-Tutorial.git
cd LangGraph-Tutorial

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### LangGraph Studio (Optional)

To run the visual Studio IDE locally:

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev  # run inside any module's /studio directory
```

This starts a local server at `http://127.0.0.1:2024` and opens the Studio UI in your browser.

---

## Module 0 - Basics

> **Notebook:** [`module-0/01-basics/basics.ipynb`](module-0/01-basics/basics.ipynb) | **Walkthrough:** [`walkthrough.md`](module-0/01-basics/walkthrough.md)

This module sets up the environment and introduces the foundational building blocks used throughout the tutorial.

### What You'll Learn

**1. Chat Models** - LangChain provides a unified interface for chat models. You instantiate one and call `.invoke()` with messages:

```python
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(model="mistral-medium", temperature=0)
llm.invoke("hello world")
# Returns: AIMessage(content="Hello, World! ...")
```

Key parameters:
- `model` - which model to use
- `temperature` - controls randomness (0 = deterministic, 1 = creative)

**2. Messages** - Chat models work with typed messages that represent conversation roles:

```python
from langchain_core.messages import HumanMessage
msg = HumanMessage(content="Hello world", name="Lance")
llm.invoke([msg])  # Pass a list of messages
```

Message types: `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`

**3. Search Tools** - Tavily is a search engine optimized for LLMs:

```python
from langchain_tavily import TavilySearch
tavily_search = TavilySearch(max_results=3)
results = tavily_search.invoke("What is LangGraph?")
```

---

## Module 1 - Simple Graphs to Agents

This module progressively builds from the simplest possible graph to a full ReAct agent with memory.

### 1.1 Simple Graph

> **Notebook:** [`module-1/01-simple-graph/simple_graph.ipynb`](module-1/01-simple-graph/simple_graph.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/01-simple-graph/walkthrough.md)

Builds a graph with 3 nodes and a conditional edge to introduce the core LangGraph concepts.

**State** - Every graph needs a state schema. Here we use `TypedDict`:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

The state is shared across all nodes. Each node reads from it and returns updates to it.

**Nodes** - Plain Python functions. They receive the current state and return a dict of updates:

```python
def node1(state: State) -> State:
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}

def node2(state: State) -> State:
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " Happy!"}

def node3(state: State) -> State:
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " Sad!"}
```

> By default, the returned value **overwrites** the existing state key.

**Edges** - Connect nodes together. LangGraph supports two types:

| Edge Type | Purpose | Example |
|-----------|---------|---------|
| **Normal Edge** | Always routes to a specific node | `node1 -> node2` |
| **Conditional Edge** | Routes based on logic/state | `node1 -> node2 OR node3` |

```python
# Conditional edge function - returns the name of the next node
def decide_node(state: State) -> Literal["node2", "node3"]:
    if random.random() < 0.5:
        return "node2"
    return "node3"
```

**Building & Running the Graph:**

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_node)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

graph = builder.compile()
graph.invoke({"graph_state": "Hi There, I'm Abood"})
# Output: {'graph_state': "Hi There, I'm Abood I am Happy!"}
```

`START` is where user input enters the graph. `END` is the terminal node.

---

### 1.2 Chain

> **Notebook:** [`module-1/02-chain/chain.ipynb`](module-1/02-chain/chain.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/02-chain/walkthrough.md)

Combines four key concepts into a working LLM chain: **messages as state**, **chat models in nodes**, **tool binding**, and **tool execution**.

**Messages as State** - Instead of a simple string, we now use a list of messages as our graph state. First attempt with `TypedDict`:

```python
from langchain_core.messages import AnyMessage

class MessageState(TypedDict):
    messages: list[AnyMessage]
```

**The Problem** - With the above, each node's return **overwrites** the message list. We lose the conversation history!

**The Solution: Reducers** - We use `Annotated` with `add_messages` to **append** instead of overwrite:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**The Shortcut: `MessagesState`** - Since message lists are so common, LangGraph provides a pre-built class:

```python
from langgraph.graph import MessagesState

# This is equivalent to the custom TypedDict above!
# It has a pre-built 'messages' key with the add_messages reducer.
```

> **Key Difference:** `TypedDict` with plain `list[AnyMessage]` **overwrites** messages. `MessagesState` (or using `Annotated[list[AnyMessage], add_messages]`) **appends** them.

**Tool Binding** - We can give the LLM access to tools:

```python
def multiply(x: int, y: int) -> int:
    """Multiply x and y."""
    return x * y

llm_with_tools = llm.bind_tools([multiply])
```

When invoked, the model decides whether to call a tool or respond directly:

```python
result = llm_with_tools.invoke("What is 5 multiplied by 10?")
result.tool_calls
# [{'name': 'multiply', 'args': {'x': 5, 'y': 10}, ...}]
```

**The Chain Graph** - A single-node graph that invokes the LLM with tools:

```python
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()
```

This graph can answer questions directly OR return tool calls - but it doesn't execute them yet.

---

### 1.3 Router

> **Notebook:** [`module-1/03-router/router.ipynb`](module-1/03-router/router.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/03-router/walkthrough.md)

Extends the chain to actually **route** between tool execution and direct response.

**Two New Concepts:**

1. **`ToolNode`** - A pre-built node that executes tool calls:
   ```python
   from langgraph.prebuilt import ToolNode
   ToolNode([multiply])  # Pass your tools list
   ```

2. **`tools_condition`** - A pre-built conditional edge that checks if the LLM returned a tool call:
   ```python
   from langgraph.prebuilt import tools_condition
   # Routes to "tools" node if tool call, END if not
   ```

**The Router Graph:**

```python
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
```

**Flow:**
- User asks "What is 5 x 10?" -> LLM returns tool call -> `tools_condition` routes to `tools` node -> Tool executes -> Returns `ToolMessage(content="50")`
- User asks "Hello!" -> LLM responds directly -> `tools_condition` routes to `END`

> **Limitation:** The tool result goes to `END`, not back to the LLM. The user gets a raw `ToolMessage`, not a natural language answer.

---

### 1.4 Agent

> **Notebook:** [`module-1/04-agent/agent.ipynb`](module-1/04-agent/agent.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/04-agent/walkthrough.md)

Upgrades the router into a full **ReAct agent** by creating a loop between the LLM and tools.

**The ReAct Pattern:**
1. **Act** - The model calls a tool
2. **Observe** - The tool result is passed back to the model
3. **Reason** - The model decides: call another tool, or respond to the user

**The Key Change** - Instead of `tools -> END`, we connect `tools -> assistant`, creating a loop:

```python
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")  # <-- THE LOOP

graph = builder.compile()
```

**Multiple Tools** - The agent now has `multiply`, `add`, and `divide`:

```python
tools = [multiply, add, divide]
llm_with_tools = llm.bind_tools(tools)
```

**System Message** - We guide the LLM's behavior without adding it to state:

```python
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic.")

def assistant(state: MessagesState):
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}
    # sys_msg is prepended every call, but NEVER stored in state
```

**Multi-Step Execution Example:**

```
Input: "Add 3 and 4. Multiply the output by 2. Divide the output by 5"

Step 1: LLM calls add(3, 4) -> 7
Step 2: LLM calls multiply(7, 2) -> 14
Step 3: LLM calls divide(14, 5) -> 2.8
Step 4: LLM responds "The final result is 2.8"
```

The loop continues until the model responds without a tool call, at which point `tools_condition` routes to `END`.

---

### 1.5 Agent with Memory

> **Notebook:** [`module-1/05-agent-memory/agent-memory.ipynb`](module-1/05-agent-memory/agent-memory.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/05-agent-memory/walkthrough.md)

Adds **persistence** so the agent remembers previous conversations.

**The Problem** - Graph state is **transient** - it only lives for one `.invoke()` call:

```python
graph.invoke({"messages": [HumanMessage("Add 3 and 4.")]})     # -> 7
graph.invoke({"messages": [HumanMessage("Multiply that by 2.")]})  # -> "What is 'that'?"
```

The agent has no memory of the previous interaction!

**The Solution: Checkpointer** - `MemorySaver` saves graph state after each step:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

**Thread IDs** - Each conversation gets a unique thread:

```python
config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": [HumanMessage("Add 3 and 4.")]}, config=config)       # -> 7
graph.invoke({"messages": [HumanMessage("Multiply that by 2.")]}, config=config) # -> 14
graph.invoke({"messages": [HumanMessage("Divide that by 5.")]}, config=config)   # -> 2.8
```

Now the agent remembers! Each invocation with the same `thread_id` continues the conversation.

**How It Works:**
- The checkpointer saves state at every step of the graph
- Checkpoints are stored in a thread (identified by `thread_id`)
- On the next invocation, the graph loads the previous state and continues from there

---

### 1.6 Deployment

> **Notebook:** [`module-1/06-deployment/deployment.ipynb`](module-1/06-deployment/deployment.ipynb) | **Walkthrough:** [`walkthrough.md`](module-1/06-deployment/walkthrough.md)

Shows how to deploy the agent locally with LangGraph Studio and to the cloud with LangGraph Cloud.

**Key Concepts:**

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Python library for building agent workflows |
| **LangGraph API** | Bundles graph code + task queue + persistence |
| **LangSmith Deployment** | Hosted cloud service for the LangGraph API |
| **LangSmith Studio** | Visual IDE for testing graphs (runs locally or cloud) |
| **LangGraph SDK** | Python client for interacting with deployed graphs |

**Local Deployment:**

```bash
cd module-1/studio
langgraph dev
# API: http://127.0.0.1:2024
# Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

**Interacting via SDK:**

```python
from langgraph_sdk import get_client
client = get_client(url="http://127.0.0.1:2024")

thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"], "agent",
    input={"messages": [HumanMessage(content="Multiply 3 by 2.")]},
    stream_mode="values",
):
    print(chunk.data["messages"][-1])
```

---

## Module 2 - State and Memory (In Progress)

This module dives deeper into how state works in LangGraph.

### 2.1 State Schema

> **Notebook:** [`module-2 /01-state-schema/state-schema.ipynb`](module-2%20/01-state-schema/state-schema.ipynb) | **Walkthrough:** [`walkthrough.md`](module-2%20/01-state-schema/walkthrough.md)

Compares three ways to define state: **TypedDict**, **Dataclass**, and **Pydantic**.

#### TypedDict

```python
from typing_extensions import TypedDict

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]
```

- Provides **type hints** only (for IDEs and static checkers like mypy)
- Does **NOT** enforce types at runtime
- Access via `state["name"]` (dict subscript)
- Invoke with a plain dict: `graph.invoke({"name": "Alice"})`

#### Dataclass

```python
from dataclasses import dataclass

@dataclass
class DataClassState:
    name: str
    mood: Literal["Happy", "Sad"]
```

- Provides **type hints** only - no runtime enforcement
- Access via `state.name` (attribute access)
- Invoke with the class: `graph.invoke(DataClassState(name="Abood", mood="sad"))`
- Nodes still return dicts for updates: `return {"name": state.name + " is ... "}`

#### Pydantic

```python
from pydantic import BaseModel, field_validator

class PydanticState(BaseModel):
    name: str
    mood: str

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value):
        if value not in ["happy", "sad"]:
            raise ValueError("Mood must be either 'happy' or 'sad'")
        return value
```

- **Enforces types and constraints at runtime** with `ValidationError`
- Invoke with the class: `graph.invoke(PydanticState(name="Mustafa", mood="happy"))`
- Invalid data raises an error immediately:
  ```python
  PydanticState(name="Mustafa", mood="happsy")
  # ValidationError: Mood must be either 'happy' or 'sad'
  ```

#### Comparison Table

| Feature | TypedDict | Dataclass | Pydantic |
|---------|-----------|-----------|----------|
| Runtime validation | No | No | **Yes** |
| Type hints | Yes | Yes | Yes |
| State access | `state["key"]` | `state.key` | `state.key` |
| Invoke with | `dict` | Instance | Instance |
| Node return type | `dict` | `dict` | `dict` |
| Best for | Prototyping | Clean syntax | Production / validation |

---

### 2.2 State Reducers

> **Notebook:** [`module-2 /02-state-reducers/state-reducers.ipynb`](module-2%20/02-state-reducers/state-reducers.ipynb) | **Walkthrough:** [`walkthrough.md`](module-2%20/02-state-reducers/walkthrough.md)

Explains **how** state updates happen - the mechanism behind appending vs. overwriting.

#### Default Behavior: Overwrite

Without a reducer, returning a value from a node **overwrites** the existing state:

```python
class State(TypedDict):
    foo: int

def node_1(state):
    return {"foo": state["foo"] + 1}

graph.invoke({"foo": 1})  # -> {'foo': 2}
```

#### The Problem: Parallel Nodes

When two nodes run in parallel and both try to overwrite the same key, LangGraph throws an error:

```python
# node_1 branches to node_2 AND node_3 (parallel)
# Both return {"foo": state["foo"] + 1}
graph.invoke({"foo": 1})
# InvalidUpdateError: Can receive only one value per step.
# Use an Annotated key to handle multiple values.
```

#### The Solution: Reducers with `Annotated`

Reducers define **how** updates are combined. Using `operator.add` on lists:

```python
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]  # <-- reducer!

def node_1(state):
    return {"foo": [state["foo"][-1] + 1]}

# With parallel nodes:
graph.invoke({"foo": [1]})
# -> {'foo': [1, 2, 3, 3]}  (1=input, 2=node1, 3=node2, 3=node3)
```

#### Custom Reducers

Handle edge cases like `None` inputs:

```python
def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class CustomReducerState(TypedDict):
    foo: Annotated[list, reduce_list]
```

#### The `add_messages` Reducer

The built-in reducer for message lists. Three superpowers:

**1. Append** - New messages are added to the list:
```python
initial = [AIMessage(content="Hello!", id="1")]
new = [HumanMessage(content="Hi!", id="2")]
add_messages(initial, new)
# -> [AIMessage("Hello!"), HumanMessage("Hi!")]
```

**2. Overwrite by ID** - A message with the same ID replaces the original:
```python
initial = [HumanMessage(content="I like cats", id="2")]
new = [HumanMessage(content="I like dogs", id="2")]
add_messages(initial, new)
# -> [HumanMessage("I like dogs")]  # replaced!
```

**3. Remove by ID** - Use `RemoveMessage` to delete messages:
```python
from langchain_core.messages import RemoveMessage
messages = [AIMessage("Hi", id="1"), HumanMessage("Hello", id="2")]
delete = [RemoveMessage(id="1")]
add_messages(messages, delete)
# -> [HumanMessage("Hello")]  # AIMessage removed!
```

#### `MessagesState` vs. Custom TypedDict

These two are **equivalent**:

```python
# Option A: Manual (verbose)
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str

# Option B: Pre-built (concise)
class ExtendedMessagesState(MessagesState):
    added_key_1: str  # just add extra keys
```

`MessagesState` is the recommended choice - it gives you the `messages` key with `add_messages` reducer out of the box.

---

## Key Concepts at a Glance

### The Progression

```
Simple Graph -> Chain -> Router -> Agent -> Agent + Memory
     |            |         |         |           |
  TypedDict    Messages  ToolNode   ReAct      Checkpointer
  + Nodes      + LLM     + Cond.    Loop       + thread_id
  + Edges      + Tools    Edge
```

### When to Use What

| Scenario | Pattern |
|----------|---------|
| Fixed workflow with branching | Simple Graph (Module 1.1) |
| LLM call with tool awareness | Chain (Module 1.2) |
| LLM decides tool vs. response | Router (Module 1.3) |
| Multi-step reasoning with tools | Agent / ReAct (Module 1.4) |
| Persistent conversations | Agent + Memory (Module 1.5) |
| Production deployment | Deployment (Module 1.6) |

### State Schema Decision Guide

| Need | Use |
|------|-----|
| Quick prototyping | `TypedDict` |
| Clean attribute access | `dataclass` |
| Runtime validation | `Pydantic` |
| Message-based state | `MessagesState` |
| Custom update logic | `Annotated` + reducer |

---

## Running LangGraph Studio

Each module has a `studio/` directory with deployment-ready code:

```bash
cd module-1/studio
langgraph dev
```

The `langgraph.json` in each studio directory defines which graphs are available. The Studio UI lets you visually test and debug your graphs.

---

## Project Structure

```
LangGraph-Tutorial/
├── module-0/
│   └── 01-basics/
│       ├── basics.ipynb              # Environment setup & LangChain basics
│       └── walkthrough.md            # Line-by-line explanation
├── module-1/
│   ├── 01-simple-graph/
│   │   ├── simple_graph.ipynb        # Nodes, edges, conditional edges
│   │   └── walkthrough.md
│   ├── 02-chain/
│   │   ├── chain.ipynb               # Messages, tools, reducers, MessagesState
│   │   └── walkthrough.md
│   ├── 03-router/
│   │   ├── router.ipynb              # ToolNode, tools_condition
│   │   └── walkthrough.md
│   ├── 04-agent/
│   │   ├── agent.ipynb               # ReAct loop with multiple tools
│   │   └── walkthrough.md
│   ├── 05-agent-memory/
│   │   ├── agent-memory.ipynb        # MemorySaver, thread_id persistence
│   │   └── walkthrough.md
│   ├── 06-deployment/
│   │   ├── deployment.ipynb          # Studio & cloud deployment
│   │   └── walkthrough.md
│   └── studio/                       # Deployable graph definitions
├── module-2 /  (in progress)
│   ├── 01-state-schema/
│   │   ├── state-schema.ipynb        # TypedDict vs Dataclass vs Pydantic
│   │   └── walkthrough.md
│   └── 02-state-reducers/
│       ├── state-reducers.ipynb      # Reducers, add_messages, RemoveMessage
│       └── walkthrough.md
├── academy_notebooks/                # Original LangChain Academy reference (modules 2-6)
├── requirements.txt
└── langgraph.json
```

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Academy](https://academy.langchain.com/)
- [LangSmith](https://smith.langchain.com/) - Tracing & Deployment
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)

---

**Happy Learning!** If you found this helpful, feel free to star the repo and share it with your network.
