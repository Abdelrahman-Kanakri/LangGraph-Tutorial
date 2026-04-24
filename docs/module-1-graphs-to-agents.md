# Module 1 — From Simple Graphs to Deployed Agents

A progressive build: static graph → LLM chain → LLM with tool routing → full ReAct agent → agent with memory → deployed service.

---

## 01 · simple-graph

**Notebook:** [simple_graph.ipynb](../module-1/01-simple-graph/simple_graph.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/01-simple-graph/walkthrough.md)

**What it teaches**
The three primitives of a LangGraph: **state** (a `TypedDict`), **nodes** (plain Python functions `(state) → dict`), and **edges** (normal or conditional). No LLM yet — just the graph mechanics.

**Key APIs**

- `class State(TypedDict): graph_state: str`
- `StateGraph(State)` → `.add_node(name, fn)` → `.add_edge(a, b)` → `.compile()`
- `.add_conditional_edges("node", router_fn)` where `router_fn` returns `Literal["nodeA", "nodeB"]`
- `START`, `END` — built-in special nodes
- `graph.invoke({"graph_state": "..."})` — runs START → END synchronously
- `graph.get_graph().draw_mermaid_png()` — Mermaid visualization

**When to use**
Any workflow with **deterministic branching** and no LLM — validators, form processors, ETL pipelines.

**Gotcha**
Node returns **overwrite** by default. Returning `{"graph_state": "x"}` replaces the value — does not append.

---

## 02 · chain

**Notebook:** [chain.ipynb](../module-1/02-chain/chain.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/02-chain/walkthrough.md)

**What it teaches**
Move from a plain string to message-based state. Introduces the `add_messages` reducer, `MessagesState`, and tool binding on a chat model. The LLM can *decide* to call a tool but the tool isn't executed yet.

**Key APIs**

- `from langgraph.graph import MessagesState` — pre-built `TypedDict` with `messages: Annotated[list, add_messages]`
- `llm.bind_tools([fn1, fn2, ...])` — gives the model function schemas to call
- Node returns **list form**: `{"messages": [llm.invoke(state["messages"])]}`
- `add_messages` reducer: appends messages; overwrites by id; removes via `RemoveMessage`

**When to use**
Simplest possible LLM integration inside a graph — a one-shot call, optionally with tool awareness.

**Best practice**
Always wrap returned messages in a list. The raw form works (thanks to `add_messages` being permissive) but breaks under stricter reducers.

---

## 03 · router

**Notebook:** [router.ipynb](../module-1/03-router/router.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/03-router/walkthrough.md)

**What it teaches**
Adds the execution half of tool calling. Uses the pre-built `ToolNode` and `tools_condition` so the LLM can *actually* run a tool — but only once, then the graph ends.

**Key APIs**

- `from langgraph.prebuilt import ToolNode, tools_condition`
- `builder.add_node("tools", ToolNode([multiply, ...]))`
- `builder.add_conditional_edges("assistant", tools_condition)` — routes to `"tools"` if LLM called one, else to `END`

**When to use**
One-shot tool use: answer a query that may require a single tool call, without follow-up reasoning.

**Limitation**
Tool result never goes back to the LLM. For multi-step reasoning, use the Agent pattern (next notebook).

---

## 04 · agent

**Notebook:** [agent.ipynb](../module-1/04-agent/agent.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/04-agent/walkthrough.md)

**What it teaches**
The **ReAct loop**: `assistant → tools → assistant → …` until the LLM responds without a tool call. One edge change compared to the router (`tools → assistant` instead of `tools → END`) turns a one-shot caller into a reasoning agent.

**Key APIs**

- `SystemMessage(content="You are a helpful assistant …")` — prepended in the node, not stored
- `builder.add_edge("tools", "assistant")` — the loop-closing edge
- `tools_condition` terminates the loop when the LLM produces a plain `AIMessage`

**When to use**
Any task that needs **multi-step reasoning with tool chaining** — arithmetic pipelines, research, multi-API workflows.

**Mental model**
Reason → Act → Observe, repeat. The LLM sees tool results and decides whether to call another tool or finish.

---

## 05 · agent-memory

**Notebook:** [agent-memory.ipynb](../module-1/05-agent-memory/agent-memory.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/05-agent-memory/walkthrough.md)

**What it teaches**
Persistent multi-turn conversations by adding a **checkpointer**. State is saved after every step, keyed by `thread_id`. Two code changes turn a stateless agent into a memoryful one.

**Key APIs**

- `from langgraph.checkpoint.memory import MemorySaver`
- `graph = builder.compile(checkpointer=MemorySaver())`
- `config = {"configurable": {"thread_id": "1"}}`
- `graph.invoke(input, config)` — same `thread_id` = same conversation

**When to use**
Any chatbot or agent that should remember across turns. `MemorySaver` for notebooks/tests, `SqliteSaver`/`PostgresSaver` for persistence.

**Separation of concerns**
Graph logic and storage backend are fully decoupled — swap checkpointers without touching nodes/edges.

---

## 06 · deployment

**Notebook:** [deployment.ipynb](../module-1/06-deployment/deployment.ipynb) · **Walkthrough:** [walkthrough.md](../module-1/06-deployment/walkthrough.md)

**What it teaches**
Run the agent as a service — locally via `langgraph dev` (Studio UI + HTTP API) and in production via LangGraph Cloud. Interact via the SDK instead of in-process Python.

**Key APIs**

- `langgraph dev` (CLI) — local server at `http://127.0.0.1:2024` + Studio UI
- `from langgraph_sdk import get_client` → `client = get_client(url=URL)`
- `thread = await client.threads.create()` — server-side thread management
- `client.runs.stream(thread_id, assistant_id="agent", input=..., stream_mode="values")`

**When to use**
Any time you want a non-notebook consumer of your graph — a web app, a Slack bot, another service. Also the path to production.

**Portability**
The same SDK code works for local dev and cloud deployment — only the URL changes.
