# Module 2 — State and Memory

Deep dive into how state is defined, how updates are combined, how to hide internal data, and how to keep conversations manageable over long horizons.

---

## 01 · state-schema

**Notebook:** [state-schema.ipynb](../module-2%20/01-state-schema/state-schema.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/01-state-schema/walkthrough.md)

**What it teaches**
Three ways to define graph state and their trade-offs: **`TypedDict`**, **`dataclass`**, and **`Pydantic`**.

**Key APIs**

- `TypedDict` — `state["key"]` access, type hints only, no runtime checks
- `@dataclass` — `state.key` access, type hints only, no runtime checks
- `BaseModel` (Pydantic) — attribute access, **raises on invalid input**, production-grade validation

**When to use**

| Need | Pick |
|------|------|
| Quick prototyping, notebooks | `TypedDict` |
| Cleaner attribute access, still no validation | `dataclass` |
| Production, untrusted inputs, schema guarantees | `Pydantic` |

**Note**
Nodes always **return a dict**, regardless of schema class. The graph structure (nodes, edges) doesn't change.

---

## 02 · state-reducers

**Notebook:** [state-reducers.ipynb](../module-2%20/02-state-reducers/state-reducers.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/02-state-reducers/walkthrough.md)

**What it teaches**
The mechanism **behind** how state updates combine. Default is overwrite; reducers customize that. Parallel nodes writing the same key **require** a reducer.

**Key APIs**

- `Annotated[list[int], operator.add]` — append via list concatenation
- Custom reducer: a 2-arg `(existing, update) -> new` function attached via `Annotated`
- `add_messages` — the flagship reducer: appends, overwrites by message `id`, removes via `RemoveMessage`
- `RemoveMessage(id=...)` — the sentinel that triggers deletion
- `MessagesState` — `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]` pre-built

**When to use**
Any state key that is **accumulated across nodes** (lists, counters, logs) or written to **in parallel**. Messages always use `add_messages`.

---

## 03 · multiple-schema

**Notebook:** [multiple-schema.ipynb](../module-2%20/03-multiple-schema/multiple-schema.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/03-multiple-schema/walkthrough.md)

**What it teaches**
Three-schema pattern for hiding internal state and controlling the public API of a graph: `OverallState` (everything), `InputState` (what the caller provides), `OutputState` (what the caller receives).

**Key APIs**

- `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)`
- Type-hint node functions with the subset they use: `def node(state: InputState)` vs `def node(state: OverallState)`
- Private state keys live on `OverallState` only — never visible to the user

**When to use**
Public-facing graphs where callers should only see a clean input/output contract, with internal scratchpad data (retrieval docs, intermediate plans, tool state) hidden.

---

## 04 · trim-filter-messages

**Notebook:** [trim-filtering-messages.ipynb](../module-2%20/04-trim-filter-messages%20/trim-filtering-messages.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/04-trim-filter-messages%20/walkthrough.md)

**What it teaches**
Three strategies, increasing in sophistication, for keeping message history cheap and within context limits.

**Key APIs**

| Strategy | Mechanism | Invasive? |
|----------|-----------|-----------|
| `RemoveMessage(id=...)` | Deletes from state via `add_messages` | Yes — state changes |
| Filter at invoke time | `state["messages"][-N:]` inside the node | No — state intact |
| `trim_messages(...)` | Token-count-based trimming from `langchain_core.messages` | No — state intact |

**When to use**

- Need to reclaim memory permanently → **RemoveMessage**
- Just want to cap the LLM's input per call → **filter** or **trim_messages**
- Budget is token-based (not message-count) → **trim_messages**

---

## 05 · chatbot-summerization

**Notebook:** [chatbot-summerization.ipynb](../module-2%20/05-chatbot-summerization/chatbot-summerization.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/05-chatbot-summerization/walkthrough.md)

**What it teaches**
A running **LLM-generated summary** so old messages can be deleted without losing context. The summary is prepended as a `SystemMessage` at invoke time.

**Key APIs**

- Extend `MessagesState` with a `summary: str` key (no reducer — overwrite)
- Two nodes: `conversation` (chat) and `summarize_conversation` (update summary + drop old messages)
- Conditional edge `should_continue` — triggers summarization when messages exceed a threshold
- `RemoveMessage(id=m.id)` for each old message to evict

**When to use**
Long-running chatbots where old context still matters (running jokes, early preferences, domain facts) but sending every message is too expensive.

**Pattern**
Summary *extends* each cycle — the old summary is fed back to the LLM with the new messages to produce the new summary.

---

## 06 · chatbot-external-memory

**Notebook:** [chatbot-external-memory.ipynb](../module-2%20/06-chatbot-external-memory/chatbot-external-memory.ipynb) · **Walkthrough:** [walkthrough.md](../module-2%20/06-chatbot-external-memory/walkthrough.md)

**What it teaches**
Drop-in swap of `MemorySaver` for `SqliteSaver` — same graph, state now persists to disk across process restarts.

**Key APIs**

- `import sqlite3` → `conn = sqlite3.connect("state.db", check_same_thread=False)`
- `from langgraph.checkpoint.sqlite import SqliteSaver`
- `graph = builder.compile(checkpointer=SqliteSaver(conn))`
- Postgres equivalent: `langgraph.checkpoint.postgres.PostgresSaver`

**When to use**
Any deployment beyond a single notebook session — long experiments, production agents, shared state across processes.

**Decoupling**
Graph logic doesn't change at all — the checkpointer interface is uniform across `MemorySaver`, `SqliteSaver`, `PostgresSaver`.
