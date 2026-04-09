# Walkthrough: Chatbot with External Database Memory

> **Notebook:** `chatbot-external-memory.ipynb`
> **Goal:** Replace the in-memory checkpointer with **SQLite** so the chatbot's memory persists even after restarting the notebook.

---

## The Big Picture

`MemorySaver` stores everything in RAM — perfect for development, but everything is lost when the process stops. For production or longer experiments, we need a **durable checkpointer** backed by a real database.

This notebook takes the exact same summarization chatbot and swaps `MemorySaver` for `SqliteSaver`.

---

## 1. SQLite Basics

```python
import sqlite3

# Option A: In-memory (same as MemorySaver, but via SQLite)
conn = sqlite3.connect(":memory:", check_same_thread=False)

# Option B: On-disk (persists across restarts!)
db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
```

**Line-by-line:**
- `sqlite3.connect(":memory:")` — creates a temporary in-memory database (gone when process ends)
- `sqlite3.connect("state_db/example.db")` — creates/opens a file-based database (persists on disk)
- `check_same_thread=False` — required because LangGraph may access the connection from different threads

---

## 2. SqliteSaver Checkpointer

```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)
```

- `SqliteSaver` is a drop-in replacement for `MemorySaver`
- It takes a `sqlite3` connection as its argument
- All checkpoint data (state snapshots, thread history) is written to the SQLite database
- Install with: `pip install langgraph-checkpoint-sqlite`

---

## 3. The Chatbot (Same as Before)

The chatbot is identical to the summarization chatbot from the previous notebook:

```python
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END

model = ChatMistralAI(model="mistral-small", temperature=0)

class State(MessagesState):
    summary: str
```

### Nodes

```python
def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END
```

All identical to the previous notebook — the logic hasn't changed at all.

### Compile with SqliteSaver

```python
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

graph = workflow.compile(checkpointer=memory)  # <-- SqliteSaver instead of MemorySaver
```

**The only difference:** `checkpointer=memory` where `memory` is `SqliteSaver(conn)` instead of `MemorySaver()`.

---

## 4. Running the Chatbot

```python
config = {"configurable": {"thread_id": "1"}}

# Turn 1
output = graph.invoke({"messages": [HumanMessage("hi! I'm Abood.")]}, config)
# -> "Hello Abood! It's nice to meet you."

# Turn 2
output = graph.invoke({"messages": [HumanMessage("what's my name?")]}, config)
# -> "Your name is Abood!"

# Turn 3
output = graph.invoke({"messages": [HumanMessage("i like Counter Strike Global Offensive!")]}, config)
# -> "That's awesome, Abood! Counter-Strike: Global Offensive..."
```

---

## 5. The Persistence Advantage

### Check State

```python
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
```

The state is stored in the SQLite file on disk.

### Restart the Notebook Kernel

After restarting:
```python
# Reconnect to the same database
conn = sqlite3.connect("state_db/example.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

# Load the same thread
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
# The full conversation history is still there!
```

With `MemorySaver`, this would return empty state. With `SqliteSaver`, the conversation **survives restarts**.

---

## MemorySaver vs. SqliteSaver

| Feature | MemorySaver | SqliteSaver |
|---------|-------------|-------------|
| **Storage** | RAM | SQLite file on disk |
| **Survives restart?** | No | **Yes** |
| **Setup** | `MemorySaver()` | `SqliteSaver(sqlite3.connect("path"))` |
| **Install** | Built-in | `pip install langgraph-checkpoint-sqlite` |
| **Best for** | Development, testing | Long-running experiments, production |
| **Graph code changes** | None | None (just swap the checkpointer) |

---

## Studio

The same chatbot can be run in LangGraph Studio:

```bash
cd module-2\ /studio
langgraph dev
```

The `studio/chatbot.py` contains the same graph packaged for Studio deployment, and `langgraph.json` registers it.

---

## Key Takeaways

1. **SqliteSaver** is a drop-in replacement for **MemorySaver** — zero changes to graph logic
2. SQLite stores checkpoints in a **file on disk** — state persists across restarts
3. Connect with `sqlite3.connect("path/to/db")` and pass to `SqliteSaver(conn)`
4. For production, LangGraph also supports **PostgreSQL** checkpointers
5. The chatbot logic (summarization, conditional edges, memory) is **completely decoupled** from the storage backend
6. This is the final piece: a chatbot with summarization + durable external memory
