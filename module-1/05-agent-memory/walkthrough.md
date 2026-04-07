# Walkthrough: Agent with Memory

> **Notebook:** `agent-memory.ipynb`
> **Goal:** Add **persistence** to the agent so it remembers previous conversations across multiple `.invoke()` calls.

---

## The Big Picture

The Agent from the previous notebook works great for a single request. But each `.invoke()` starts fresh — the agent has no memory of past interactions. This notebook introduces:

1. **The memory problem** — state is transient by default
2. **Checkpointers** — save graph state automatically after each step
3. **Thread IDs** — organize conversations into separate threads

---

## 1. The Same Agent (Recap)

The notebook rebuilds the exact same agent from the previous lesson:

```python
tools = [multiply, add, divide]
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic.")

def assistant(state: MessagesState):
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()
```

Nothing new here — same ReAct loop.

---

## 2. The Memory Problem

```python
# First call
messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph.invoke({"messages": messages})
# -> "The sum of 3 and 4 is 7."

# Second call — tries to reference the previous result
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph.invoke({"messages": messages})
# -> "I don't have any prior input or context. Could you clarify?"
```

**Why it fails:** Graph state is **transient** — it only exists during a single `.invoke()` call. When the second call starts, the state is empty. The agent has no idea what "that" refers to.

---

## 3. The Solution: Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

**Line-by-line:**
- `MemorySaver()` — an in-memory key-value store that saves graph state after every step
- `builder.compile(checkpointer=memory)` — the **only change** to enable persistence. Same graph, same nodes, same edges — just compiled with a checkpointer.

**What `MemorySaver` does:**
- After every node executes, the checkpointer **saves a snapshot** of the full state
- These snapshots are organized by **thread ID** (see next section)
- On the next `.invoke()`, the graph **loads the previous state** and continues from there

> `MemorySaver` stores everything in RAM. For production, use `SqliteSaver` or `PostgresSaver` for persistent storage.

---

## 4. Thread IDs

```python
config = {"configurable": {"thread_id": "1"}}
```

- `thread_id` identifies a **conversation thread**. Think of it like a chat session ID.
- All invocations with the same `thread_id` share the same state history.
- Different `thread_id`s are completely independent conversations.

---

## 5. Multi-Turn Conversation with Memory

**First turn:**

```python
messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph_memory.invoke({"messages": messages}, config=config)
# -> "The sum of 3 and 4 is 7."
```

State saved: `[HumanMessage("Add 3 and 4"), AIMessage(tool_call: add), ToolMessage(7), AIMessage("7")]`

**Second turn (same `thread_id`):**

```python
messages = [HumanMessage(content="Multiply that by 2")]
messages = react_graph_memory.invoke({"messages": messages}, config=config)
```

**What happens under the hood:**
1. Checkpointer loads the saved state for thread `"1"` → `[Human, AI, Tool, AI]`
2. New `HumanMessage("Multiply that by 2")` is **appended** via `add_messages`
3. The LLM now sees the full history: it knows "that" = 7
4. LLM calls `multiply(7, 2)` → 14
5. State is saved again

**Output shows the full conversation:**
```
Human: Add 3 and 4.
AI: Tool Call → add(3, 4)
Tool: 7
AI: The sum of 3 and 4 is 7.
Human: Multiply that by 2      ← new message appended
AI: Tool Call → multiply(7, 2)
Tool: 14
AI: The product of 7 and 2 is 14.
```

**Third turn:**

```python
messages = [HumanMessage(content="Divide that by 5.")]
messages = react_graph_memory.invoke({"messages": messages}, config=config)
# -> "The quotient of 14 divided by 5 is 2.8."
```

The entire conversation chain is preserved: 3+4=7, 7*2=14, 14/5=2.8.

---

## Without Memory vs. With Memory

| Feature | Without Memory | With Memory |
|---------|---------------|-------------|
| State lifetime | Single `.invoke()` | Across all `.invoke()` calls |
| Compilation | `builder.compile()` | `builder.compile(checkpointer=memory)` |
| Invocation | `graph.invoke(input)` | `graph.invoke(input, config=config)` |
| Config needed | No | Yes (`thread_id` required) |
| Multi-turn | Broken ("what is 'that'?") | Works ("that" = previous result) |

---

## How It Works Under the Hood

```
Thread "1":
┌──────────────────────────────────────────────────┐
│ Checkpoint 1: [Human("Add 3 and 4")]             │
│ Checkpoint 2: [Human, AI(tool_call: add)]        │
│ Checkpoint 3: [Human, AI, Tool(7)]               │
│ Checkpoint 4: [Human, AI, Tool, AI("7")]         │
│ Checkpoint 5: [... + Human("Multiply that by 2")]│
│ ...                                              │
└──────────────────────────────────────────────────┘
```

- Every single step creates a checkpoint
- The checkpointer always loads the **latest** checkpoint for a thread
- This is also the foundation for **time travel** (going back to earlier checkpoints) — covered in later modules

---

## Key Takeaways

1. **State is transient by default** — each `.invoke()` starts fresh
2. **`MemorySaver`** adds persistence with zero changes to the graph structure
3. **`thread_id`** organizes conversations — same ID = same conversation
4. The checkpointer saves state **after every step**, not just at the end
5. Multi-turn conversations "just work" once memory is enabled
6. The only code changes: `compile(checkpointer=memory)` and passing `config` to `.invoke()`
