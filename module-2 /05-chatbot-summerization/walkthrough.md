# Walkthrough: Chatbot with Message Summarization

> **Notebook:** `chatbot-summerization.ipynb`
> **Goal:** Build a chatbot that uses an LLM to produce a **running summary** of the conversation, preserving context without sending the entire message history every time.

---

## The Big Picture

Trimming and filtering throw away old messages. But what if those messages contained important context? Summarization offers a middle ground:

- **Keep a compressed summary** of the full conversation
- **Delete old messages** to save tokens
- **Prepend the summary** as a system message so the LLM still has context

This notebook builds a full chatbot with memory and automatic summarization.

---

## 1. Extended State: Adding a `summary` Key

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    summary: str
```

**What's new:**
- We extend `MessagesState` with a custom `summary` key
- `MessagesState` gives us `messages` with the `add_messages` reducer (append behavior)
- `summary` is a plain `str` — it **overwrites** on each update (no reducer, default behavior)

---

## 2. The Conversation Node

```python
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

def call_model(state: State):
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}
```

**Line-by-line:**
- `state.get("summary", "")` — safely get the summary (empty string if none exists yet)
- If a summary exists, create a `SystemMessage` with it and **prepend** it to the messages
- If no summary, just use the messages as-is
- The LLM sees: `[SystemMessage(summary), ...recent_messages]`
- The summary is NOT in `state["messages"]` — it's prepended only at invoke time (same pattern as the system message in Module 1's agent)

---

## 3. The Summarization Node

```python
def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add the prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": response.content, "messages": delete_messages}
```

**Line-by-line:**
- **Build the prompt:** If a summary already exists, ask the LLM to **extend** it. If not, ask it to **create** one.
- `state["messages"] + [HumanMessage(content=summary_message)]` — append the summarization prompt to the full message history so the LLM can see everything.
- `llm.invoke(messages)` — the LLM generates a summary.
- `RemoveMessage(id=m.id) for m in state["messages"][:-2]` — delete all messages **except the last 2**.
- Returns BOTH: the new summary AND the delete instructions.

**What the state looks like after summarization:**
- `summary` = new compressed summary (overwrites the old one)
- `messages` = only the 2 most recent messages (old ones deleted via `RemoveMessage`)

---

## 4. Conditional Edge: When to Summarize

```python
from typing_extensions import Literal

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END
```

**The logic:**
- After every LLM response, check the message count
- If > 6 messages → trigger summarization (compress and clean up)
- If <= 6 → just end (no summarization needed yet)

**Why 6?** It's a threshold to balance context preservation vs. token efficiency. In production, you might use a token count instead.

---

## 5. Build the Graph with Memory

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

workflow = StateGraph(State)

workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
```

**The flow:**

```
START → conversation → [should_continue] → summarize_conversation → END
                              ↓
                             END (if <= 6 messages)
```

- `MemorySaver()` enables multi-turn persistence (same as Module 1's agent-memory)
- Each thread accumulates messages across invocations
- When messages exceed 6, summarization kicks in

---

## 6. Multi-Turn Conversation

```python
config = {"configurable": {"thread_id": "1"}}

# Turn 1
output = graph.invoke({"messages": [HumanMessage("hi! I'm Abood.")]}, config)
# -> "Hi Abood! How can I assist you today?"

# Turn 2
output = graph.invoke({"messages": [HumanMessage("what's my name?")]}, config)
# -> "Your name is Abood!"

# Turn 3
output = graph.invoke({"messages": [HumanMessage("i like the 49ers!")]}, config)
# -> "That's awesome! The San Francisco 49ers..."
```

After 3 turns (6 messages: 3 human + 3 AI), no summary yet:

```python
graph.get_state(config).values.get("summary", "")
# '' (empty — threshold not yet reached)
```

### Turn 4 (Triggers Summarization!)

```python
output = graph.invoke(
    {"messages": [HumanMessage("i like Nick Bosa, isn't he the highest paid defensive player?")]},
    config
)
```

Now there are 8 messages (> 6), so `should_continue` routes to `summarize_conversation`:

```python
graph.get_state(config).values.get("summary", "")
# "The chat started with Abood introducing himself. He shared his support
#  for the San Francisco 49ers. Abood mentioned liking Nick Bosa, and the
#  conversation explored his record-breaking contract..."
```

The summary now captures the entire conversation. Old messages are deleted, only the last 2 remain in state. On the next turn, the summary is prepended as a `SystemMessage`.

---

## How the Cycle Works

```
Turn 1-3:  Messages accumulate normally (≤ 6)
Turn 4:    Messages > 6 → summarize_conversation fires
           → LLM generates summary
           → Old messages deleted (keep last 2)
           → Summary stored in state
Turn 5+:   Summary prepended as SystemMessage
           → Messages accumulate again
           → When > 6 again → summary is EXTENDED, old messages deleted
           ... cycle repeats
```

---

## Key Takeaways

1. **Summarization > trimming** when you need to preserve context from old messages
2. The `summary` key is a plain string that **overwrites** (no reducer) — always the latest summary
3. `RemoveMessage` cleans up old messages after summarization
4. The summary is **prepended as a SystemMessage** at invoke time, never stored in `messages`
5. The conditional edge (`should_continue`) triggers summarization based on message count
6. With `MemorySaver`, the summary persists across turns — enabling long-running conversations
7. The summary is **extended** on each cycle, not rewritten from scratch
