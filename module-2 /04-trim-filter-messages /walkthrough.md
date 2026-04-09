# Walkthrough: Trim & Filter Messages

> **Notebook:** `trim-filtering-messages.ipynb`
> **Goal:** Learn three strategies to manage long conversation histories — **RemoveMessage**, **filtering**, and **trimming** — to control token usage and latency.

---

## The Big Picture

As conversations grow, so does the message list. Sending hundreds of messages to the LLM on every call means:
- **High token usage** (cost)
- **High latency** (slow responses)
- **Context window overflow** (model limit)

This notebook covers three approaches, from most to least invasive:

| Strategy | Modifies state? | What it does |
|----------|----------------|-------------|
| **RemoveMessage** | Yes | Permanently deletes messages from state |
| **Filtering** | No | Passes only a subset to the LLM (state untouched) |
| **Trimming** | No | Token-based filtering with `trim_messages` |

---

## 1. Setup: A Simple Chat Graph

```python
from langchain_mistralai import ChatMistralAI
from langgraph.graph import MessagesState, StateGraph, START, END

llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)

def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()
```

A minimal graph: messages in → LLM → response appended to messages.

---

## 2. Strategy 1: RemoveMessage (State Modification)

```python
from langchain_core.messages import RemoveMessage

def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_message = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {"messages": delete_message}

def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}
```

**Line-by-line:**
- `state['messages'][:-2]` — all messages except the last 2
- `RemoveMessage(id=m.id)` — creates a removal instruction for each old message
- The `add_messages` reducer processes these and **permanently deletes** them from state
- After this node runs, the state only contains the 2 most recent messages

### The Graph

```python
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()
```

**Flow:** `START → filter (delete old messages) → chat_model → END`

### Testing

```python
messages = [AIMessage("Hi.", name="Bot", id="1"),
            HumanMessage("Hi.", name="Abood", id="2"),
            AIMessage("So you were researching ocean mammals?", name="Bot", id="3"),
            HumanMessage("Yes, what others should I learn about?", name="Abood", id="4")]

output = graph.invoke({'messages': messages})
```

The `filter` node removes messages `id="1"` and `id="2"`. The LLM only sees messages 3 and 4.

**Trade-off:** Messages are gone forever. If the user asks "what did I say first?", the context is lost.

---

## 3. Strategy 2: Filtering (No State Modification)

```python
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}
```

**The key change:** `state["messages"][-1:]`

- We only pass the **last message** to the LLM
- But we return the response wrapped in a list → `add_messages` **appends** it to the full state
- The state keeps **all** messages — we just filter what the LLM sees

**Trade-off:** The LLM has no memory of earlier messages (it only sees the latest one), but the full history is preserved in state for other purposes.

### Testing

```python
messages.append(HumanMessage("Tell me more about Narwhals!", name="Abood"))
output = graph.invoke({'messages': messages})
```

The state output contains ALL messages (including the early ones), but the LLM only received `"Tell me more about Narwhals!"` — verified by checking the LangSmith trace.

---

## 4. Strategy 3: Trimming (Token-Based Filtering)

```python
from langchain_core.messages import trim_messages
import tiktoken

def count_tokens(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(m.content)) for m in messages)

def chat_model_node(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=count_tokens,
        allow_partial=False,
    )
    return {"messages": [llm.invoke(messages)]}
```

**Line-by-line:**
- `trim_messages(...)` — built-in LangChain utility for token-aware trimming
- `max_tokens=100` — keep only the last messages that fit within 100 tokens
- `strategy="last"` — keep the most recent messages (discard oldest first)
- `token_counter=count_tokens` — custom function using `tiktoken` to count tokens
- `allow_partial=False` — don't split a message mid-way; either include it fully or not

### How Token Counting Works

```python
def count_tokens(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(m.content)) for m in messages)
```

- `tiktoken` is OpenAI's tokenizer library (works for estimating tokens for any model)
- `cl100k_base` is the encoding used by GPT-4 / modern models
- We sum the token count of each message's content

### Testing

```python
trim_messages(messages, max_tokens=100, strategy="last",
              token_counter=count_tokens, allow_partial=False)
# -> [HumanMessage("Tell me where Orcas live!")]
```

Only the last message fits within 100 tokens, so everything else is trimmed away. The LLM responds to just that one message, but the **full state** is preserved.

---

## Comparison

| Strategy | State modified? | LLM sees | Best for |
|----------|----------------|----------|----------|
| **RemoveMessage** | Yes (messages deleted) | Remaining messages | Hard cleanup of old conversations |
| **Filtering** (`[-1:]`) | No | Last N messages | Simple, when you want minimal context |
| **Trimming** (`trim_messages`) | No | Last N tokens | Token budgets, production systems |

---

## Key Takeaways

1. **RemoveMessage** permanently deletes messages from state via the `add_messages` reducer
2. **Filtering** (`state["messages"][-1:]`) passes a subset to the LLM while keeping state intact
3. **trim_messages** is the most sophisticated — it trims based on **token count**, not message count
4. All three approaches prevent context window overflow and reduce cost
5. Filtering and trimming preserve full state — the LLM just doesn't see old messages
6. These strategies are building blocks for the chatbot with summarization (next notebook)
