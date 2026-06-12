# Walkthrough: Chatbot with Long-Term Memory (Memory Store)

> **Notebook:** `academy_notebooks/module-5/memory_store.ipynb`
> **Goal:** Build a chatbot that remembers facts about a user *across separate chat sessions* using the LangGraph Memory Store.

---

## The Big Picture

Every chatbot we built before only remembered what happened *within the current session*. Once a new thread started, the slate was wiped clean.

This notebook introduces a second, separate layer of memory that persists *across threads*:

```
┌─────────────────────────────────────────────────────────┐
│  Short-term (within-thread)   →   MemorySaver            │
│  Persists: chat history in a single thread              │
├─────────────────────────────────────────────────────────┤
│  Long-term (across-thread)    →   InMemoryStore          │
│  Persists: user facts across ALL threads for that user  │
└─────────────────────────────────────────────────────────┘
```

The two layers are independent. A new thread gets a blank chat history but a full user memory.

---

## 1. The LangGraph Memory Store

The `InMemoryStore` is a key-value store organized by **namespace**. Think of it like a filesystem:

```
namespace  →  a tuple that acts like a folder path   ("memory", "user-123")
key        →  a filename within that folder          "user_memory"
value      →  the file contents                     {"memory": "User likes pizza"}
```

### Creating the store

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

### The three operations

**`put` — save a value:**
```python
namespace = ("user-id", "memories")
key = str(uuid.uuid4())          # unique ID for this memory entry
value = {"food_preference": "I like pizza"}

in_memory_store.put(namespace, key, value)
```

**`search` — retrieve all entries in a namespace:**
```python
memories = in_memory_store.search(namespace)  # returns a list

# Each item has .key, .value, .namespace, .created_at, .updated_at
print(memories[0].key)    # the UUID
print(memories[0].value)  # {"food_preference": "I like pizza"}
```

**`get` — retrieve one specific entry by key:**
```python
memory = in_memory_store.get(namespace, key)
print(memory.value)  # {"food_preference": "I like pizza"}
```

> `search` is for scanning a namespace (you don't know the key). `get` is for fetching when you know the exact key.

---

## 2. The Two-Node Chatbot

The graph has two nodes that run sequentially on every message:

```
START → call_model → write_memory → END
```

### `call_model` — respond using memory

```python
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    # Load existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Inject memory into the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

    # Reply using both memory and the current chat history
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": response}
```

The key idea: `user_id` comes from the config, not the message. The store is namespaced per user, so two different users never share memory even on the same graph.

### `write_memory` — reflect and save

```python
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)

    # Load whatever was saved before
    existing_memory = store.get(namespace, "user_memory")
    existing_memory_content = existing_memory.value.get('memory') if existing_memory else "No existing memory found."

    # Ask the LLM to merge old memory with the new conversation
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'])

    # Overwrite the single "user_memory" key with the updated content
    store.put(namespace, "user_memory", {"memory": new_memory.content})
```

**Why overwrite instead of append?** This notebook uses a **profile** pattern — one summary document per user, updated in place. The LLM merges old and new information each time. This is different from a **collection** pattern where each fact gets its own UUID key. See module-05/03-memory-schema-collection for that approach.

---

## 3. Dual Persistence — Wiring It Together

```python
# Long-term: survives new threads
across_thread_memory = InMemoryStore()

# Short-term: scoped to a thread
within_thread_memory = MemorySaver()

graph = builder.compile(
    checkpointer=within_thread_memory,
    store=across_thread_memory
)
```

LangGraph automatically injects `store` into any node that declares it in its signature. You don't pass it manually — just add `store: BaseStore` as a parameter.

---

## 4. Running the Graph

Each invocation needs two IDs in `config`:

```python
config = {"configurable": {
    "thread_id": "1",    # which conversation thread (short-term memory scope)
    "user_id":   "1"     # which user (long-term memory scope)
}}
```

### Thread 1 — building up memory

```python
# Message 1
input_messages = [HumanMessage(content="Hi, my name is Lance")]
# → "Hello, Lance! How can I assist you today?"
# → write_memory saves: "User's name is Lance."

# Message 2
input_messages = [HumanMessage(content="I like to bike around San Francisco")]
# → "That sounds great, Lance! Do you have a favorite route?"
# → write_memory updates: "User's name is Lance. Likes to bike around San Francisco."
```

### Thread 2 — same user, fresh chat history

```python
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
# New thread = blank chat history (MemorySaver knows nothing)
# But store still has: "User's name is Lance. Likes to bike around SF."

input_messages = [HumanMessage(content="Hi! Where would you recommend that I go biking?")]
# → "Hi Lance! Since you enjoy biking around San Francisco, here are some routes..."
```

The agent used the user's name and location without being told again — that came from the long-term store, not the chat history.

---

## 5. What the Memory Looks Like in the Store

After two messages in thread 1:

```python
namespace = ("memory", "1")
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
# {
#   'value': {'memory': "**Updated User Information:**\n- User's name is Lance.\n- Likes to bike around San Francisco."},
#   'key': 'user_memory',
#   'namespace': ['memory', '1'],
#   'created_at': '2024-11-05T00:12:17.383918+00:00',
#   'updated_at': '2024-11-05T00:12:25.469528+00:00'
# }
```

Note `created_at` vs `updated_at` — these are different because the memory was first created on message 1 and then overwritten on message 2.

---

## Key Takeaways

1. **Two memory layers**: `MemorySaver` (within-thread, chat history) + `InMemoryStore` (across-thread, user facts). They are independent.
2. **Store = namespace + key + value**: namespace scopes by user, key identifies the document, value is a plain dict.
3. **`search` vs `get`**: use `search` when scanning a namespace, `get` when you know the exact key.
4. **Profile pattern**: one summary document per user, overwritten on each turn. Simpler than a collection but loses the history of individual edits.
5. **`user_id` in config**: the store is partitioned by user via the namespace tuple — always pull `user_id` from `config["configurable"]`, never from the message.
6. **`store: BaseStore` in node signature**: LangGraph auto-injects the compiled store. No manual passing needed.
7. **Memory is written "in the hot path"**: `write_memory` runs after every single message. This means memory is always fresh but adds one extra LLM call per turn.
