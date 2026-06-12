# Walkthrough: Chatbot with Collection Schema

> **Notebook:** `module-05/03-memoryschema-collection/memoryschema_collection.ipynb`
> **Goal:** Replace the single-document profile approach with a *collection* — many independent memory entries, each with its own UUID key, that can be inserted or updated individually.

---

## The Big Picture

The previous notebook (02-memory-schema-profile) stored everything about a user as **one document**, overwritten on every turn. This works for structured facts (name, location, job) but loses flexibility — you can't add a new fact without touching the whole profile.

A **collection** stores each fact as a **separate entry**:

```
Profile approach (one doc, overwritten):
  namespace: ("memory", user_id)
  key: "user_memory"   ← always the same
  value: { name, location, job, ... }  ← entire doc replaced each time

Collection approach (many docs, each with its own key):
  namespace: ("memories", user_id)
  key: "uuid-1"  →  { content: "User's name is Abood." }
  key: "uuid-2"  →  { content: "User likes biking around Irbid, Jordan." }
  key: "uuid-3"  →  { content: "User enjoys going to bakeries." }
```

Individual entries can be inserted (new UUID) or patched (same UUID) without touching any other entry.

---

## 1. Defining the Collection Schema

```python
from pydantic import BaseModel, Field

class Memory(BaseModel):
    """Memory schema represents a single memory entry with content."""
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    """The MemoryCollection schema represents a collection of memory entries."""
    memories: list[Memory] = Field(description="A list of memories about the user.")
```

Two models, two roles:
- **`Memory`** — the unit of storage. One entry, one fact.
- **`MemoryCollection`** — used only with `with_structured_output` to extract an initial batch of memories from a message. Not used in the Trustcall path.

> **Docstrings are required.** Groq rejects tool schemas where `description` is `null`. The class docstring becomes the tool's `description` field in the API request. See the notes in `module-05/04-memory-agent/walkthrough.md` for details.

---

## 2. Initial Extraction with `with_structured_output`

For a one-shot extraction (no existing memories yet), the simplest approach is `with_structured_output`:

```python
model_with_structure = model.with_structured_output(MemoryCollection)

memory_collections = model_with_structure.invoke(
    [HumanMessage(content="My name is Abood. I like to bike.")]
)
# → MemoryCollection(memories=[Memory(content='User Abood expressed interest in biking.')])
```

To store each memory with its own UUID:

```python
for memory in memory_collections.memories:
    key = str(uuid.uuid4())
    value = memory.model_dump()          # {"content": "User Abood expressed interest in biking."}
    in_memory_store.put(namespace, key, value)
```

**The limitation:** `with_structured_output` always regenerates the entire output. If you call it again with new information, it produces a brand-new list with no connection to the entries already in the store — you'd have to diff manually to avoid duplicates or overwrites. Trustcall solves this.

---

## 3. Updating a Collection with Trustcall

Trustcall needs `enable_inserts=True` to work with collections. Without this flag, it can only update existing entries via `PatchDoc` — insertions are disabled.

```python
from trustcall import create_extractor

trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True      # ← required for collections
)
```

### First extraction (no existing memories)

```python
conversation = [
    HumanMessage(content="Hi, I'm Abood."),
    AIMessage(content="Nice to meet you, Abood."),
    HumanMessage(content="This morning I had a nice bike ride in Irbid."),
]

result = trustcall_extractor.invoke({
    "messages": [SystemMessage(content="Extract memories from the following conversation:")] + conversation
})
# result["responses"] → [Memory(content='User had a bike ride in Irbid this morning')]
```

### Second extraction (pass existing memories)

For Trustcall to update existing entries (instead of always inserting), you must tell it what is already in the store. You do this via the `"existing"` key — a list of `(id, tool_name, value_dict)` tuples:

```python
tool_name = "Memory"
existing_memories = [
    (str(i), tool_name, memory.model_dump())
    for i, memory in enumerate(result["responses"])
]
# → [('0', 'Memory', {'content': 'User had a bike ride in Irbid this morning'})]

result = trustcall_extractor.invoke({
    "messages": updated_conversation,
    "existing": existing_memories
})
```

### Reading the result — `json_doc_id` tells you insert vs update

```python
for m in result["response_metadata"]:
    print(m)
# {'id': 'pwz9mjyek'}                  ← no json_doc_id → new entry (INSERT)
# {'id': 'h5eawmaak', 'json_doc_id': '0'}  ← has json_doc_id → patched entry '0' (UPDATE)
```

- **No `json_doc_id`** → Trustcall created a new memory.
- **Has `json_doc_id`** → Trustcall patched an existing memory with that ID.

This is how you know which UUID to use when writing back to the store:

```python
for r, rmeta in zip(result["responses"], result["response_metadata"]):
    store.put(
        namespace,
        rmeta.get("json_doc_id", str(uuid.uuid4())),  # reuse old UUID if update, new UUID if insert
        r.model_dump(mode="json")
    )
```

---

## 4. Full Chatbot with Collection Memory

The graph structure is identical to the profile chatbot:

```
START → call_model → write_memory → END
```

But the store access pattern is different in both nodes.

### `call_model` — load all entries via `search`

```python
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # search returns ALL entries in the namespace (one per memory)
    memories = store.search(namespace)

    # Format as a bulleted list for the system prompt
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}
```

With the profile approach you used `store.get(namespace, "user_memory")` (one fixed key). With a collection you use `store.search(namespace)` because there are many keys and you don't know them in advance.

### `write_memory` — format existing entries for Trustcall

```python
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # Retrieve all existing entries
    existing_items = store.search(namespace)

    # Build the (key, tool_name, value) tuples Trustcall expects
    existing_memories = (
        [(item.key, "Memory", item.value) for item in existing_items]
        if existing_items else None
    )

    # Merge instruction into the conversation
    updated_messages = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]
    ))

    result = trustcall_extractor.invoke({
        "messages": updated_messages,
        "existing": existing_memories
    })

    # Write back — reuse existing UUID for updates, generate new UUID for inserts
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json")
        )
```

**Critical difference from profile:** The profile chatbot passed `existing_item.key` as a static string (`"user_memory"`). Here, `existing_item.key` is the actual UUID — Trustcall's `json_doc_id` in the response will match those UUIDs so updates land on the right entry.

---

## 5. What the Store Looks Like After Conversations

After three messages:

```python
for m in across_thread_memory.search(("memories", "1")):
    print(m.dict())

# {'key': 'uuid-1', 'value': {'content': 'User name is Abood and he likes biking.'}, ...}
# {'key': 'uuid-2', 'value': {'content': 'User name is Abood, he likes biking around Irbid, Jordan.'}, ...}
# {'key': 'uuid-3', 'value': {'content': 'User also enjoys going to bakeries.'}, ...}
```

Each entry lives independently. Trustcall added entries 2 and 3 without touching entry 1.

---

## 6. Cross-Thread Memory in Action

```python
# New thread, same user_id
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]
# → "Abood, since you like biking around Irbid, Jordan, and visiting bakeries..."
```

Thread 2 has no chat history — but the store still has all three memory entries from thread 1. The agent addresses the user by name and references their location and interests without being told again.

---

## Profile vs Collection — When to Use Which

| | Profile | Collection |
|---|---|---|
| **Schema** | One structured Pydantic model | Many entries, each a simple `content: str` |
| **Store key** | Fixed (`"user_memory"`) | UUID per entry |
| **Update** | Overwrite the whole doc | Patch just the changed entry |
| **Trustcall flag** | `enable_inserts=False` (default) | `enable_inserts=True` |
| **Best for** | Structured facts (name, job, location) | Open-ended, growing list of observations |
| **`store.get` vs `store.search`** | `get` (known key) | `search` (unknown keys) |

---

## Key Takeaways

1. **Collection = one UUID key per memory entry.** Unlike the profile (one doc, always same key), each fact lives independently.
2. **`enable_inserts=True` is required.** Without it, Trustcall can only patch — it cannot create new entries.
3. **Pass `"existing"` to Trustcall.** Without it, Trustcall has no way to match new information to old entries and will always insert.
4. **`json_doc_id` in `response_metadata` signals update vs insert.** Present → reuse the old UUID. Absent → generate a new UUID.
5. **`store.search` instead of `store.get`** — you don't know the keys in a collection, so scan the namespace.
6. **`merge_message_runs`** — merges the Trustcall instruction with the conversation history into one clean message list before invoking the extractor.
