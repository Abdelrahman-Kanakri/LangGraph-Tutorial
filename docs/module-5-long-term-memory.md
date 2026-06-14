# Module 5 — Long-term Memory

Cross-session memory using the LangGraph **memory store** (`BaseStore`). Where Module 2 covered in-thread persistence (checkpoints), this module covers *across*-thread persistence: facts that should survive conversation restarts and be shared across separate threads for the same user.

---

## 📌 Memory store vs. checkpointer — the core distinction

| | Checkpointer | Memory Store |
|---|---|---|
| **What it stores** | Full graph state at each step | Arbitrary key-value items you explicitly `put` |
| **Scope** | One thread | Any namespace, across all threads |
| **Retrieval** | Automatic (bound to `thread_id`) | Manual — you call `store.search(namespace)` |
| **Backend (dev)** | `MemorySaver` / `SqliteSaver` | `InMemoryStore` |
| **Backend (prod)** | Postgres checkpointer | Postgres store (in LangGraph Platform) |

Use a checkpointer for conversation history. Use a store for facts you want to remember *about* a user across conversations.

---

## 01 · memory-store

**Notebook:** [memory_store.ipynb](../module-5/01-memory-store/memory_store.ipynb) · **Walkthrough:** [walkthrough.md](../module-5/01-memory-store/walkthrough.md)

**What it teaches**
The raw primitives of the store: `put`, `get`, `search`, and `delete` on namespaced items. How to inject the store into a node and how to pass it to the graph at compile time.

**Key APIs**

- `from langgraph.store.memory import InMemoryStore` — in-process store for development
- `store.put(namespace, key, value)` — write an item; `namespace` is a tuple of strings, `key` is a string, `value` is any JSON-serialisable dict
- `store.search(namespace)` — returns all `Item` objects in a namespace; each has `.key` and `.value`
- `store.get(namespace, key)` — fetch one item by key
- `store.delete(namespace, key)` — remove an item
- Node signature: `def node(state, config, *, store: BaseStore)` — LangGraph injects the store when the node declares it
- `builder.compile(store=store)` — bind the store to the graph at compile time

**Namespace pattern**
`(memory_type, user_id)` — e.g., `("memories", "alice")`. The tuple acts like a folder path; `search` returns everything under it.

**When to use**
Any time you want the graph to remember something about a user across separate conversations — preferences, facts, prior decisions.

---

## 02 · memory-schema-profile

**Notebook:** [memoryschema_profile.ipynb](../module-5/02-memory-schema-profile/memoryschema_profile.ipynb) · **Walkthrough:** [walkthrough.md](../module-5/02-memory-schema-profile/walkthrough.md)

**What it teaches**
Structured memory using [Trustcall](https://github.com/hinthornw/trustcall): an LLM-powered extractor that reads chat history and writes (or patches) a typed Pydantic schema into the store. Teaches the **single-document profile** pattern — one record per user, updated in-place.

**Key APIs**

- `from trustcall import create_extractor` — build an extractor bound to a Pydantic schema and a model
- `create_extractor(model, tools=[MySchema], tool_choice="MySchema")` — the model will always call the schema tool
- `extractor.invoke({"messages": ..., "existing": existing_memories})` — pass prior records so Trustcall can patch rather than overwrite
- `existing_memories = [(key, schema_name, value_dict), ...]` — format for existing items
- `result["responses"]` — list of populated Pydantic model instances
- `result["response_metadata"]` — includes `json_doc_id` for upsert key
- `store.put(namespace, rmeta.get("json_doc_id", str(uuid4())), r.model_dump(mode="json"))` — the upsert write

**Profile pattern**
One Pydantic model (e.g., `Profile`) with optional fields (`name`, `location`, `job`, `connections`, `interests`). Trustcall fills in or patches only the fields it can infer from the conversation — it never blanks out fields it has no signal about.

**`PatchDoc` tool**
Trustcall uses an internal `PatchDoc` tool to express partial updates to existing records. When you pass `existing`, it patches; when you don't (or the doc doesn't exist yet), it creates.

**When to use**
When you want the model to silently learn *who the user is* across conversations — name, job, preferences — with no explicit memory commands from the user.

---

## 03 · memoryschema-collection

**Notebook:** [memoryschema_collection.ipynb](../module-5/03-memoryschema-collection/memoryschema_collection.ipynb) · **Walkthrough:** [walkthrough.md](../module-5/03-memoryschema-collection/walkthrough.md)

**What it teaches**
The **collection** pattern: one store namespace holds *many* items of the same schema (e.g., a ToDo list). Each item has its own key and can be independently created, patched, or deleted by Trustcall.

**Key APIs**

- `create_extractor(..., enable_inserts=True)` — allows Trustcall to create new documents in addition to patching existing ones
- `extractor.with_listeners(on_end=spy)` — attach a callback to inspect raw tool calls (useful for logging what changed)
- `spy.called_tools` — the raw tool call log, exposing `PatchDoc` patches and new schema creations
- `store.put(namespace, key, value)` in a loop over `result["responses"]` — each response is one item in the collection

**Profile vs. Collection**

| | Profile (notebook 02) | Collection (notebook 03) |
|---|---|---|
| Items per namespace | 1 | N |
| `enable_inserts` | Not needed | Required |
| Use case | "Who is this user?" | "What tasks does this user have?" |
| Key source | Fixed (`"profile"`) | `json_doc_id` per item |

**When to use**
ToDo lists, bookmark libraries, contact lists, conversation notes — anything where the user can have many items of the same type and you want the model to manage them automatically.

---

## 04 · memory-agent

**Notebook:** [memory_agent.ipynb](../module-5/04-memory-agent/memory_agent.ipynb) · **Walkthrough:** [walkthrough.md](../module-5/04-memory-agent/walkthrough.md)

**What it teaches**
Assembles everything from this module into a full production agent: `task_mAIstro`. The agent manages three separate memory namespaces (profile, todos, instructions) and routes to the right update node via a tool call.

**Key APIs / Architecture**

```
user message
     ↓
task_mAIstro node  ──[no tool call]──→  END
     │
     └─[UpdateMemory tool call]──→  route_message()
                                          ├── "user"         →  update_profile
                                          ├── "todo"         →  update_todos
                                          └── "instructions" →  update_instructions
                                                   ↓
                                           back to task_mAIstro
```

- `class UpdateMemory(TypedDict): update_type: Literal["user", "todo", "instructions"]` — the routing tool the LLM calls
- `model.bind_tools([UpdateMemory], parallel_tool_calls=False)` — force at most one routing decision per turn
- `config_schema=configuration.Configuration` — wires the graph's `Configuration` dataclass so callers can inject `user_id`, `todo_category`, `task_maistro_role` at invocation time
- Three `create_extractor` instances — one per memory namespace, each bound to its own Pydantic schema
- `merge_message_runs` — collapses consecutive same-role messages before passing to Trustcall (reduces token usage)

**The three namespaces**

| Namespace | Schema | What it stores |
|-----------|--------|---------------|
| `("profile", todo_category, user_id)` | `Profile` | Name, location, job, connections, interests |
| `("todo", todo_category, user_id)` | `ToDo` | Task, deadline, time estimate, status, solutions |
| `("instructions", todo_category, user_id)` | Plain string | User preferences for how to manage their todo list |

**When to use**
This is the capstone pattern: a multi-memory agent that silently learns about the user, tracks structured tasks, and adapts its own update behaviour based on user feedback — all without explicit commands from the user.

---

## Module 5 quick decision tree

```
Need to persist data across threads?
   ├── One record per user (profile, settings)     → Profile pattern  (notebook 02)
   ├── Many records per user (tasks, notes)        → Collection pattern (notebook 03)
   └── Multiple namespaces in one agent            → Full memory agent (notebook 04)

What backend?
   ├── Development / local notebook               → InMemoryStore
   └── Production / deployed                      → Postgres store (via LangGraph Platform)
```
