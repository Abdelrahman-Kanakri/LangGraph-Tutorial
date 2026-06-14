# Walkthrough: Assistants

> **Notebook:** `assistant.ipynb`
> **Goal:** Use the LangGraph Assistants API to create multiple versioned configurations of the same deployed graph — one for personal tasks, one for work tasks.

---

## The Big Picture

An **assistant** is a named, versioned configuration for a deployed graph. Instead of deploying a separate graph for each use case, you deploy one graph and create multiple assistants from it, each with different `configurable` values.

Think of it this way: the graph is the engine; an assistant is a saved set of settings for that engine.

---

## 1. How the Graph Supports Assistants

The `task_maistro` graph is already set up for this. It has a `configuration.py` that defines the configurable fields:

```python
@dataclass(kw_only=True)
class Configuration:
    user_id: str = "default-user"
    todo_category: str = "general"
    task_maistro_role: str = "You are a helpful task management assistant..."
```

Inside the graph nodes, these fields are read from the runtime config:

```python
configurable = configuration.Configuration.from_runnable_config(config)
user_id = configurable.user_id
todo_category = configurable.todo_category
task_maistro_role = configurable.task_maistro_role
```

This means you can change the bot's persona, the storage namespace, and the user identity — all without touching the graph logic.

---

## 2. Creating an Assistant

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")

personal_assistant = await client.assistants.create(
    "task_maistro",                                        # graph name from langgraph.json
    config={"configurable": {"todo_category": "personal"}}
)
```

This creates a record in Postgres with a unique `assistant_id` and `version: 1`. The graph is not re-deployed — the same running container serves this assistant with the provided config applied at call time.

---

## 3. Updating an Assistant (Versioning)

```python
task_maistro_role = """You are a friendly and organized personal task assistant..."""

personal_assistant = await client.assistants.update(
    personal_assistant["assistant_id"],
    config={"configurable": {
        "todo_category": "personal",
        "user_id": "abood",
        "task_maistro_role": task_maistro_role,
    }}
)
```

Each `update` creates a **new version** of the assistant (version 2, 3, etc.) without deleting the old one. This gives you an audit trail of configuration changes.

---

## 4. Creating the Work Assistant

```python
task_maistro_role = """You are a focused and efficient work task assistant..."""

work_assistant = await client.assistants.create(
    "task_maistro",
    config={"configurable": {
        "todo_category": "work",
        "user_id": "Abood",
        "task_maistro_role": task_maistro_role,
    }}
)
```

Now you have two assistants from the same graph:

| Assistant | `todo_category` | Storage namespace | Persona |
|-----------|-----------------|-------------------|---------|
| Personal | `"personal"` | `("todo", "personal", "abood")` | Friendly, supportive |
| Work | `"work"` | `("todo", "work", "Abood")` | Focused, deadline-aware |

The `todo_category` value flows into the namespace used when saving to the memory store, so todos for each assistant are stored separately.

---

## 5. Using an Assistant

Pass the `assistant_id` instead of the graph name when creating a run. The server applies the assistant's config automatically.

```python
user_input = "Create a ToDo to re-film Module 6, lesson 5 by end of day today."
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"],
    work_assistant_id,          # assistant_id, not graph name
    input={"messages": [HumanMessage(content=user_input)]},
    stream_mode="values",
):
    if chunk.event == "values":
        convert_to_messages(chunk.data["messages"])[-1].pretty_print()
```

The work assistant will use its custom role prompt — for example, it proactively suggests deadlines based on task type.

---

## 6. Searching and Deleting Assistants

```python
# List all assistants
assistants = await client.assistants.search()
for a in assistants:
    print(a["assistant_id"], a["version"], a["config"])

# Delete one
await client.assistants.delete(assistant_id)
```

Assistants are stored in Postgres in your deployment. `search()` returns them all. A deleted assistant is gone permanently; its versioned history is also deleted.

---

## The Default Assistant

Every deployed graph has one default assistant created automatically — `assistant_id` with no config overrides. This is what gets used when you pass the graph name (`"task_maistro"`) directly instead of an `assistant_id`.

---

## What to Understand Here

1. **One graph, many personas** — assistants let you use the same deployed graph for different users, categories, or roles without separate deployments or code branches.
2. **`todo_category` drives storage isolation** — because it flows into the memory store namespace, personal and work todos are completely separate even though both assistants use the same graph.
3. **Versioning is automatic** — each `update` call creates a new version. Old versions are preserved, which is useful for auditing configuration changes.
4. **Assistants are not threads** — an assistant is a saved config; a thread is a saved conversation. You can run many threads with the same assistant.
