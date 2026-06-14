# Module 6 — Deployment

Taking the `task_mAIstro` agent from a local notebook to a **self-hosted production deployment** on Docker, then interacting with it via the LangGraph SDK and the Assistants API.

---

## 📌 The deployment stack

```
┌─────────────────────────────────────────────┐
│            LangGraph Platform Stack         │
│                                             │
│  langgraph-api    ← your graph + server     │
│  langgraph-postgres ← checkpoints + store  │
│  langgraph-redis    ← streaming updates    │
└─────────────────────────────────────────────┘
         ↑  managed by `langgraph up`
```

**Four required files in `module-6/deployment/`:**

| File | Role |
|------|------|
| `langgraph.json` | Control file — graph name, Python version, env path, dependencies |
| `task_maistro.py` | Your graph — exports the `graph` variable |
| `requirements.txt` | Packages baked into the Docker image at build time |
| `.env` | API keys — referenced from `langgraph.json`, never committed |

> **Before running the notebooks:** follow the step-by-step [DEPLOYMENT_GUIDE.md](../module-6/DEPLOYMENT_GUIDE.md) to build the image and start the stack.

---

## 01 · creating-deployment

**Notebook:** [creating_deployment.ipynb](../module-6/01-createing_deployment/creating_deployment.ipynb) · **Walkthrough:** [walkthrough.md](../module-6/01-createing_deployment/walkthrough.md)

**What it teaches**
How to package a LangGraph app into a Docker image and bring up the full 3-container stack.

**Key commands**

```bash
cd module-6/deployment
langgraph build -t my-image   # build — reads langgraph.json + requirements.txt
langgraph up                  # run — starts api + postgres + redis
```

**`langgraph.json` anatomy**

```json
{
  "graphs":          { "task_maistro": "./task_maistro.py:graph" },
  "env":             "../../.env",
  "python_version":  "3.12",
  "dependencies":    ["."]
}
```

- `graphs` — maps deployment name → `file:variable`
- `env` — path to your `.env` (no `--env-file` flag exists; configure it here)
- `python_version` — baked into the Docker image

**`langgraph up` vs `docker run`**
`langgraph up` generates and runs a Compose file that starts all three containers and wires them together. `docker run my-image` starts only the API container with no database or Redis to connect to — it crashes immediately.

**Verify it's running**
- API docs: `http://localhost:8123/docs`
- Health: `http://localhost:8123/ok`

**When to use**
Every time you want to move a graph from "runs in a notebook" to "accessible via HTTP with real persistence and streaming".

---

## 02 · connecting

**Notebook:** [connecting.ipynb](../module-6/02-connecting/connecting.ipynb) · **Walkthrough:** [walkthrough.md](../module-6/02-connecting/walkthrough.md)

**What it teaches**
Two ways to connect to the deployment (SDK vs. `RemoteGraph`), the three run modes, full thread control including forking, and direct store access via the SDK.

**Key APIs**

- `from langgraph_sdk import get_client; client = get_client(url="http://localhost:8123")` — SDK client
- `from langgraph.pregel.remote import RemoteGraph; g = RemoteGraph("task_maistro", url=url)` — drop-in local graph replacement

**Run modes**

| Mode | Code | Use when |
|------|------|----------|
| Fire-and-forget | `client.runs.create(...)` | Start and don't wait |
| Blocking | `+ await client.runs.join(thread_id, run_id)` | Need the result before continuing |
| Streaming | `client.runs.stream(..., stream_mode="messages-tuple")` | Chat UI — tokens as they arrive |

**Threads**

```python
# Check state
state = await client.threads.get_state(thread_id)

# Fork (independent copy of an existing thread)
copied = await client.threads.copy(thread_id)

# Human-in-the-loop: edit a past checkpoint, then resume
forked_config = await client.threads.update_state(
    thread_id, new_input, checkpoint_id=old_checkpoint_id
)
await client.runs.stream(thread_id, graph_name, input=None,
                         checkpoint_id=forked_config["checkpoint_id"], ...)
```

**Key:** supply the original message's `id` when updating state to *overwrite* rather than append — otherwise the `add_messages` reducer appends and you end up with duplicate messages.

**Store API**

```python
await client.store.search_items(("todo", "general", "test"), limit=5)
await client.store.put_item(namespace, key=str(uuid4()), value={...})
await client.store.delete_item(namespace, key=key)
```

**When to use**
Any code that talks to the deployed graph — chatbots, CLI tools, integration tests, dashboards.

---

## 03 · double-texting

**Notebook:** [double_texting.ipynb](../module-6/03-double_texting/double_texting.ipynb) · **Walkthrough:** [walkthrough.md](../module-6/03-double_texting/walkthrough.md)

**What it teaches**
How to handle a user sending a second message before the first run finishes — a common real-world problem for chat applications.

**Key API**

```python
await client.runs.create(
    thread_id, graph_name, input={...},
    multitask_strategy="reject" | "enqueue" | "interrupt" | "rollback"
)
```

**Strategy comparison**

| Strategy | Run 1 fate | Run 2 starts | Thread state after |
|----------|-----------|--------------|-------------------|
| `reject` | Continues normally | Error — never starts | Only run 1 |
| `enqueue` | Runs to completion | After run 1 finishes | Both, in order |
| `interrupt` | Stopped at next checkpoint | Immediately | Partial run 1 + full run 2 |
| `rollback` | Stopped **and deleted** | Immediately, fresh | Only run 2 |

**Default:** `enqueue` — if you don't set `multitask_strategy`, new runs are automatically queued.

**When to use**
- `reject` — correctness-first; show "please wait" to the user
- `enqueue` — all messages matter; order preserved (default for most apps)
- `interrupt` — "never mind, do this instead" but keep the partial work
- `rollback` — second message completely replaces the first, no trace of old intent

---

## 04 · assistants

**Notebook:** [assistant.ipynb](../module-6/04-assistant/assistant.ipynb) · **Walkthrough:** [walkthrough.md](../module-6/04-assistant/walkthrough.md)

**What it teaches**
One deployed graph → many named, versioned configurations. Create separate personal and work assistants from the same `task_maistro` graph by injecting different `configurable` values.

**Key APIs**

```python
# Create
asst = await client.assistants.create("task_maistro",
    config={"configurable": {"todo_category": "personal", "user_id": "abood"}})

# Update (creates a new version — old versions are preserved)
asst = await client.assistants.update(asst["assistant_id"],
    config={"configurable": {"task_maistro_role": "..."}})

# Use — pass assistant_id instead of graph name
await client.runs.stream(thread_id, asst["assistant_id"], input={...})

# List / delete
assistants = await client.assistants.search()
await client.assistants.delete(assistant_id)
```

**Why `todo_category` matters**
It flows into the store namespace: `("todo", todo_category, user_id)`. Personal and work todos are stored in completely separate namespaces even though they use the same graph — switching categories switches storage buckets.

**Versioning**
Each `update` call increments the version number and keeps older versions. Useful for auditing configuration changes in production.

**When to use**
Multi-tenant apps, per-user persona customization, A/B testing prompts, or any case where one graph needs to behave differently for different users or contexts — without redeploying.

---

## Module 6 quick decision tree

```
Deploying a graph?
   └── langgraph build -t <image> && langgraph up

Connecting from client code?
   ├── Pure Python / async                     → langgraph_sdk client
   └── Existing code that calls a local graph  → RemoteGraph (drop-in)

Choosing a run mode?
   ├── Start and move on                        → fire-and-forget (runs.create)
   ├── Need the result synchronously            → blocking (runs.create + runs.join)
   └── Streaming chat UI                        → stream (stream_mode="messages-tuple")

Handling concurrent messages?
   ├── Never allow overlap                      → reject
   ├── Queue and process all                    → enqueue (default)
   ├── Replace but keep partial work            → interrupt
   └── Replace entirely, erase old             → rollback

Multiple personas from one graph?
   └── Assistants API (client.assistants.create with configurable values)
```
