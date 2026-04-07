# Walkthrough: Deployment

> **Notebook:** `deployment.ipynb`
> **Goal:** Deploy the agent locally with LangGraph Studio and to the cloud with LangGraph Cloud, then interact via the SDK.

---

## The Big Picture

So far we've built and run graphs inside Jupyter notebooks. Now we learn how to **deploy** them as a service — both locally and in the cloud — and interact with them programmatically using the SDK.

---

## 1. Key Concepts

| Component | What it is |
|-----------|-----------|
| **LangGraph** | The Python/JS library for building agent workflows (what we've been using) |
| **LangGraph API** | A server that bundles your graph code with a task queue and persistence |
| **LangSmith Deployment** | Hosted cloud service that runs the LangGraph API for you |
| **LangSmith Studio** | Visual IDE for testing and debugging graphs (runs locally or in the cloud) |
| **LangGraph SDK** | Python client library for interacting with deployed graphs |

---

## 2. Local Deployment with Studio

### The Studio Directory

Each module has a `studio/` folder containing:
- `agent.py` — the graph code (same logic as notebooks, packaged as a module)
- `langgraph.json` — config file telling Studio which graphs to load
- `requirements.txt` — dependencies
- `.env` — API keys

### Starting the Local Server

```bash
cd module-1/studio
langgraph dev
```

**Output:**
```
- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://127.0.0.1:2024/docs
```

This starts:
1. A **local LangGraph API server** on port 2024
2. A **Studio UI** in your browser for visual testing

---

## 3. Interacting via the SDK

### Connect to the Server

```python
from langgraph_sdk import get_client

URL = "http://127.0.0.1:2024"
client = get_client(url=URL)
```

- `get_client(url=...)` creates a client that talks to the LangGraph API server
- Works the same whether the server is local or cloud-hosted (just change the URL)

### Discover Assistants

```python
assistants = await client.assistants.search()
```

- Lists all graphs available on the server
- Each assistant has an `assistant_id`, `graph_id`, and `name`

```python
assistants[-3]
# {'assistant_id': 'fe096781-...', 'graph_id': 'agent', 'name': 'agent', ...}
```

### Create a Thread

```python
thread = await client.threads.create()
```

- Creates a new conversation thread (equivalent to `{"configurable": {"thread_id": "..."}}`)
- The server manages thread persistence for you

### Stream a Run

```python
from langchain_core.messages import HumanMessage

input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=input,
    stream_mode="values",
):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data["messages"][-1])
```

**Line-by-line:**
- `client.runs.stream(...)` — starts a run on the server and streams results back
- `thread["thread_id"]` — which conversation thread to use
- `"agent"` — the `graph_id` to run (matches what's in `langgraph.json`)
- `input=input` — the messages to send
- `stream_mode="values"` — stream the **full state** after each step (not just deltas)
- `chunk.data["messages"][-1]` — the latest message in the state at each step

**Output shows each step:**
```
Human: Multiply 3 by 2.
AI: [Tool Call: multiply(a=3, b=2)]
Tool: 6
AI: 3 multiplied by 2 equals 6.
```

---

## 4. Cloud Deployment

### Push to GitHub

```bash
git remote add origin https://github.com/your-username/your-repo-name.git
git push -u origin main
```

### Deploy via LangSmith

1. Go to [LangSmith](https://smith.langchain.com/)
2. Click **Deployments** → **+ New Deployment**
3. Select your GitHub repository
4. Set `LangGraph API config file` to `module-1/studio/langgraph.json`
5. Add your API keys

### Use the Cloud URL

```python
URL = "https://your-deployment-id.default.us.langgraph.app"
client = get_client(url=URL)

# Everything else is identical to local!
thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"], "agent",
    input={"messages": [HumanMessage(content="Multiply 3 by 2.")]},
    stream_mode="values",
):
    print(chunk.data["messages"][-1])
```

**The beauty:** The SDK code is **identical** for local and cloud. Just swap the URL.

---

## Local vs. Cloud

| Feature | Local (`langgraph dev`) | Cloud (LangSmith) |
|---------|------------------------|-------------------|
| URL | `http://127.0.0.1:2024` | `https://your-id.langgraph.app` |
| Setup | One command | GitHub + LangSmith dashboard |
| Persistence | In-memory (lost on restart) | Durable (database-backed) |
| Scaling | Single process | Managed infrastructure |
| Cost | Free | Pay per usage |
| Use case | Development & testing | Production |

---

## Key Takeaways

1. **`langgraph dev`** starts a local server + Studio UI for testing
2. **LangGraph SDK** (`get_client`) provides a uniform interface for local and cloud
3. **Threads** manage conversation persistence server-side
4. **Streaming** lets you see each step as it happens (not just the final result)
5. **Cloud deployment** = push to GitHub + connect via LangSmith dashboard
6. **Same SDK code** works for both local and cloud — just change the URL
