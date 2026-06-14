# Walkthrough: Creating a LangGraph Deployment

> **Notebook:** `creating_deployment.ipynb`
> **Goal:** Package the `task_maistro` app from Module 5 into a self-hosted Docker deployment using the LangGraph CLI.

---

## The Big Picture

In previous modules, we ran graphs locally in a notebook. A "deployment" means packaging that same graph into a production-ready server that exposes an HTTP API, backed by Postgres (for checkpoints) and Redis (for streaming). The LangGraph CLI handles all of that.

---

## 1. What a Deployment Needs

LangGraph Platform requires four things to exist before you can build:

| File | Purpose |
|------|---------|
| `langgraph.json` | Control file — tells the CLI which graph to serve, which Python version, and where your keys are |
| `task_maistro.py` | Your graph — the file that exports the `graph` variable |
| `requirements.txt` | Python packages to install into the Docker image |
| `.env` | Your API keys (never committed to git) |

All four live in `module-6/deployment/`. You do not need to create them; they are already there.

---

## 2. `langgraph.json` — the Control File

```json
{
    "dockerfile_lines": [],
    "graphs": {
        "task_maistro": "./task_maistro.py:graph"
    },
    "env": "../../.env",
    "python_version": "3.12",
    "dependencies": ["."]
}
```

**Line-by-line:**
- `"graphs"` — maps a deployment name (`task_maistro`) to the Python object that is the graph (`./task_maistro.py:graph`). The part after `:` is the variable name inside the file.
- `"env": "../../.env"` — tells the server where your keys are. This version of langgraph has no `--env-file` flag; you set it here instead.
- `"python_version": "3.12"` — bakes Python 3.12 into the Docker image.
- `"dependencies": ["."]` — install everything in `requirements.txt` (the `.` refers to this folder).

---

## 3. `requirements.txt` — Image Dependencies

```
langgraph
langchain-core
langchain-community
langchain-openai
trustcall
langchain-groq
langchain-mistralai
```

These are the packages installed **inside the Docker image** during `langgraph build`. They are separate from your local venv. If you add a new import to `task_maistro.py`, add its package here and rebuild.

---

## 4. Installing the LangGraph CLI

```python
%pip install -U langgraph-cli
```

The CLI (`langgraph`) is what builds and runs the stack. It lives in your local venv (not the Docker image). **Run it from Konsole, not the VS Code terminal** — VS Code's Flatpak sandbox cannot see the host's Docker daemon.

---

## 5. Building the Docker Image

```bash
cd module-6/deployment
langgraph build -t my-image
```

**What happens:**
1. The CLI reads `langgraph.json` to find your graph, Python version, and dependencies.
2. It generates a Dockerfile and runs `docker build`.
3. The result is a Docker image named `my-image` that contains your graph code, all dependencies, and the LangGraph Server.

This step only needs to be re-run when you change `task_maistro.py` or `requirements.txt`. Changing `.env` does not require a rebuild.

---

## 6. Running the Full Stack

### Option A — `langgraph up` (recommended)

```bash
langgraph up
```

This is the simplest path. The CLI generates a Docker Compose file on the fly that starts three containers:

| Container | What it runs |
|-----------|-------------|
| `langgraph-api` | The LangGraph Server (your graph) |
| `langgraph-postgres` | Postgres — stores thread checkpoints and cross-thread memories |
| `langgraph-redis` | Redis — used for streaming run updates to clients |

The three containers are networked together automatically.

> Do NOT run `docker run my-image` by itself. Without Postgres and Redis, the API container crashes on startup.

### Option B — `docker compose up` (advanced)

If you need to customize the stack (different ports, external database, etc.), copy `docker-compose-example.yml` to `docker-compose.yml`, fill in the environment variables, and run:

```bash
docker compose up
```

The three required environment variables in that file are:
- `IMAGE_NAME` — the image name you passed to `langgraph build -t`
- `LANGSMITH_API_KEY`
- `OPENAI_API_KEY` (or whichever model key your graph uses)

---

## 7. Confirming It's Running

After `langgraph up` starts streaming logs, open in a browser:

- **API docs:** `http://localhost:8123/docs`
- **Health check:** `http://localhost:8123/ok`

In a second terminal:
```bash
docker ps   # should show 3 running containers
```

---

## What to Understand Here

1. **`langgraph.json` is the entry point** — everything (which graph, which Python, which env file) is configured there. The notebooks and the CLI both read it.
2. **`langgraph build` bakes a snapshot** — the Docker image is frozen at build time. Changes to your Python code require a new `langgraph build`.
3. **Three containers, one stack** — Postgres stores durable state (threads, memories); Redis enables streaming; the API container runs your graph. All three are required.
4. **`langgraph up` vs `docker run`** — `langgraph up` orchestrates all three containers. `docker run` starts only the API, which then has nothing to connect to and crashes.
