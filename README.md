# LangGraph Tutorial — From Basics to Production Deployment

A hands-on, notebook-by-notebook walkthrough of [LangGraph](https://github.com/langchain-ai/langgraph) — the framework for building stateful, multi-step AI agent workflows. Every notebook has a companion **walkthrough.md** that explains every line of code, the *why* behind each decision, and the patterns to take away.

> **Note:** This tutorial uses **Mistral AI** and **Groq** models instead of OpenAI. You can swap in any LangChain-compatible chat model.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Reference](#quick-reference)
- [Module 0 — Basics](#module-0--basics)
- [Module 1 — Simple Graphs to Agents](#module-1--simple-graphs-to-agents)
- [Module 2 — State and Memory](#module-2--state-and-memory)
- [Module 3 — Human-in-the-Loop](#module-3--human-in-the-loop)
- [Module 4 — Controllability](#module-4--controllability)
- [Module 5 — Long-term Memory](#module-5--long-term-memory)
- [Module 6 — Deployment](#module-6--deployment)
- [Key Concepts at a Glance](#key-concepts-at-a-glance)
- [Running LangGraph Studio](#running-langgraph-studio)
- [Project Structure](#project-structure)
- [Resources](#resources)

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — package and environment manager (replaces pip + venv)
- **[Docker](https://www.docker.com/)** — required for Module 6 deployment
- API keys (add to a `.env` file at the repo root):

| Key | Used in | Where to get it |
|-----|---------|-----------------|
| `MISTRAL_API_KEY` | Modules 1–3 | [console.mistral.ai](https://console.mistral.ai/) |
| `GROQ_API_KEY` | Modules 5–6 | [console.groq.com](https://console.groq.com/) |
| `TAVILY_API_KEY` | Modules 1, 4 | [tavily.com](https://tavily.com/) |
| `LANGSMITH_API_KEY` | All (tracing) + Module 6 (deployment) | [smith.langchain.com](https://smith.langchain.com/) |
| `EMAIL_ADDRESS` | Module 4 (Wikipedia API `User-Agent`) | Your own email |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Abdelrahman-Kanakri/LangGraph-Tutorial.git
cd LangGraph-Tutorial

# Create a virtual environment and install all dependencies
uv sync

# Activate it
source .venv/bin/activate
```

Add your API keys to a `.env` file at the repo root:

```bash
MISTRAL_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
EMAIL_ADDRESS=...
```

### Adding a dependency

```bash
uv add <package-name>   # never pip install into this project
```

---

## Quick Reference

The [`docs/`](docs/README.md) folder has one-page summaries per module — what it teaches, the key APIs, and when to use each pattern:

- [Module 0 — Basics](docs/module-0-basics.md)
- [Module 1 — From Simple Graphs to Deployed Agents](docs/module-1-graphs-to-agents.md)
- [Module 2 — State and Memory](docs/module-2-state-and-memory.md)
- [Module 3 — Human-in-the-Loop](docs/module-3-human-in-the-loop.md)
- [Module 4 — Controllability](docs/module-4-controllability.md)
- [Module 5 — Long-term Memory](docs/module-5-long-term-memory.md)
- [Module 6 — Deployment](docs/module-6-deployment.md)

For a full line-by-line explanation of any notebook, open its `walkthrough.md` next to the `.ipynb` file.

---

## Module 0 — Basics

Environment setup and the core building blocks: Chat Models, Messages, and Search Tools.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [basics.ipynb](module-0/01-basics/basics.ipynb) | [walkthrough.md](module-0/01-basics/walkthrough.md) | Chat models, `HumanMessage`/`AIMessage`, temperature, Tavily search |

---

## Module 1 — Simple Graphs to Agents

A progressive build from the simplest graph to a deployed ReAct agent with persistent memory.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 0 | [motivation](module-1/00-motivation/walkthrough.md) | [walkthrough.md](module-1/00-motivation/walkthrough.md) | Why LangGraph? The gap between a plain LLM call and a real-world agent |
| 1 | [simple_graph.ipynb](module-1/01-simple-graph/simple_graph.ipynb) | [walkthrough.md](module-1/01-simple-graph/walkthrough.md) | `TypedDict` state, nodes, edges, conditional edges, `START`/`END` |
| 2 | [chain.ipynb](module-1/02-chain/chain.ipynb) | [walkthrough.md](module-1/02-chain/walkthrough.md) | Messages as state, `add_messages` reducer, `MessagesState`, tool binding |
| 3 | [router.ipynb](module-1/03-router/router.ipynb) | [walkthrough.md](module-1/03-router/walkthrough.md) | `ToolNode`, `tools_condition`, routing between tool execution and direct response |
| 4 | [agent.ipynb](module-1/04-agent/agent.ipynb) | [walkthrough.md](module-1/04-agent/walkthrough.md) | ReAct loop (`tools → assistant`), multi-step tool chaining, system messages |
| 5 | [agent-memory.ipynb](module-1/05-agent-memory/agent-memory.ipynb) | [walkthrough.md](module-1/05-agent-memory/walkthrough.md) | `MemorySaver` checkpointer, `thread_id`, multi-turn persistence |
| 6 | [deployment.ipynb](module-1/06-deployment/deployment.ipynb) | [walkthrough.md](module-1/06-deployment/walkthrough.md) | LangGraph Studio, LangGraph SDK, local deployment |

---

## Module 2 — State and Memory

Deep dive into state schemas, reducers, message management, and building a production chatbot with durable memory.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [state-schema.ipynb](module-2/01-state-schema/state-schema.ipynb) | [walkthrough.md](module-2/01-state-schema/walkthrough.md) | `TypedDict` vs `Dataclass` vs `Pydantic`, runtime validation |
| 2 | [state-reducers.ipynb](module-2/02-state-reducers/state-reducers.ipynb) | [walkthrough.md](module-2/02-state-reducers/walkthrough.md) | Default overwrite, `operator.add`, custom reducers, `add_messages`, `RemoveMessage` |
| 3 | [multiple-schema.ipynb](module-2/03-multiple-schema/multiple-schema.ipynb) | [walkthrough.md](module-2/03-multiple-schema/walkthrough.md) | Private state between nodes, input/output schemas |
| 4 | [trim-filtering-messages.ipynb](module-2/04-trim-filter-messages/trim-filtering-messages.ipynb) | [walkthrough.md](module-2/04-trim-filter-messages/walkthrough.md) | `RemoveMessage`, message filtering, `trim_messages` with token counting |
| 5 | [chatbot-summerization.ipynb](module-2/05-chatbot-summerization/chatbot-summerization.ipynb) | [walkthrough.md](module-2/05-chatbot-summerization/walkthrough.md) | Running summary, conditional summarization, extended `MessagesState` |
| 6 | [chatbot-external-memory.ipynb](module-2/06-chatbot-external-memory/chatbot-external-memory.ipynb) | [walkthrough.md](module-2/06-chatbot-external-memory/walkthrough.md) | `SqliteSaver`, persistent memory on disk, `MemorySaver` vs `SqliteSaver` |

---

## Module 3 — Human-in-the-Loop

Streaming, interrupting, and editing graph execution mid-run — enabling approval workflows, human feedback, and time travel.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [streaming-interruption.ipynb](module-3/01-streaming-interruption/streaming-interruption.ipynb) | [walkthrough.md](module-3/01-streaming-interruption/walkthrough.md) | `stream_mode` values/updates, `astream_events`, token-by-token streaming |
| 2 | [breakpoints.ipynb](module-3/02-breakpoints/breakpoints.ipynb) | [walkthrough.md](module-3/02-breakpoints/walkthrough.md) | `interrupt_before`, `graph.get_state()`, resuming with `None`, human approval |
| 3 | [edit-state-human-feedback.ipynb](module-3/03-edit-state-human-feedback/edit-state-human-feedback.ipynb) | [walkthrough.md](module-3/03-edit-state-human-feedback/walkthrough.md) | `graph.update_state()`, append vs overwrite by message ID, `human_feedback` node |
| 4 | [dynamic-breakpoints.ipynb](module-3/04-dynamic-breakpoints/dynamic-breakpoints.ipynb) | [walkthrough.md](module-3/04-dynamic-breakpoints/walkthrough.md) | `NodeInterrupt`, conditional interrupts from inside a node, fixing state to resume |
| 5 | [time-travel.ipynb](module-3/05-time-travel/time-travel.ipynb) | [walkthrough.md](module-3/05-time-travel/walkthrough.md) | `get_state_history`, replaying via `checkpoint_id`, forking with `update_state` |

---

## Module 4 — Controllability

Advanced graph patterns for parallelism, composition, and building a full multi-agent research assistant.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [parallelization.ipynb](module-4/01-parallelization/parallelization.ipynb) | [walkthrough.md](module-4/01-parallelization/walkthrough.md) | Fan-out/fan-in, parallel node execution, `Send` API |
| 2 | [sub-graph.ipynb](module-4/02-sub-graph/sub-graph.ipynb) | [walkthrough.md](module-4/02-sub-graph/walkthrough.md) | Nesting graphs, state handoff at boundaries, independent subgraph compilation |
| 3 | [map-reduce.ipynb](module-4/03-map-reduce/map-reduce.ipynb) | [walkthrough.md](module-4/03-map-reduce/walkthrough.md) | Dynamic fan-out with `Send`, aggregating results, map-reduce over variable inputs |
| 4 | [research-assistant.ipynb](module-4/04-research-assistant/research-assistant.ipynb) | [walkthrough.md](module-4/04-research-assistant/walkthrough.md) | Multi-agent research pipeline: plan → search → write using parallelization + subgraphs |

---

## Module 5 — Long-term Memory

Cross-session memory using the LangGraph memory store — profiles, collections, and a full memory-aware agent.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [memory_store.ipynb](module-5/01-memory-store/memory_store.ipynb) | [walkthrough.md](module-5/01-memory-store/walkthrough.md) | `InMemoryStore`, namespaces, `store.put` / `store.search`, cross-thread persistence |
| 2 | [memoryschema_profile.ipynb](module-5/02-memory-schema-profile/memoryschema_profile.ipynb) | [walkthrough.md](module-5/02-memory-schema-profile/walkthrough.md) | Trustcall extractor, single-document profile schema, upsert with `PatchDoc` |
| 3 | [memoryschema_collection.ipynb](module-5/03-memoryschema-collection/memoryschema_collection.ipynb) | [walkthrough.md](module-5/03-memoryschema-collection/walkthrough.md) | Collection-style memory, `enable_inserts=True`, managing many items per namespace |
| 4 | [memory_agent.ipynb](module-5/04-memory-agent/memory_agent.ipynb) | [walkthrough.md](module-5/04-memory-agent/walkthrough.md) | Full `task_mAIstro` agent: profile + todo + instructions memory, `UpdateMemory` routing |

---

## Module 6 — Deployment

Package, deploy, and interact with a LangGraph app in production using Docker, the LangGraph SDK, and the Assistants API.

> **Before running the notebooks:** follow the [Deployment Guide](module-6/DEPLOYMENT_GUIDE.md) to build the Docker image and start the stack.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [creating_deployment.ipynb](module-6/01-createing_deployment/creating_deployment.ipynb) | [walkthrough.md](module-6/01-createing_deployment/walkthrough.md) | `langgraph.json`, `langgraph build`, `langgraph up`, 3-container stack (API + Postgres + Redis) |
| 2 | [connecting.ipynb](module-6/02-connecting/connecting.ipynb) | [walkthrough.md](module-6/02-connecting/walkthrough.md) | SDK vs `RemoteGraph`, background/blocking/streaming runs, threads, fork & HITL, store API |
| 3 | [double_texting.ipynb](module-6/03-double_texting/double_texting.ipynb) | [walkthrough.md](module-6/03-double_texting/walkthrough.md) | `multitask_strategy`: reject / enqueue / interrupt / rollback |
| 4 | [assistant.ipynb](module-6/04-assistant/assistant.ipynb) | [walkthrough.md](module-6/04-assistant/walkthrough.md) | Creating versioned assistants, `todo_category` for storage isolation, search & delete |

---

## Key Concepts at a Glance

### The Full Progression

```text
Module 0     Module 1                              Module 2                              Module 3
─────────    ─────────────────────────────────    ──────────────────────────────────    ─────────────────────────────
Basics    →  Simple Graph → Chain → Router →      State Schema → Reducers →             Streaming → Breakpoints →
             Agent → Agent Memory → Deploy        Multiple Schemas → Trim/Filter →      Human Feedback →
                                                  Summarization → External DB           Dynamic Breakpoints → Time Travel

Module 4                              Module 5                              Module 6
──────────────────────────────────    ──────────────────────────────────    ──────────────────────────────────────
Parallelization → Subgraphs →         Memory Store → Schema Profile →       Creating Deployment →
Map-Reduce → Research Assistant       Schema Collection → Memory Agent      Connecting → Double Texting → Assistants
```

### When to Use What

| Scenario | Pattern |
|----------|---------|
| Fixed workflow with branching | Simple Graph |
| LLM call with tool awareness | Chain |
| LLM decides: tool vs. response | Router |
| Multi-step reasoning with tools | Agent (ReAct) |
| Persistent multi-turn conversations | Agent + `MemorySaver` / `SqliteSaver` |
| Long conversations (token management) | Trim / Filter / Summarize |
| Human approval before an action | Breakpoint (`interrupt_before`) |
| Editing past state mid-run | `update_state` + `checkpoint_id` |
| Parallel workloads | Fan-out + `Send` API |
| Reusable graph components | Subgraphs |
| Dynamic number of parallel tasks | Map-Reduce (`Send`) |
| Memory across sessions | `InMemoryStore` / Postgres store + Trustcall |
| Production deployment | `langgraph build` + `langgraph up` |
| Multiple personas from one graph | Assistants API (`client.assistants.create`) |
| Concurrent user messages | `multitask_strategy` (enqueue / reject / interrupt / rollback) |

### Best Practice: Always Use List Form for Messages

```python
# ✅ Always do this
{"messages": [response]}
graph.invoke({"messages": [HumanMessage("What is 5 * 3?")]})

# ❌ Fragile — works only because add_messages is permissive
{"messages": response}
graph.invoke({"messages": HumanMessage("What is 5 * 3?")})
```

The raw form is silently accepted today but breaks if you ever swap in a stricter reducer. Use the list form everywhere.

### State Schema Decision Guide

| Need | Use |
|------|-----|
| Quick prototyping | `TypedDict` |
| Clean attribute access | `dataclass` |
| Runtime validation | `Pydantic` |
| Message-based state | `MessagesState` |
| Custom update logic | `Annotated` + reducer |
| Hide internal node data | Private state / input-output schemas |

---

## Running LangGraph Studio

Modules 1–5 each have a `studio/` directory with deployment-ready code for local Studio:

```bash
cd module-3/studio   # or any other module's studio/
langgraph dev
```

The `langgraph.json` in each studio directory defines which graphs are served. Studio runs at `http://localhost:2024` by default.

---

## Project Structure

```text
LangGraph-Tutorial/
├── .env                          # API keys (not committed)
├── langgraph.json                # Root deployment config
├── pyproject.toml                # Project deps (managed by uv)
├── module-0/
│   └── 01-basics/
│       ├── basics.ipynb
│       └── walkthrough.md
├── module-1/
│   ├── 00-motivation/
│   │   └── walkthrough.md
│   ├── 01-simple-graph/
│   ├── 02-chain/
│   ├── 03-router/
│   ├── 04-agent/
│   ├── 05-agent-memory/
│   ├── 06-deployment/
│   └── studio/
├── module-2/
│   ├── 01-state-schema/
│   ├── 02-state-reducers/
│   ├── 03-multiple-schema/
│   ├── 04-trim-filter-messages/
│   ├── 05-chatbot-summerization/
│   ├── 06-chatbot-external-memory/
│   └── studio/
├── module-3/
│   ├── 01-streaming-interruption/
│   ├── 02-breakpoints/
│   ├── 03-edit-state-human-feedback/
│   ├── 04-dynamic-breakpoints/
│   ├── 05-time-travel/
│   └── studio/
├── module-4/
│   ├── 01-parallelization/
│   ├── 02-sub-graph/
│   ├── 03-map-reduce/
│   ├── 04-research-assistant/
│   └── studio/
├── module-5/
│   ├── 01-memory-store/
│   ├── 02-memory-schema-profile/
│   ├── 03-memoryschema-collection/
│   ├── 04-memory-agent/
│   └── studio/
├── module-6/
│   ├── 01-createing_deployment/
│   ├── 02-connecting/
│   ├── 03-double_texting/
│   ├── 04-assistant/
│   ├── DEPLOYMENT_GUIDE.md       # Step-by-step Docker build + run guide
│   └── deployment/               # Deployable app (task_maistro)
│       ├── task_maistro.py
│       ├── configuration.py
│       ├── requirements.txt
│       ├── langgraph.json
│       └── .env                  # → points to ../../.env
└── docs/                         # One-page quick references per module
    ├── README.md
    ├── module-0-basics.md
    ├── module-1-graphs-to-agents.md
    ├── module-2-state-and-memory.md
    ├── module-3-human-in-the-loop.md
    └── module-4-controllability.md
```

Each leaf folder (`01-simple-graph/`, etc.) contains one `.ipynb` notebook and one `walkthrough.md`.

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Platform Docs](https://docs.langchain.com/langsmith/)
- [LangChain Academy](https://academy.langchain.com/)
- [LangSmith](https://smith.langchain.com/) — tracing, evaluation, and deployment
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)
- [Trustcall](https://github.com/hinthornw/trustcall) — structured memory extraction used in Module 5–6

---

If you found this helpful, feel free to star the repo.
