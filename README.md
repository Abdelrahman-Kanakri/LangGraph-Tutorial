# LangGraph Tutorial - From Basics to Agents

A hands-on, notebook-by-notebook walkthrough of [LangGraph](https://github.com/langchain-ai/langgraph) — the framework for building stateful, multi-step AI agent workflows. Each notebook comes with a detailed **walkthrough.md** that explains every line of code.

> **Note:** This tutorial uses **Mistral AI** models (`mistral-small`, `mistral-medium`) instead of OpenAI. You can swap in any LangChain-compatible chat model.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Module 0 - Basics](#module-0---basics)
- [Module 1 - Simple Graphs to Agents](#module-1---simple-graphs-to-agents)
- [Module 2 - State and Memory](#module-2---state-and-memory)
- [Key Concepts at a Glance](#key-concepts-at-a-glance)
- [Running LangGraph Studio](#running-langgraph-studio)
- [Project Structure](#project-structure)
- [Resources](#resources)

---

## Prerequisites

- Python 3.11+
- [Docker](https://www.docker.com/) (for LangGraph Studio)
- API keys:
  - `MISTRAL_API_KEY` (or `OPENAI_API_KEY` if you swap models)
  - `TAVILY_API_KEY` (for search tools — used in later modules)
  - `LANGSMITH_API_KEY` (optional, for tracing & deployment)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/LangGraph-Tutorial.git
cd LangGraph-Tutorial

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### LangGraph Studio (Optional)

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev  # run inside any module's /studio directory
```

---

## Module 0 - Basics

Environment setup and the core building blocks — Chat Models, Messages, and Search Tools.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [basics.ipynb](module-0/01-basics/basics.ipynb) | [walkthrough.md](module-0/01-basics/walkthrough.md) | Chat models, `HumanMessage`/`AIMessage`, temperature, Tavily search |

---

## Module 1 - Simple Graphs to Agents

A progressive build from the simplest graph to a deployed ReAct agent with memory.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [simple_graph.ipynb](module-1/01-simple-graph/simple_graph.ipynb) | [walkthrough.md](module-1/01-simple-graph/walkthrough.md) | `TypedDict` state, nodes, edges, conditional edges, `START`/`END` |
| 2 | [chain.ipynb](module-1/02-chain/chain.ipynb) | [walkthrough.md](module-1/02-chain/walkthrough.md) | Messages as state, `add_messages` reducer, `MessagesState`, tool binding |
| 3 | [router.ipynb](module-1/03-router/router.ipynb) | [walkthrough.md](module-1/03-router/walkthrough.md) | `ToolNode`, `tools_condition`, routing between tool execution and direct response |
| 4 | [agent.ipynb](module-1/04-agent/agent.ipynb) | [walkthrough.md](module-1/04-agent/walkthrough.md) | ReAct loop (`tools -> assistant`), multi-step tool chaining, system messages |
| 5 | [agent-memory.ipynb](module-1/05-agent-memory/agent-memory.ipynb) | [walkthrough.md](module-1/05-agent-memory/walkthrough.md) | `MemorySaver` checkpointer, `thread_id`, multi-turn persistence |
| 6 | [deployment.ipynb](module-1/06-deployment/deployment.ipynb) | [walkthrough.md](module-1/06-deployment/walkthrough.md) | LangGraph Studio, LangGraph SDK, local & cloud deployment |

---

## Module 2 - State and Memory

Deep dive into state schemas, reducers, message management, and building a production chatbot.

| # | Notebook | Walkthrough | Topics |
|---|----------|-------------|--------|
| 1 | [state-schema.ipynb](module-2%20/01-state-schema/state-schema.ipynb) | [walkthrough.md](module-2%20/01-state-schema/walkthrough.md) | `TypedDict` vs `Dataclass` vs `Pydantic`, runtime validation |
| 2 | [state-reducers.ipynb](module-2%20/02-state-reducers/state-reducers.ipynb) | [walkthrough.md](module-2%20/02-state-reducers/walkthrough.md) | Default overwrite, `operator.add`, custom reducers, `add_messages`, `RemoveMessage` |
| 3 | [multiple-schema.ipynb](module-2%20/03-multiple-schema/multiple-schema.ipynb) | [walkthrough.md](module-2%20/03-multiple-schema/walkthrough.md) | Private state between nodes, input/output schemas |
| 4 | [trim-filtering-messages.ipynb](module-2%20/04-trim-filter-messages%20/trim-filtering-messages.ipynb) | [walkthrough.md](module-2%20/04-trim-filter-messages%20/walkthrough.md) | `RemoveMessage`, message filtering, `trim_messages` with token counting |
| 5 | [chatbot-summerization.ipynb](module-2%20/05-chatbot-summerization/chatbot-summerization.ipynb) | [walkthrough.md](module-2%20/05-chatbot-summerization/walkthrough.md) | Running summary, conditional summarization, extended `MessagesState` |
| 6 | [chatbot-external-memory.ipynb](module-2%20/06-chatbot-external-memory/chatbot-external-memory.ipynb) | [walkthrough.md](module-2%20/06-chatbot-external-memory/walkthrough.md) | `SqliteSaver`, persistent memory on disk, `MemorySaver` vs `SqliteSaver` |

---

## Key Concepts at a Glance

### The Progression

```
Module 0          Module 1                                           Module 2
────────    ─────────────────────────────────────    ──────────────────────────────────────────
Basics  →   Simple Graph → Chain → Router → Agent    State Schema → Reducers → Multiple Schemas
              │              │       │        │      → Trim/Filter → Summarization → External DB
           TypedDict     Messages ToolNode  ReAct
           + Nodes       + LLM    + Cond.   Loop
           + Edges       + Tools   Edge    + Memory
```

### When to Use What

| Scenario | Pattern |
|----------|---------|
| Fixed workflow with branching | Simple Graph |
| LLM call with tool awareness | Chain |
| LLM decides tool vs. response | Router |
| Multi-step reasoning with tools | Agent (ReAct) |
| Persistent conversations | Agent + Memory |
| Long conversations (token management) | Trim / Filter / Summarize |
| Production deployment | SqliteSaver + LangGraph Studio |

### State Schema Decision Guide

| Need | Use |
|------|-----|
| Quick prototyping | `TypedDict` |
| Clean attribute access | `dataclass` |
| Runtime validation | `Pydantic` |
| Message-based state | `MessagesState` |
| Custom update logic | `Annotated` + reducer |
| Hide internal data | Private state / input-output schemas |

---

## Running LangGraph Studio

Each module has a `studio/` directory with deployment-ready code:

```bash
cd module-1/studio   # or module-2\ /studio
langgraph dev
```

The `langgraph.json` in each studio directory defines which graphs are available.

---

## Project Structure

```
LangGraph-Tutorial/
├── module-0/
│   └── 01-basics/
│       ├── basics.ipynb
│       └── walkthrough.md
├── module-1/
│   ├── 01-simple-graph/
│   │   ├── simple_graph.ipynb
│   │   └── walkthrough.md
│   ├── 02-chain/
│   │   ├── chain.ipynb
│   │   └── walkthrough.md
│   ├── 03-router/
│   │   ├── router.ipynb
│   │   └── walkthrough.md
│   ├── 04-agent/
│   │   ├── agent.ipynb
│   │   └── walkthrough.md
│   ├── 05-agent-memory/
│   │   ├── agent-memory.ipynb
│   │   └── walkthrough.md
│   ├── 06-deployment/
│   │   ├── deployment.ipynb
│   │   └── walkthrough.md
│   └── studio/
├── module-2 /
│   ├── 01-state-schema/
│   │   ├── state-schema.ipynb
│   │   └── walkthrough.md
│   ├── 02-state-reducers/
│   │   ├── state-reducers.ipynb
│   │   └── walkthrough.md
│   ├── 03-multiple-schema/
│   │   ├── multiple-schema.ipynb
│   │   └── walkthrough.md
│   ├── 04-trim-filter-messages/
│   │   ├── trim-filtering-messages.ipynb
│   │   └── walkthrough.md
│   ├── 05-chatbot-summerization/
│   │   ├── chatbot-summerization.ipynb
│   │   └── walkthrough.md
│   ├── 06-chatbot-external-memory/
│   │   ├── chatbot-external-memory.ipynb
│   │   └── walkthrough.md
│   └── studio/
├── academy_notebooks/            # Original LangChain Academy reference (modules 2-6)
├── requirements.txt
└── langgraph.json
```

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Academy](https://academy.langchain.com/)
- [LangSmith](https://smith.langchain.com/) — Tracing & Deployment
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)

---

**Happy Learning!** If you found this helpful, feel free to star the repo and share it with your network.
