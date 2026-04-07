# Walkthrough: Basics

> **Notebook:** `basics.ipynb`
> **Goal:** Set up the environment and understand the core building blocks — Chat Models, Messages, and Search Tools.

---

## 1. Install Dependencies

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langchain_community langchain-tavily
```

- `%%capture --no-stderr` suppresses install output in the notebook (keeps things clean)
- We install the LangChain core packages and the Tavily search integration

---

## 2. Set API Keys

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("MISTRAL_API_KEY")
```

**Line-by-line:**
- `os.environ.get(var)` — checks if the key is already set (e.g., from a `.env` file)
- `getpass.getpass(...)` — prompts you to type the key **without** echoing it on screen (secure input)
- The key is stored in `os.environ` so that LangChain SDK picks it up automatically

> This tutorial uses **Mistral AI** models. If you prefer OpenAI, replace `MISTRAL_API_KEY` with `OPENAI_API_KEY` and swap the model class.

---

## 3. Initialize Chat Models

```python
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

mistral_small = ChatMistralAI(model="mistral-small", temperature=0)
mistral_medium = ChatMistralAI(model="mistral-medium", temperature=0)
```

**Key parameters:**
| Parameter | Purpose |
|-----------|---------|
| `model` | Which model to use (e.g., `"mistral-small"`, `"gpt-4o"`) |
| `temperature` | Controls randomness. `0` = deterministic/factual, `1` = creative/varied |

**Why two models?** Different models trade off cost, speed, and quality. `mistral-small` is cheaper and faster; `mistral-medium` is more capable. You can swap freely because LangChain gives every chat model the **same interface**.

---

## 4. Messages

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="Hello world", name="Lance")
messages = [msg]
mistral_medium.invoke(messages)
```

**What's happening:**
- `HumanMessage` represents something **the user** says. The `name` field is optional metadata.
- Chat models accept a **list of messages** — this is the conversation history.
- `.invoke(messages)` sends the list to the model and returns an `AIMessage`.

**The response:**
```python
AIMessage(content="Hello, World! ...", response_metadata={...})
```

- `content` — the model's text reply
- `response_metadata` — token usage, model name, finish reason, etc.

**Shortcut** — you can also pass a plain string (it auto-wraps into a `HumanMessage`):

```python
mistral_medium.invoke("hello world")
```

---

## 5. Search Tools (Tavily)

```python
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)
data = tavily_search.invoke("What is LangGraph?")
search_docs = data.get("results", data)
```

**What's happening:**
- `TavilySearch` creates a search tool optimized for LLM consumption
- `max_results=3` limits to 3 results (less noise, lower cost)
- `.invoke(...)` takes a query string and returns structured results with `url`, `title`, `content`, and `score`

**Why Tavily?** Regular search engines return HTML pages. Tavily returns **clean, structured content** that's ready to feed into an LLM — making it ideal for RAG (Retrieval-Augmented Generation) and agent tool use.

---

## Key Takeaways

1. **Chat models** have a unified `.invoke()` interface — switch providers by changing one line
2. **Messages** are typed objects (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) that represent conversation roles
3. **Tools** like Tavily let models interact with external systems — this becomes critical when building agents in Module 1
