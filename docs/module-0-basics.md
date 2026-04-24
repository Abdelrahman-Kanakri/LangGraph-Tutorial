# Module 0 — Basics

Environment setup and the three primitives every LangGraph app is built on: chat models, messages, and external tools.

---

## 01 · basics

**Notebook:** [basics.ipynb](../module-0/01-basics/basics.ipynb) · **Walkthrough:** [walkthrough.md](../module-0/01-basics/walkthrough.md)

**What it teaches**
The core LangChain building blocks used throughout the rest of the tutorial — installing packages, setting API keys securely, initializing a chat model, working with message types, and calling a search tool.

**Key APIs**

- `ChatMistralAI(model="mistral-small", temperature=0)` — swap for `ChatOpenAI` to switch providers
- `HumanMessage(content=..., name=...)`, `AIMessage`, `SystemMessage`, `ToolMessage`
- `model.invoke(messages)` — single unified interface across all chat models
- `TavilySearch(max_results=3)` — LLM-friendly search returning structured `{url, title, content, score}`
- `getpass.getpass(...)` — prompt for API keys without echoing

**When to use**
Any time you're starting a new notebook that needs a chat model or a search tool — the setup pattern here is reused in every later module.

**Output type recap**
`.invoke(messages)` returns an `AIMessage` whose `.content` is the text and `.response_metadata` contains tokens/finish reason.
