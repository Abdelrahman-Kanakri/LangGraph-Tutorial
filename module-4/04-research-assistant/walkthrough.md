# Walkthrough: Research Assistant — The Capstone

> **Notebook:** `research-assistant.ipynb`
> **Goal:** Put everything from modules 1–4 together — **human-in-the-loop planning**, **sub-graphs**, **parallelization**, and **map-reduce via `Send()`** — into a single multi-agent system that produces a finished, cited research report from a topic prompt.

---

## The Big Picture

This is the capstone of module 4. Every primitive we built on its own — fan-out, sub-graphs, `Send()`, reducers, interrupts — now combines into one pipeline that mirrors how a real research team works:

```
            user gives a topic
                    │
                    ▼
         ┌── create_analysts ──┐         ← LLM picks the sub-topics
         │                     │
         ▼                     │
   human_feedback ─────────────┘         ← INTERRUPT: human can refine the team
         │
         ▼  (Send → one per analyst, in parallel)
   ┌──── conduct_interview (sub-graph) ────┐
   │   ask_question                        │
   │   ├── search_web ──┐                  │  ← parallel retrieval
   │   └── search_wiki ─┤                  │
   │       answer_question                 │
   │       (loop until max_turns)          │
   │       save_interview → write_section  │
   └───────────────────────────────────────┘
         │  (all interviews fan in)
         ▼
   ┌── write_introduction ──┐
   │── write_report ────────┤              ← parallel writing
   └── write_conclusion ────┘
         │
         ▼
      finalize_report ─→ END
```

**Map the diagram back to the four module-4 lessons:**

| Lesson | Where it shows up |
|--------|-------------------|
| **01 — Parallelization** | `search_web` + `search_wikipedia` run side by side inside each interview; `write_introduction` + `write_report` + `write_conclusion` run side by side after interviews |
| **02 — Sub-graphs** | `conduct_interview` is a **compiled sub-graph** plugged in as a single node of the outer graph |
| **03 — Map-reduce (`Send()`)** | `initiate_all_interviews` fans out **one `Send` per analyst** so all interviews run in parallel; the reducer on `sections` collects every interview's output |
| **Module 3 — Human-in-the-loop** | `interrupt_before=['human_feedback']` lets the user edit the analyst team between runs |

> ### 📌 Reducers — the rule of this module
>
> *A reducer controls how a new value is merged into the existing value for a key. You need one when you either want accumulation, or when multiple nodes write the same key concurrently and you can't afford to lose any write.*
>
> Two reducers are doing real work in this notebook:
> - `context: Annotated[list, operator.add]` on **`InterviewState`** — `search_web` and `search_wikipedia` write `context` in the same super-step.
> - `sections: Annotated[list, operator.add]` on **`ResearchGraphState`** — N `Send`-spawned interviews all write `sections` concurrently. Without this reducer, only one interview's section would survive.

---

## 1. Setup

The notebook installs the usual stack and sets the keys it needs. Two are strictly required to run the graph (the LLM key and Tavily); the Wikipedia user-agent email is required by Wikipedia's API policy; LangSmith is optional but useful for inspecting traces.

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENROUTER_API_KEY")     # or MISTRAL_API_KEY, depending on the model you wire in
_set_env("TAVILY_API_KEY")         # required for search_web
_set_env("LANGSMITH_API_KEY")      # optional — for tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"
```

### Wikipedia user-agent

Wikipedia's [API policy](https://meta.wikimedia.org/wiki/User-Agent_policy) requires every client to identify itself with a **User-Agent string that includes a contact email** — so they can reach you if your traffic starts causing issues. The notebook and studio file both wire this up from an env var rather than hard-coding an address:

```python
import wikipedia
from langchain_community.document_loaders import WikipediaLoader

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
wikipedia.set_user_agent(f"research-assistant/1.0 ({EMAIL_ADDRESS})")
```

If `EMAIL_ADDRESS` isn't set, the User-Agent literally becomes `research-assistant/1.0 (None)` — Wikipedia may rate-limit or block you. Make sure `EMAIL_ADDRESS` is in your `.env` before running the graph.

> **Studio note.** The `studio/research_assistant.py` version uses `ChatMistralAI` so the deployment only needs a single `MISTRAL_API_KEY` (the same key the other studio graphs in this module use). The notebook itself uses `ChatOpenRouter` so you can swap models freely. Both versions read `EMAIL_ADDRESS` from the environment for the Wikipedia user-agent.

---

## 2. Analysts — Structured Output + Why `Field()` Matters

The first thing the graph does is **generate a team of analysts** to investigate the topic. Each analyst is a Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name:        str = Field(description="Name of the analyst.")
    role:        str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return (f"Name: {self.name}\nRole: {self.role}\n"
                f"Affiliation: {self.affiliation}\nDescription: {self.description}\n")

class Perspectives(BaseModel):
    analysts: List[Analyst]

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
```

`BaseModel` enforces types at runtime; `Field(description=...)` is what the LLM actually sees when called with `with_structured_output(Perspectives)`. The descriptions become part of the JSON Schema sent to the model, so they're how you **prompt the model from inside the data model**. Vague descriptions → vague analysts. Specific descriptions → useful analysts.

The `.persona` property is just a string-formatter; it's what we splice into the interviewer's system message later.

---

## 3. Generating Analysts (and Letting a Human Edit Them)

```python
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=state["topic"],
        human_analyst_feedback=state.get("human_analyst_feedback", ""),
        max_analysts=state["max_analysts"],
    )
    analysts = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts."),
    ])
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on."""
    pass
```

`human_feedback` is the **interrupt anchor** — a no-op the graph stops *before*, so the caller can either:

- inspect the proposed analysts and accept them (`update_state(..., {"human_analyst_feedback": None})`), or
- send revisions (`update_state(..., {"human_analyst_feedback": "Add a startup founder"})`), which routes the graph back to `create_analysts` for another draft.

This is exactly the human-in-the-loop pattern from module 3 — interrupt + `update_state` — reused as the *planning gate* of the research workflow.

```python
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback",  human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

memory = MemorySaver()
graph  = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
```

The checkpointer is **required** here — `update_state` only works on a graph that persists state. Without it, the interrupt would freeze the run with no way to resume.

---

## 4. The Interview Sub-graph

Each analyst conducts a full interview by themselves, in their own private state space. That's the perfect job for a sub-graph (lesson 02).

### 4a. Interview state

```python
class InterviewState(MessagesState):
    max_num_turns: int
    context:   Annotated[list, operator.add]   # parallel retrieval → REDUCER
    analyst:   Analyst
    interview: str
    sections:  list                            # written by write_section, surfaces to parent
```

`MessagesState` gives us the standard `messages` channel (with its built-in reducer for chat messages). `context` gets a reducer because two retrievers will write it concurrently inside the same super-step.

### 4b. The expert's question

```python
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + state["messages"])
    return {"messages": [question]}
```

`MessagesState`'s built-in messages reducer appends the new question to the running transcript — no manual list-concat needed.

### 4c. Parallel retrieval

Two retrieval nodes run side by side every turn, with the same query-refinement step in front:

```python
MAX_QUERY_LEN = 380   # Tavily errors above 400

def _build_query(state):
    last_question = next(
        (m.content for m in reversed(state['messages']) if isinstance(m, AIMessage)),
        "",
    )
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([
        search_instructions,
        HumanMessage(content=f"Generate a concise web search query (under 50 words) for this question:\n\n{last_question}"),
    ])
    query = search_query.search_query if (search_query and getattr(search_query, "search_query", None)) \
            else (last_question.split('.')[0] if last_question else "")
    return query[:MAX_QUERY_LEN].strip()

def search_web(state):
    query = _build_query(state)
    if not query: return {"context": []}
    data = tavily_search.invoke({"query": query})
    if not isinstance(data, dict) or "results" not in data: return {"context": []}
    formatted = "\n\n---\n\n".join(
        f'<Document href="{d["url"]}"/>\n{d["content"]}\n</Document>'
        for d in data["results"]
    )
    return {"context": [formatted]}

def search_wikipedia(state):
    query = _build_query(state)
    if not query: return {"context": []}
    try:
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
    except Exception:
        return {"context": []}
    formatted = "\n\n---\n\n".join(
        f'<Document source="{d.metadata.get("source", "Wikipedia")}" title="{d.metadata.get("title", "")}"/>\n{d.page_content}\n</Document>'
        for d in docs
    )
    return {"context": [formatted]}
```

**Three patterns to notice — all from lesson 01:**

1. Each node returns `{"context": [formatted]}` — a **list of length 1**. `operator.add` concatenates the two lists into a 2-element `context`.
2. Defensive fallbacks (`if not query: return {"context": []}`, `try/except` around the Wikipedia loader) keep one bad call from crashing the whole interview.
3. `MAX_QUERY_LEN = 380` is a hard cap because Tavily errors above 400 chars. The LLM doesn't always honor "under 50 words" — the slice is the safety net.

### 4d. Mistral-specific gotcha: `ensure_human_last`

The notebook documents a Mistral-only quirk: Mistral rejects requests where the **last message is an `AIMessage`** with a 400 error, because it strictly requires a `HumanMessage` (or tool result) before responding.

In a normal chatbot you never hit this — humans naturally type between AI turns. In an agent graph like ours, two nodes call the LLM back-to-back; by turn 2, `state['messages']` ends with an `AIMessage`. The fix is a tiny guard that appends a throwaway `HumanMessage` to a **local copy** of the message list before `.invoke()`:

```python
def ensure_human_last(messages):
    if messages and not isinstance(messages[-1], HumanMessage):
        return messages + [HumanMessage(content="continue")]
    return messages
```

The throwaway never touches state. OpenAI and Anthropic don't need this. If you switch models, the guard is harmless to leave in.

### 4e. Expert answer + loop control

```python
def generate_answer(state):
    system_message = answer_instructions.format(
        goals=state["analyst"].persona, context=state["context"]
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + state["messages"])
    answer.name = "expert"     # tags the message so route_messages can count expert turns
    return {"messages": [answer]}

def route_messages(state, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    if num_responses >= max_num_turns:
        return "save_interview"
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"
```

Two stop conditions: **max turns reached**, or the analyst says the magic phrase. Both end the interview; otherwise we loop back to `ask_question`.

`answer.name = "expert"` is what makes the counting possible — `route_messages` filters messages by `.name == "expert"` to know how many *expert* turns have passed, ignoring analyst turns.

### 4f. Save + write the section

```python
def save_interview(state):
    return {"interview": get_buffer_string(state["messages"])}

def write_section(state):
    system_message = section_writer_instructions.format(focus=state["analyst"].description)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {state['context']}"),
    ])
    return {"sections": [section.content]}
```

`save_interview` flattens the transcript into a string (handy for tracing/debugging). `write_section` returns `{"sections": [section.content]}` — again a **list of length 1**, because `sections` is reducer-merged into the *outer* graph's `sections` list when this sub-graph runs under `Send()`.

### 4g. Wiring the interview sub-graph

```python
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question",    generate_question)
interview_builder.add_node("search_web",      search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview",  save_interview)
interview_builder.add_node("write_section",   write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web",       "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)
```

This is a standalone, testable `StateGraph` — you can compile and invoke it on a single `Analyst` without touching the outer graph.

---

## 5. The Outer Graph — Map-Reduce Over Interviews

Now we wrap the interview sub-graph into the full research pipeline.

### 5a. Outer state

```python
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]   # ← THE Send() reducer
    introduction: str
    content: str
    conclusion: str
    final_report: str
```

| Key | Writers | Reducer | Why |
|-----|---------|---------|-----|
| `topic`, `max_analysts` | caller, once | none | Single write at `invoke()` |
| `human_analyst_feedback` | `update_state` after interrupt | none | Single writer per cycle |
| `analysts` | `create_analysts` only | none | One writer per cycle |
| `sections` | **N parallel interview sub-graphs** | **`operator.add`** | Concurrent `Send`-spawned writes |
| `introduction`, `content`, `conclusion` | one writer each | none | Three sibling nodes, but each writes a *different* key |
| `final_report` | `finalize_report` only | none | Last writer |

The only reducer the outer graph *needs* is on `sections` — because that's the key the parallel interviews all push to.

### 5b. The `Send()` fan-out

```python
def initiate_all_interviews(state: ResearchGraphState):
    """Map step: route back if there's feedback, or fan out one Send per analyst."""
    if state.get("human_analyst_feedback"):
        return "create_analysts"      # branch A — back to planning

    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")],
        })
        for analyst in state["analysts"]
    ]
```

This is the **map step from lesson 03**, with one extra trick: the *same* conditional edge function decides between "loop back to re-plan" (returning a string node name) and "fan out" (returning a list of `Send` objects). LangGraph happily handles both branches from a single function — that's why the third argument to `add_conditional_edges` lists *both* possible targets:

```python
builder.add_conditional_edges(
    "human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"]
)
```

Each `Send("conduct_interview", payload)` spawns one copy of the interview sub-graph with a custom payload. Because every interview returns `{"sections": [its_section]}`, the `operator.add` reducer on the parent's `sections` key merges them into a single list of length N.

### 5c. Parallel report writing

After all interviews finish, three writer nodes run **in parallel** (lesson 01 fan-out):

```python
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
```

No reducer needed on `introduction`, `content`, or `conclusion`: each is written by exactly one node, so single-writer overwrite is fine. The `["write_conclusion", "write_report", "write_introduction"]` list-form fan-in makes `finalize_report` wait until **all three** finish.

### 5d. Finalize: stitch + dedupe sources

```python
def finalize_report(state):
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except ValueError:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}
```

The body of the report already comes back from `write_report` with a `## Sources` block. We split it out, glue the report together in the right order, and re-attach the sources at the very bottom so they appear only once.

### 5e. Wiring + compile

```python
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts",       create_analysts)
builder.add_node("human_feedback",        human_feedback)
builder.add_node("conduct_interview",     interview_builder.compile())   # ← sub-graph as a node
builder.add_node("write_report",          write_report)
builder.add_node("write_introduction",    write_introduction)
builder.add_node("write_conclusion",      write_conclusion)
builder.add_node("finalize_report",       finalize_report)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews,
                              ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

memory = MemorySaver()
graph  = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
```

`interview_builder.compile()` returns a runnable, and we register it as a node. The outer graph never knows it isn't a regular function — that's the whole point of sub-graphs.

---

## 6. Running It End-to-End

The full run is interactive — the graph **stops twice** at `human_feedback`:

```python
thread = {"configurable": {"thread_id": "1"}}

# Stream until the first interrupt — analysts are now in state.
for event in graph.stream(
    {"topic": "The benefits of adopting LangGraph as an agent framework",
     "max_analysts": 3},
    thread, stream_mode="values"
):
    for a in event.get("analysts", []):
        print(a.name, "—", a.role)
```

Now you've seen the proposed team. Push a revision:

```python
graph.update_state(thread, {"human_analyst_feedback":
    "Add in the CEO of a gen-ai-native startup"})
```

And stream again — the conditional routes back to `create_analysts`, generates a new team, and stops at the interrupt a second time. When the team looks good:

```python
graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--", next(iter(event.keys())))
```

Output:

```
--Node-- conduct_interview
--Node-- conduct_interview
--Node-- conduct_interview      # N parallel interviews — note three lines in a row
--Node-- write_conclusion
--Node-- write_introduction     # three parallel writers
--Node-- write_report
--Node-- finalize_report
```

Then read the final report:

```python
final_state = graph.get_state(thread)
print(final_state.values["final_report"])
```

---

## 7. Built-in Diagnostics — The "Test Codes" Section

The notebook ends with three diagnostic cells. **They are not part of the graph** — they're standalone probes you run when something stops working. Treat them as part of the lesson, not as throwaway scratch code: retrieval is the single most common failure point in this kind of graph, and these probes are *the* fastest way to isolate where the failure is.

The pattern across all three is the same: **start at the lowest layer (raw HTTP or env vars), climb up to the LangChain wrapper, and finish by simulating exactly what the graph does**. If layer N passes and layer N+1 fails, you know the bug is in N+1. That's strictly more useful than re-running the graph and seeing "empty sources".

### 7a. Wikipedia probe

Four tests, each one layer higher than the last:

```python
QUERY = "LangGraph"

# Test 1: raw HTTP — does Wikipedia respond at all?
import requests
r = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={"action": "query", "list": "search", "srsearch": QUERY,
            "format": "json", "srlimit": 2},
    headers={"User-Agent": "test-script/1.0 (test@example.com)"},
    timeout=10,
)
# checks r.status_code == 200 and r.content is non-empty

# Test 2: the `wikipedia` package directly
import wikipedia
results = wikipedia.search(QUERY, results=2)
page = wikipedia.page(results[0], auto_suggest=False)
# checks page.title and len(page.content)

# Test 3: WikipediaAPIWrapper.run() (the LangChain utility)
from langchain_community.utilities import WikipediaAPIWrapper
out = WikipediaAPIWrapper(top_k_results=2).run(QUERY)

# Test 4: WikipediaLoader (what the graph actually uses)
from langchain_community.document_loaders import WikipediaLoader
docs = WikipediaLoader(query=QUERY, load_max_docs=2).load()
# inspects len(docs), docs[0].metadata, docs[0].page_content
```

**What each test tells you when it fails:**

| Test that fails | Most likely cause |
|---|---|
| Test 1 (raw HTTP) | Network/DNS, firewall, blocked egress, or wrong User-Agent header |
| Test 2 (`wikipedia` package) | Package install corrupted, or rate-limited despite raw HTTP working |
| Test 3 (`WikipediaAPIWrapper`) | LangChain version mismatch with the underlying `wikipedia` lib |
| Test 4 (`WikipediaLoader`) | Loader-specific bug, or your query is degenerate (empty / too long) |

If Test 1 passes but Test 4 fails, the network is fine — the bug is somewhere in the LangChain stack, and you don't waste time debugging connectivity.

### 7b. Tavily probe

Same shape, four tests, starts one layer lower because Tavily needs a key:

```python
QUERY = "benefits of LangGraph agent framework"

# Test 0: API key check
import os
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("TAVILY_API_KEY")
# fails fast with a clear error if the key isn't loaded

# Test 1: raw HTTP to api.tavily.com/search
import requests
r = requests.post(
    "https://api.tavily.com/search",
    json={"api_key": key, "query": QUERY, "max_results": 2},
    timeout=15,
)
# checks r.status_code == 200 and parses r.json()["results"]

# Test 2: langchain_tavily.TavilySearch.invoke()
from langchain_tavily import TavilySearch
data = TavilySearch(max_results=2).invoke({"query": QUERY})
# checks data["results"] and the keys of results[0]

# Test 3: simulate the graph's formatting logic
formatted = "\n\n---\n\n".join(
    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
    for doc in data["results"]
)
```

**What each test tells you when it fails:**

| Test that fails | Most likely cause |
|---|---|
| Test 0 (key check) | `.env` not loaded, wrong filename, or key was never set |
| Test 1 (raw HTTP) | Key is invalid/expired, quota exhausted, or Tavily is down |
| Test 2 (`TavilySearch`) | `langchain_tavily` version mismatch, or the wrapper changed shape |
| Test 3 (formatting) | Response shape drifted from what the graph code assumes (`url`, `content` keys missing) |

Test 3 is the one that catches the most insidious bug: Tavily can return a 200 with a valid payload that no longer matches the dict keys your `search_web` node expects, and the graph then writes empty context without raising. The formatting simulator exposes that immediately.

### 7c. Structured-output probe

A single one-cell sanity check that the LLM-side of the graph is wired correctly:

```python
from langchain_core.messages import SystemMessage, HumanMessage

test_llm = llm.with_structured_output(SearchQuery)
result = test_llm.invoke([
    SystemMessage(content="Generate a search query about LangGraph."),
    HumanMessage(content="What are the benefits of LangGraph?"),
])
# expected: result is a SearchQuery instance, result.search_query is a non-empty string
```

This isolates a class of bugs that has nothing to do with the network: the model not honoring `with_structured_output`. Failure modes you'll see here are usually:

- The model returns `None` (structured-output coercion silently failed → the graph's `_build_query` falls back to the raw question).
- The model returns the right type but with an empty `search_query` (prompt is too vague or the model is too small).
- A `ValidationError` from Pydantic, which tells you exactly which field the model got wrong.

If this cell passes and the graph still produces empty searches, you've ruled out the model and can focus on the surrounding logic.

### 7d. How to use these probes

A practical debugging order when a research run produces empty sources or weak answers:

1. **Run the structured-output probe (7c)** — fastest, no network. Eliminates the LLM-coercion layer.
2. **Run the Wikipedia probe (7a)** Tests 1 → 4 — walks up the stack, points at the exact layer that breaks.
3. **Run the Tavily probe (7b)** Tests 0 → 3 — same idea, plus a key-presence check up front.
4. **Only then** re-run the graph. By now you know whether the bug is network, key, library, wrapper, or graph logic.

The lesson generalizes beyond this notebook: **for any graph that depends on an external retrieval or API call, ship a vertical probe alongside it**. The cost of writing one is ~30 lines; the cost of *not* having one is hours of guessing why your agent is giving bad answers.

---

## 8. Studio Deployment

`studio/research_assistant.py` is the same graph repackaged for `langgraph dev`:

- `ChatMistralAI` instead of `ChatOpenRouter`, so it shares a single `MISTRAL_API_KEY` with the other studio graphs in this folder.
- No `MemorySaver()` — studio supplies its own checkpointer.
- `interrupt_before=['human_feedback']` is preserved, so the human-in-the-loop step still works in the studio UI.
- `langgraph.json` exposes all four module-4 graphs side by side: `parallelization`, `sub_graphs`, `map_reduce`, `research_assistant`.

To run it locally:

```bash
cd module-4/04-research-assistant/studio
cp .env.example .env        # fill in real keys
langgraph dev
```

---

## Key Takeaways

1. **This is a compositional graph.** Every piece — parallel retrieval, sub-graph, `Send()` fan-out, parallel writers — comes from an earlier lesson. Nothing in this notebook is a new primitive; the lesson is *how the primitives combine*.
2. **Human-in-the-loop is the planning gate.** Interrupting before `human_feedback` and routing back to `create_analysts` is exactly the module-3 pattern, used here to refine *who* researches before *what* is researched.
3. **Sub-graphs give each analyst private state.** `InterviewState` carries `analyst`, `context`, `interview` — none of which the outer graph cares about. The only key that crosses the boundary is `sections`.
4. **`Send()` is how N parallel interviews share one downstream node.** Each `Send` carries a different `Analyst` payload; `operator.add` on `sections` merges every interview's output back into one list.
5. **Three reducers, three reasons.** `messages` (built into `MessagesState`) for the conversation transcript; `context` for parallel retrieval; `sections` for `Send()`-spawned interviews. Strip any one of them and the graph errors with `INVALID_CONCURRENT_GRAPH_UPDATE`.
6. **Fan-in barrier on the writers.** `add_edge([...], "finalize_report")` makes the final stitch wait for intro, body, and conclusion to all finish — even though those three never write the same key.
7. **Build diagnostic cells next to retrieval.** The Wikipedia and Tavily probes at the bottom of the notebook are the right template: when a research-style graph fails, you want to know in seconds whether it's the network, the library, or your code.
