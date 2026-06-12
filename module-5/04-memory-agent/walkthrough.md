# Walkthrough: Memory Agent — task_mAIstro

> **Notebook:** `module-05/1-memory_agent.ipynb`
> **Goal:** Build a ReAct agent that decides *when* to save memories and *what type* to save — profile, ToDo collection, or instructions.

---

## The Big Picture

All previous chatbots had two hard limits:
1. They **always** saved memory after every message (no decision-making).
2. They **always** saved one type of memory (either a profile or a collection).

`task_mAIstro` removes both limits. It is a **ReAct agent** that:
- Decides *whether* memory needs updating at all
- Chooses *which* of three memory types to update: `profile`, `todo`, or `instructions`
- Uses Trustcall under the hood to perform the actual update surgically

---

## 1. The Three Memory Types

```
┌──────────────┬────────────────────────────────────────┬─────────────────┐
│ Type         │ What it stores                         │ Schema          │
├──────────────┼────────────────────────────────────────┼─────────────────┤
│ profile      │ User facts (name, location, family)    │ Profile         │
│ todo         │ Collection of tasks (each a UUID key)  │ ToDo            │
│ instructions │ Agent's own rules for handling tasks   │ free-form text  │
└──────────────┴────────────────────────────────────────┴─────────────────┘
```

---

## 2. Pydantic Schemas

### Profile — one document per user (always updated in place)

```python
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: Optional[list[str]] = Field(
        description="Personal connections of the user",
        default=None
    )
    interests: Optional[list[str]] = Field(
        description="Interests that the user has",
        default=None
    )
```

### ToDo — one document per task (UUID key, can insert many)

```python
class ToDo(BaseModel):
    """A task that the user wants to accomplish in the future"""
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated minutes to complete.")
    deadline: Optional[datetime] = Field(description="When the task is due.", default=None)
    solutions: Optional[list[str]] = Field(
        description="Specific, actionable options to complete the task.",
        default=None
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )
```

> **Why `Optional[list[str]]` instead of `list[str]`?** See the [Groq Error section](#groq-error-2-null-rejected-for-list-fields) below.

---

## 3. The Router Tool — `UpdateMemory`

`task_mAIstro` only exposes one tool to the LLM: `UpdateMemory`. It is a simple routing signal — the LLM calls it to declare *which* memory type to update, then LangGraph routes to the appropriate node.

```python
class UpdateMemory(BaseModel):
    """Decision on what memory type to update"""
    update_type: Literal['user', 'todo', 'instructions']
```

> **Why `BaseModel` and not `TypedDict`?** See the [Groq Error section](#groq-error-3-typeddict-tool-schema-fails) below.

---

## 4. Visibility into Trustcall — the `Spy` Class

When Trustcall runs, it calls either:
- Your **schema tool** (e.g. `ToDo`) — to create a new entry
- **`PatchDoc`** — to surgically update an existing entry via JSON Patch ops

By default, you can't see which one was called. The `Spy` listener fixes that.

```python
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )
```

**How it works:** `Spy` is a listener added via `.with_listeners(on_end=spy)`. After each extractor run it walks the run tree, finds every `chat_model` step, and collects the raw tool calls. This gives you visibility into both `ToDo` calls (new entries) and `PatchDoc` calls (patches to existing ones).

### `extract_tool_info` — parsing the spy output

```python
def extract_tool_info(tool_calls, schema_name="Memory"):
    changes = []
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({'type': 'new', 'value': call['args']})
    ...
```

This is passed back to `task_mAIstro` as the tool response message, so the agent sees a human-readable summary of what changed.

---

## 5. Graph Architecture

```
START
  │
  ▼
task_mAIstro ──[no tool call]──────────────────► END
  │
  ├──[update_type == "user"]────► update_profile ──────┐
  ├──[update_type == "todo"]────► update_todos ─────────┤
  └──[update_type == "instructions"]► update_instructions┘
                                           │
                                    (all loop back)
                                           │
                                           ▼
                                     task_mAIstro
```

The graph has two persistence layers:
- **`MemorySaver` (checkpointer)** — short-term, within-thread message history
- **`InMemoryStore`** — long-term, across-thread memory (profile / todo / instructions)

```python
graph = builder.compile(
    checkpointer=within_thread_memory,
    store=across_thread_memory
)
```

---

## 6. Key Nodes

### `task_mAIstro` — the decision node

Loads all three memory namespaces from the store, injects them into the system prompt, then calls the model with `UpdateMemory` as the only available tool.

```python
response = model.bind_tools(
    [UpdateMemory],
    parallel_tool_calls=False
).invoke([SystemMessage(content=system_msg)] + state["messages"])
```

`parallel_tool_calls=False` forces the model to make one routing decision at a time, keeping the flow predictable.

### `update_todos` — the Trustcall node for ToDos

Creates a fresh `Spy`, builds a `todo_extractor` with `enable_inserts=True` (critical for inserting new tasks), runs Trustcall over the conversation, then saves each result to the store under its own UUID key.

```python
todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice="ToDo",
    enable_inserts=True
).with_listeners(on_end=spy)
```

---

## 7. The `TRUSTCALL_INSTRUCTION` — Insert vs Update Rules

Left to its own devices, Groq's llama model tends to call `PatchDoc` on an existing entry instead of inserting a new one. This collapses multiple distinct tasks into one store key.

The fix is explicit rules in the instruction:

```python
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

IMPORTANT RULES FOR ToDo ITEMS:
- If the user mentions a task that does NOT exist in the current list, ALWAYS create a NEW ToDo entry using the ToDo tool (do NOT patch an existing task).
- Only use PatchDoc to update fields (status, deadline, solutions) of a task that ALREADY EXISTS in the list by name.
- Each distinct real-world task must be its own separate ToDo entry.

System Time: {time}"""
```

---

## 8. End-to-End Flow Example

```
User: "My wife asked me to book swim lessons for the baby."

1. task_mAIstro reads store (profile, todo, instructions)
2. Calls UpdateMemory(update_type="todo")
3. route_message → update_todos
4. update_todos: Trustcall calls ToDo tool (new entry, UUID key)
5. Store: {"task": "book swim lessons for the baby", "status": "not started", ...}
6. task_mAIstro replies: "I've added it to your ToDo list."

---

User: "For the swim lessons, I need to get that done by end of November."

1. task_mAIstro reads store
2. Calls UpdateMemory(update_type="todo")
3. route_message → update_todos
4. update_todos: Trustcall calls PatchDoc (existing UUID, patches deadline field only)
5. Store: same UUID, deadline updated, everything else unchanged

---

New thread, same user_id:

User: "I have 30 minutes, what tasks can I get done?"
→ Agent reads across_thread_memory, sees all tasks including done ones
→ Recommends tasks with time_to_complete ≤ 30 and status != done
```

---

## Groq Compatibility Errors Encountered

This notebook was adapted from the original (which uses OpenAI GPT-4o) to run on **Groq (llama-3.3-70b-versatile)**. Groq is significantly stricter in its tool-call validation, which caused three distinct errors.

---

### Groq Error 1: Missing Docstring on Pydantic Models

**Error:**
```
BadRequestError: 400 - 'tools.0.function.description': Value is not nullable
```

**Root cause:** When LangChain converts a Pydantic `BaseModel` into a tool schema for the API, the class docstring becomes the `description` field in the JSON. OpenAI accepts `null` for `description`; Groq rejects it with a `400`.

**Fix:** Add a docstring to every `BaseModel` class that is passed in `tools=[...]`:

```python
# BEFORE (fails on Groq):
class Memory(BaseModel):
    content: str = Field(description="...")

# AFTER (works on Groq):
class Memory(BaseModel):
    """A single memory about the user, extracted from the conversation."""
    content: str = Field(description="...")
```

**Rule:** Every Pydantic model in `tools=[...]` needs both a class docstring (tool description) and `Field(description=...)` on each field (field description). Classes that are never passed to an LLM API don't need this.

---

### Groq Error 2: `null` Rejected for List Fields

**Error:**
```
BadRequestError: 400 - /interests: expected array, but got null
```

**Root cause:** When the LLM has no data for an optional list field (e.g. `interests`), it outputs `null`. A bare `list[str]` field in Pydantic produces a JSON schema that says the field must be an array — Groq's validator rejects `null`. OpenAI is lenient and allows it anyway.

**Fix:** Wrap all optional list fields in `Optional[...]` and set `default=None`:

```python
# BEFORE (fails on Groq):
connections: list[str] = Field(..., default_factory=list)
interests: list[str] = Field(..., default_factory=list)
solutions: list[str] = Field(..., min_items=1, default_factory=list)

# AFTER (works on Groq):
connections: Optional[list[str]] = Field(description="...", default=None)
interests: Optional[list[str]] = Field(description="...", default=None)
solutions: Optional[list[str]] = Field(description="...", default=None)
# Note: min_items=1 is invalid in Pydantic v2 — remove it entirely
```

`Optional[list[str]]` produces a JSON schema of `{"anyOf": [{"type": "array"}, {"type": "null"}]}` which Groq accepts.

---

### Groq Error 3: `TypedDict` Tool Schema Fails

**Error:**
```
BadRequestError: 400 - Failed to call a function.
failed_generation: '<function=UpdateMemory {"update_type": "todo"}</function>'
```

**Root cause:** The original `UpdateMemory` was defined as a `TypedDict`. `TypedDict` produces a minimal JSON schema that Groq's llama model cannot format into a proper tool call — it falls back to an XML-style format (`<function=...>`) which Groq then rejects.

**Fix:** Convert `UpdateMemory` to a Pydantic `BaseModel`:

```python
# BEFORE (fails on Groq):
from typing import TypedDict
class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""
    update_type: Literal['user', 'todo', 'instructions']

# AFTER (works on Groq):
from pydantic import BaseModel
class UpdateMemory(BaseModel):
    """Decision on what memory type to update"""
    update_type: Literal['user', 'todo', 'instructions']
```

Pydantic `BaseModel` generates a richer JSON schema with explicit `type`, `enum`, and `description` fields that Groq requires to format tool calls correctly.

---

### Groq Behavioral Issue: Single Task in Store

**Symptom:** After sending two separate task messages, only one task appeared in the store.

**Root cause:** Groq's llama model tends to call `PatchDoc` on an existing entry instead of calling the schema tool to insert a new one. When "fix the jammed lock" arrived while "swim lessons" was already in the store, the model patched the swim lessons entry instead of creating a new UUID key.

**Fix:** Add explicit insert-vs-update rules to `TRUSTCALL_INSTRUCTION` (see [Section 7](#7-the-trustcall_instruction--insert-vs-update-rules) above). This behavior does not occur with OpenAI GPT-4o, which follows the schema intent correctly without explicit guidance.

---

## Key Takeaways

1. `task_mAIstro` is a ReAct agent — it decides whether and what type of memory to update.
2. Three memory namespaces: `profile` (one doc, updated in place), `todo` (collection, UUID keys), `instructions` (free-form text).
3. `UpdateMemory` is a router — a simple tool with one `Literal` field that LangGraph uses to pick the next node.
4. The `Spy` listener gives you visibility into whether Trustcall used your schema tool (new entry) or `PatchDoc` (patch to existing entry).
5. `enable_inserts=True` on the Trustcall extractor is required for the ToDo collection — without it, only updates are possible.
6. Groq requires: docstrings on all tool models, `Optional[list[str]]` for nullable lists, and `BaseModel` (not `TypedDict`) for router tools.
