# Walkthrough: Chatbot with a Profile Schema

> **Notebook:** `module-05/02-memory-schema-profile/memoryschema_profile.ipynb`
> **Goal:** Upgrade the long-term-memory chatbot so memories are saved as a *structured* user profile (a single, continuously-updated schema) instead of a free-form string — and use **Trustcall** to update that schema reliably and efficiently.

---

## The Big Picture

In module 01 the chatbot stored memory as a plain string and **regenerated it from scratch** on every turn. That's wasteful (re-emits the whole profile each time) and lossy (it can drop facts it forgot to copy).

This notebook fixes both by giving memory a **schema** and updating it with **JSON Patch** instead of full rewrites:

```
┌──────────────────────────────────────────────────────────────┐
│  module 01:  memory = free-form string, rewritten every turn  │
│              ▶ wasteful + can lose information                │
├──────────────────────────────────────────────────────────────┤
│  module 02:  memory = structured UserProfile (a schema)       │
│              updated via Trustcall JSON Patch (only deltas)   │
│              ▶ efficient + preserves existing fields          │
└──────────────────────────────────────────────────────────────┘
```

Three building blocks, in order:
1. Define a **profile schema** and save it to the store.
2. Use **`with_structured_output`** to extract a profile — and see where it breaks.
3. Bring in **Trustcall** to extract *and update* the profile robustly inside a chatbot graph.

---

## Namespace structure (how the store is organized)

The store is a key-value store organized like a **filesystem**:

| Store concept | Filesystem analogy |
|---|---|
| `namespace` (a tuple) | the folder path |
| `key` | the filename |
| `value` (a dict) | the file contents |

The namespace tuple is read **left → right = outermost → innermost folder**:

```
(user_id, "memory")   →   1/memory/        root = 1,       sub = memory
("memory", user_id)   →   memory/1/        root = memory,  sub = 1
```

So **swapping the elements flips the hierarchy**:

- `(user_id, ...)` first → top-level folders are *users*; everything about user 1 lives together under `1/`. Good when you usually ask *"give me everything for this user."*
- `("memory", ...)` first → top-level folder is the *category*; all users' memories live together under `memory/`. Good when you usually ask *"give me all memories across users."*

⚠️ **The store matches the tuple literally.** Whatever tuple you `put` with, you must `get`/`search` with the **exact same tuple, in the same order** — otherwise you're looking in a different folder and get nothing back. Pick one convention and stay consistent within a graph.

> Note: this course uses both orderings across notebooks (`("memory", user_id)` in the graph cells, `(user_id, "user_memory")` in the intro `put` cell). Both are valid — just different folder layouts. What matters is internal consistency.

---

## 1. Defining a User Profile Schema

A schema is just a typed shape for the memory. The notebook shows two ways to declare it:

**TypedDict** (lightweight, dict-like):
```python
from typing import TypedDict, List

class UserProfile(TypedDict):
    """User profile schema with typed fields."""
    user_name: str          # The user's preferred name
    interests: List[str]    # A list of the user's interests
```

**Pydantic `BaseModel`** (richer — field descriptions, validation):
```python
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """User profile schema with typed fields"""
    user_name: str = Field(description="The user's preferred name")
    interests: List[str] = Field(description="A list of the user's interests")
```

The store accepts any plain dict as `value`, so a schema instance is saved the same way as before:

```python
namespace_for_memory = (user_id, "user_memory")
memory_store.put(namespace_for_memory, "user_profile", user_profile)
```

> **Why Pydantic matters for tool calling:** the class docstring becomes the tool's `description`, and `Field(description=...)` becomes each parameter's description. On Groq these are **required** (see Groq Error 1 below).

---

## 2. Creating Memories with `with_structured_output`

To turn a conversation into a profile, bind the schema to the model:

```python
model = ChatGroq(model="qwen/qwen3-32b", temperature=0)
model_with_structure = model.with_structured_output(UserProfile)

model_with_structure.invoke([HumanMessage(content="My name is Abood, and I like to bike.")])
# → {'user_name': 'Abood', 'interests': ['biking']}
```

This is then dropped into the chatbot's `write_memory` node: instead of saving a string, it saves a schema-conforming dict.

### When it fails — complex schemas

The notebook then deliberately tries a **deeply nested** schema (`TelegramAndTrustFallPreferences`, 6 levels deep) and shows that naive `with_structured_output` is fragile with it — even strong models struggle. This is the motivation for Trustcall.

---

## 3. Trustcall — Extract and Update Robustly

[Trustcall](https://github.com/hinthornw/trustcall) wraps the model and uses tool calling to produce structured output, with two superpowers over plain `with_structured_output`:

**Extraction:**
```python
from trustcall import create_extractor

trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile",   # force the model to emit this exact schema
)

result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + conversation})
result["responses"][0]   # → UserProfile(user_name='Abood', interests=['biking'])
```

`invoke` returns three things:
- `messages` — the raw `AIMessage`s containing the tool calls
- `responses` — the parsed objects matching your schema
- `response_metadata` — which response maps to which existing object (used when updating)

**Updating (the key win):** pass the current profile as `existing`, and Trustcall prompts the model to produce a **[JSON Patch](https://jsonpatch.com/)** — changing *only* the fields that differ, instead of regenerating the whole document:

```python
result = trustcall_extractor.invoke(
    {"messages": [SystemMessage(content=system_msg)] + updated_conversation},
    {"existing": {"UserProfile": schema[0].model_dump()}}
)
# interests went from ['biking'] → ['biking', 'visiting bakeries'] — nothing else regenerated
```

This is cheaper (fewer tokens) and safer (won't silently drop existing facts).

---

## 4. The Chatbot Graph with Trustcall

Same two-node shape as module 01:

```
START → call_model → write_memory → END
```

- `call_model` loads the profile from the store, formats it into the system prompt, and replies.
- `write_memory` calls the **Trustcall extractor** with the existing profile as `existing`, gets the updated profile, and `put`s it back.

```python
def write_memory(state, config, store):
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Pass the existing profile so Trustcall PATCHES instead of rewriting
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"],
         "existing": existing_profile}
    )

    updated_profile = result["responses"][0].model_dump()
    store.put(namespace, "user_memory", updated_profile)
```

Across threads, the profile persists: start a new `thread_id` (blank chat history) with the same `user_id`, and the bot still knows the user's name and interests from the store.

---

## 5. Model Choice — Which Groq Model for Which Cell

This was the single biggest source of friction, so each model-init cell now carries a comment explaining its choice. The rule we landed on:

| Cell | Model | Why |
|---|---|---|
| `with_structured_output`, simple schema | `qwen/qwen3-32b` | Works — the input states an interest, so every field has data to fill. |
| Trustcall extractor, simple schema | `qwen/qwen3-32b` | Works — conversation contains an explicit interest; clean tool call. |
| Complex nested schema (`bound`) | (any) | **Demo of failure** — fragile on Groq regardless of model. |
| Live chatbot graph | `llama-3.3-70b-versatile` | First turn is often *just a name* with no interests; reasoning models refuse to emit a partial tool call under forced `tool_choice`. llama honors it reliably. |

**The principle:** for **forced tool calling / structured output**, prefer a solid *instruction* model (Llama, standard Qwen). **Reasoning models** (`gpt-oss`, `qwen3` in thinking mode) "think out loud" and tend to answer in prose or ask follow-up questions instead of emitting a clean tool call — which Groq rejects when `tool_choice` is forced.

---

## Groq Compatibility Errors Encountered

This notebook was written for OpenAI (`gpt-4o`). Swapping in `ChatGroq` surfaced several provider-strictness issues. Here's each one and its fix:

### Groq Error 1: `'tools.0.function.description' : Value is not nullable`

**Cause:** Trustcall builds a tool from the schema; the tool's `description` comes from the class **docstring**. The complex `TelegramAndTrustFallPreferences` classes had **no docstrings**, so `description` was `null`. OpenAI tolerates this; Groq 400s.

```python
# BEFORE (fails on Groq):
class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences

# AFTER (works on Groq):
class TelegramAndTrustFallPreferences(BaseModel):
    """A user's telegram, communication, and trust-fall preferences."""
    pertinent_user_preferences: UserPreferences
```

### Groq Error 2: `tool_use_failed` — "model did not call a tool"

**Cause:** A reasoning model (`gpt-oss-120b`, or `qwen3` in thinking mode) answered in **prose** (a ```json block) instead of emitting a tool call. Under forced `tool_choice`, Groq requires an actual tool call → 400.

**Fix:** use a model that reliably emits tool calls → `llama-3.3-70b-versatile`.

### Groq Error 3: `tool_use_failed` — `<function=...>{...}</function>` in `failed_generation`

**Cause:** The model produced the **correct** extraction, but wrapped it in its native `<function=NAME>{...}</function>` text format instead of the standard OpenAI `tool_calls` JSON. Groq's parser couldn't convert it. Made worse by **verbatim quoted text** (`"Daredevil"`, `I'm`) in the `sentence_preference_revealed` fields, which produced malformed/over-escaped JSON.

**Fix:** prefer flatter schemas and a tool-calling model; this complex-schema cell is best treated as a *demonstration that complex extraction is fragile* rather than something to force through on Groq.

### Groq Behavioral Issue: model asks instead of extracting

**Cause:** First chatbot turn is just `"Hi, my name is Abood"` — a name, **no interests**. Because `interests` is a **required** field, the model (especially reasoning models) replied `"What are your interests, Abood?"` instead of saving a partial profile → `tool_use_failed`.

**Fixes:** (1) use `llama-3.3-70b-versatile`; optionally (2) make the field optional so a name-only turn still produces a valid call:
```python
interests: List[str] = Field(default_factory=list, description="A list of the user's interests")
```

### Bonus trap: stale variables from out-of-order cell execution

Running cells out of order left **stale, wrong-typed variables** in the kernel — bit us twice:
- `model` is assigned in multiple cells; whichever ran *last* wins (not whichever is positionally above).
- `conversation` is a **message list** in one cell but a **string** in others → `TypeError: can only concatenate list (not "str") to list`.

**Tell:** a cell showing `In[ ]` in the margin never ran, so any variable it defines is missing or stale. Run sections top-to-bottom, and `print(type(var))` when in doubt.

---

## Key Takeaways

1. **Schema beats string**: a structured `UserProfile` is queryable, validated, and updatable field-by-field.
2. **Trustcall = JSON Patch updates**: pass `existing=` and it changes only what's different — cheaper and won't drop existing facts.
3. **Pydantic docstrings are mandatory on Groq**: the class docstring becomes the tool description, which Groq requires to be non-null.
4. **Match the model to the task**: forced tool calling → instruction models (`llama-3.3-70b-versatile`); reasoning models (`gpt-oss`, `qwen3`) are fragile under forced `tool_choice`.
5. **Complex nested schemas are fragile across providers** — the trust-fall schema is a *cautionary demo*, not a target to force through on Groq.
6. **Required fields + sparse input = the model asks instead of extracting**: make fields optional, or accept partial profiles.
7. **Watch the `In[ ]` counters**: most "impossible" errors here were stale variables from running cells out of order.
