# Choosing the Right Model for Tool Calling in LangGraph

When building agents with tool calling (especially with Trustcall and structured memory schemas), not all models behave the same way. This file documents the key differences observed during development of the `task_mAIstro` memory agent, and gives you a repeatable process for evaluating any new model.

---

## The Three Dimensions of Compatibility

Model compatibility for tool calling splits into three independent concerns. A model can pass one and fail another.

```
┌──────────────────────────────────────────────────────────────────┐
│  Dimension 1: Schema Validation (before any generation)          │
│  → Does the provider accept your tool schema without a 400?      │
├──────────────────────────────────────────────────────────────────┤
│  Dimension 2: Generation Quality (behavioral)                    │
│  → Does the model call the right tool with the right structure?  │
├──────────────────────────────────────────────────────────────────┤
│  Dimension 3: Instruction Following (behavioral)                 │
│  → Does the model follow explicit rules in the system prompt?    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dimension 1 — Schema Validation (Provider-level Strictness)

This happens before the model even generates a response. The provider validates your tool schema JSON against its own rules, and rejects requests that don't comply.

| Requirement | Groq | OpenAI | OpenRouter |
|---|---|---|---|
| Tool `description` must not be `null` | **Strict** (400 error) | Lenient | Varies by model |
| `null` value for a `list` field | **Strict** (400 error) | Lenient | Varies by model |
| `TypedDict`-based tool schemas | **Fails** | Works | Varies |
| `BaseModel`-based tool schemas | Works | Works | Works |
| Missing `Field(description=...)` | May cause issues | Fine | Varies |

### Error signatures for Dimension 1 failures

```
# Missing docstring on a Pydantic model used as a tool:
BadRequestError: 400 - 'tools.0.function.description': Value is not nullable

# Bare list[str] field when model outputs null:
BadRequestError: 400 - /interests: expected array, but got null

# TypedDict used as a tool schema:
BadRequestError: 400 - Failed to call a function.
failed_generation: '<function=UpdateMemory {"update_type": "todo"}</function>'
```

### Fix: Write Groq-compatible schemas from the start

Apply these rules to any Pydantic model placed in `tools=[...]`:

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ToDo(BaseModel):
    # 1. Class docstring is mandatory — becomes the tool's description field in the API request
    """A task that the user wants to accomplish in the future"""

    # 2. Every field gets Field(description=...) — becomes the field description in the schema
    task: str = Field(description="The task to be completed.")

    # 3. Optional lists use Optional[list[...]] with default=None, NOT list[...] with default_factory=list
    #    Reason: bare list[str] generates {"type": "array"} which rejects null.
    #            Optional[list[str]] generates {"anyOf": [{"type": "array"}, {"type": "null"}]} which accepts null.
    solutions: Optional[list[str]] = Field(description="Specific actionable options.", default=None)

    # 4. Use BaseModel, NOT TypedDict, for any class that goes into tools=[]
    #    TypedDict produces a minimal schema that Groq llama cannot format into a proper tool call.
```

---

## Dimension 2 — Generation Quality (Behavioral)

This is about what the model *generates*, not what the provider validates. These issues show up as wrong outputs, not errors.

| Behavior | Groq llama-3.3-70b | OpenAI GPT-4o | Impact |
|---|---|---|---|
| Insert new item vs PatchDoc existing | Tends to PatchDoc (wrong) | Correct | Multiple tasks collapse into one store entry |
| Parallel tool calls | Unreliable | Reliable | Multi-step updates may drop some calls |
| Respects `Literal` enum values | Usually | Always | Router may output an invalid update_type |
| Follows custom insertion rules in prompt | Only with explicit rules | Follows intent | Requires extra prompt engineering |

### The insert vs PatchDoc problem (Groq-specific)

When you have an existing item in the store and send a message about a *different* new task, Groq's llama model tends to call `PatchDoc` on the existing item instead of calling your schema tool to create a new entry. This collapses multiple distinct tasks into one store key.

**Why it happens:** The model sees `PatchDoc` in its available tools (added by Trustcall) and prefers a low-cost "update" over generating a full new document, even when the task is semantically new.

**Fix — explicit rules in `TRUSTCALL_INSTRUCTION`:**

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

This does not change anything for OpenAI — GPT-4o follows the schema intent correctly without these rules.

---

## Dimension 3 — Instruction Following

Larger, more capable models follow system prompt instructions more reliably. This affects:
- Whether the agent updates the right memory type
- Whether it creates new entries vs patching old ones
- Whether it stays within the `Literal` enum values

**General observation:** GPT-4o follows intent. Groq llama follows instructions literally — which means it needs those instructions to be explicit and unambiguous.

---

## Quick Evaluation — 3 Probes for Any New Model

Before committing to a model, run these three probes in order:

### Probe 1 — Schema validation (schema level)

```python
model.bind_tools([YourSchema]).invoke([HumanMessage(content="hello")])
# Pass: returns an AIMessage (possibly with a tool call or plain text)
# Fail: BadRequestError 400 → fix docstrings and Optional[list[str]] fields first
```

### Probe 2 — Insert vs PatchDoc (behavioral)

```python
# Set up the extractor with Spy, put one item in existing_memories,
# then ask about a DIFFERENT new task:
spy = Spy()
extractor = create_extractor(model, tools=[ToDo], tool_choice="ToDo", enable_inserts=True)
extractor = extractor.with_listeners(on_end=spy)

result = extractor.invoke({
    "messages": [HumanMessage(content="I need to fix the jammed lock on the door.")],
    "existing": [("uuid-1", "ToDo", {"task": "book swim lessons", "status": "not started", ...})]
})

print(spy.called_tools)
# Pass: see one ToDo call (new entry) — no PatchDoc for the lock
# Fail: see a PatchDoc on uuid-1 — model incorrectly "updated" the swim lessons entry
```

### Probe 3 — Literal enum respect (behavioral)

```python
result = model.bind_tools([UpdateMemory]).invoke(
    [HumanMessage(content="I need to book swim lessons for the baby.")]
)
print(result.tool_calls[0]['args'])
# Pass: {"update_type": "todo"}
# Fail: any value not in ["user", "todo", "instructions"], or no tool call at all
```

---

## Model Comparison (Observed in Practice)

| Model | Probe 1 (schema) | Probe 2 (insert) | Probe 3 (enum) | Notes |
|---|---|---|---|---|
| **OpenAI GPT-4o** | Pass | Pass | Pass | Reference implementation. No extra prompt engineering needed. |
| **Groq llama-3.3-70b-versatile** | Pass (after fixes) | Fail (without rules) | Pass | Requires `Optional[list[str]]`, docstrings, `BaseModel` for tools, and explicit TRUSTCALL_INSTRUCTION rules. |
| **Google Gemma-4-26b (via OpenRouter)** | Untested | Untested | Untested | Used as primary model in this project; fell back to Groq when it failed. |

---

## Notation: What each fix actually does

| Fix | What it changes in the schema/prompt | Why it's needed |
|---|---|---|
| Add class docstring | `"description": "..."` instead of `null` in the tool JSON | Groq rejects null description |
| `Optional[list[str]]` | `{"anyOf": [{"type":"array"},{"type":"null"}]}` instead of `{"type":"array"}` | Groq rejects null for array fields |
| `BaseModel` over `TypedDict` | Full JSON schema with `type`, `enum`, `description` per field | llama needs rich schema to format tool call correctly |
| `TRUSTCALL_INSTRUCTION` insert rules | Explicit guidance to the LLM about when to insert vs patch | llama defaults to cheapest operation (PatchDoc) without guidance |
| `enable_inserts=True` on extractor | Trustcall registers your schema tool for new entries in addition to PatchDoc | Without this flag, only updates (PatchDoc) are possible — insertions are silently disabled |

---

## Quick Decision Checklist

When you switch to a new model and things break, go through this order:

```
1. Getting a 400 before any output?
   → Check: docstring on model? Optional[list[str]]? BaseModel not TypedDict?

2. Getting a 400 with failed_generation in the error?
   → TypedDict tool schema. Convert to BaseModel.

3. Graph runs but wrong number of store entries?
   → Insert vs PatchDoc issue. Add explicit rules to TRUSTCALL_INSTRUCTION.

4. Graph runs but agent picks wrong memory type?
   → Probe 3 failing. Model not following Literal enum. Try a more capable model
      or add a stricter example in the system prompt.

5. Everything passes but results are inconsistent run to run?
   → Set temperature=0 on the model. If still inconsistent, the model is too
      small for reliable structured output.
```
