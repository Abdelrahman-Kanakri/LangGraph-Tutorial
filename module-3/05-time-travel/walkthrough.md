# Walkthrough: Time Travel

> **Notebook:** `time-travel.ipynb`
> **Goal:** Learn how to **browse**, **replay**, and **fork** from past checkpoints — the debugging superpower LangGraph calls *time travel*.

---

## The Big Picture

Every time a node runs, the checkpointer saves a **checkpoint** — a snapshot of state plus "what runs next". Those checkpoints accumulate into a **state history** you can:

| Operation | What it does |
|-----------|-------------|
| **Browse** | List every past checkpoint for a thread |
| **Replay** | Re-run from a past checkpoint, re-emitting already-executed steps |
| **Fork** | Modify state at a past checkpoint and run forward — creating a new branch |

This unlocks powerful workflows: "rewind to before the bad tool call, change my prompt, run forward again" — without losing the original run.

---

## 1. Setup: The Arithmetic Agent (No Breakpoints)

Same agent as the previous notebooks, but this time compiled **without interrupts**:

```python
tools = [add, multiply, divide]
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=MemorySaver())
```

> **Best practice:** time travel requires a **checkpointer** — `MemorySaver` for notebooks/tests, `SqliteSaver` or `PostgresSaver` for persistence across runs.

Run the graph once so we have history to browse:

```python
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

Output: user message → tool call → tool result → final answer (`6`).

---

## 2. Browsing History

**First, the vocabulary:**

- A **run** is one invocation of the graph (one `.stream()` or `.invoke()` call).
- Each run progresses through a series of **steps** — one step per node execution.
- Each step produces a **checkpoint** — a saved snapshot of state at that point.

So: **checkpoints save steps, and a run is a sequence of checkpoints**. If your graph ran `assistant → tools → assistant`, that's 3 steps → 3 new checkpoints added to the thread's history.

**Current state** (same as what `state.next` returned earlier):

```python
graph.get_state({"configurable": {"thread_id": "1"}})
```

**Full history** — every checkpoint this thread has ever written:

```python
all_states = [s for s in graph.get_state_history(thread)]
len(all_states)
# e.g. 6
```

**Ordering — history is a stack (newest on top):**

```
index     checkpoint
─────     ──────────
  [0]     ← current (newest) state
  [1]     ← 1 step older
  [2]     ← 2 steps older
  ...
[-1]      ← oldest (just after the initial input)
```

So:
- `all_states[0]` — the **current** state (same as `graph.get_state(thread)`)
- `all_states[1]` — one step back
- `all_states[-1]` — the very first checkpoint of the thread
- `all_states[-2]` — the state right after the human message was added, just before `assistant` first ran

Each `StateSnapshot` has:
- `.values` — the state dict at that point
- `.next` — which node would run next
- `.config` — includes `thread_id` **and** `checkpoint_id` (the pointer we need for time travel)
- `.metadata` — step count, writes, etc.

---

## 3. Replaying from a Past Checkpoint

> **How `graph.stream(None, config)` picks a checkpoint**
>
> The `config` dict you pass controls **which** checkpoint is resumed from:
>
> | `config` contains | What happens when you pass `None` as input |
> |-------------------|--------------------------------------------|
> | `thread_id` only | Picks up from the **current (newest)** checkpoint of that thread |
> | `thread_id` **+** `checkpoint_id` | **Replays from that specific checkpoint** (any step of the run) |
>
> So `thread = {"configurable": {"thread_id": "1"}}` continues where you left off, while `to_replay.config` (which already includes a `checkpoint_id`) jumps back in time. Same function, completely different behavior depending on whether `checkpoint_id` is present.

**Replaying** = re-run the graph from an old checkpoint. If the checkpoint's subsequent steps have already been executed, LangGraph re-emits them from the saved state (no LLM calls, no tool calls — it's reading, not running).

Pick the checkpoint right after the human message:

```python
to_replay = all_states[-2]
to_replay.values   # {'messages': [HumanMessage('Multiply 2 and 3')]}
to_replay.next     # ('assistant',)
to_replay.config   # {'configurable': {'thread_id': '1', 'checkpoint_id': '...'}}
```

**Replay by passing the config back to `.stream()`:**

```python
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

LangGraph recognizes the `checkpoint_id` as **already executed**, so it just replays the cached transitions from that point forward. No new LLM calls happen — it's a replay, not a re-run.

---

## 3b. Replay vs Re-run / Re-execute

Same API (`graph.stream(None, config)`) — two completely different behaviors depending on whether the checkpoint already has executed children in the history.

### Replay

Resume from an old checkpoint that has **already been executed past**. The graph sees stored children of that checkpoint, so it doesn't re-invoke any nodes — it just streams back the cached states.

- No new LLM calls
- No new tool executions
- Deterministic: you get exactly what happened last time
- Free (no API cost)
- No new checkpoints created

### Re-run / Re-execute

Resume from a checkpoint that has **no executed children yet** (typically because you forked it — `update_state` created a new branch). The graph has to actually run the nodes to compute what comes next.

- Fresh LLM calls
- Fresh tool executions
- Output may differ (temperature, time-sensitive data, etc.)
- Costs tokens
- New checkpoints appended to history

### Side-by-side

| | Replay | Re-run |
|---|---|---|
| Triggered by | Passing a past `checkpoint_id` whose children exist | Passing a `checkpoint_id` from a freshly forked branch |
| LLM / tools | Not called | Called fresh |
| Output | Bit-for-bit identical | May differ |
| Cost | Free | Tokens + tool cost |
| History | Nothing added | New checkpoints appended |
| Use for | Debugging, reproducing a run | Forking with new input, retrying with a different prompt |

**The mental rule:** if the checkpoint has a "future" already stored, you replay that future. If it doesn't (because you just edited state and invalidated the old future), the graph must build a new one — that's a re-run.

---

## 4. Forking: Edit State at a Past Checkpoint

**Forking** = edit state at a past checkpoint, then run forward. Because the state is different from what was saved, LangGraph can't replay — it must **actually execute** from that point. This creates a branch off the original history.

Pick the same checkpoint:

```python
to_fork = all_states[-2]
to_fork.values["messages"]
# [HumanMessage(content='Multiply 2 and 3', id='abc-123')]
```

**The message-id trick:**

Because `add_messages` **overwrites** when you supply a matching `id` (and appends otherwise), you can swap the original human message by reusing its id:

```python
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(
        content='Multiply 5 and 3',
        id=to_fork.values["messages"][0].id,   # <— same id = overwrite
    )]},
)
```

**What you get back:**

```python
fork_config
# {'configurable': {'thread_id': '1', 'checkpoint_id': '<new-id>'}}
```

A **new checkpoint** with:
- the edited state (original message replaced)
- the same `.next` (`('assistant',)`) — the fork happens at the same point in the graph

**Run forward from the fork:**

```python
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

This time LangGraph sees an unexecuted checkpoint and actually runs — calls the LLM, executes the tool, produces a new answer (`15`). The original `6` run is still in the history; you've just created a parallel branch.

---

## Replay vs Fork — The Mental Model

```
              all_states[-2]           ── pick a past checkpoint
                    │
        ┌───────────┴────────────┐
        │                        │
      REPLAY                   FORK
        │                        │
  stream(None, cfg)        update_state(cfg, patch)
        │                        │
  re-emit saved steps       → returns NEW config
  (no LLM calls)                  │
        │                   stream(None, new_cfg)
        ▼                         │
  finishes as before         runs for real, produces
                             NEW results
```

**Rule of thumb:**
- No state change → **replay** (passing the same checkpoint config)
- State changed → **fork** (passing the new config from `update_state`)

---

## 5. Time Travel via the LangGraph SDK

The same primitives exist when the graph is deployed via `langgraph dev`. Start the server:

```bash
cd module-1/studio   # or wherever your graph lives
langgraph dev
```

Connect with the SDK:

```python
from langgraph_sdk import get_client
client = get_client(url="http://127.0.0.1:2024")
```

**Run the agent once to build history:**

```python
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=initial_input,
    stream_mode="updates",
):
    ...
```

**Browse history:**

```python
states = await client.threads.get_history(thread['thread_id'])
to_replay = states[-2]
to_replay['checkpoint_id']
```

**Replay** — pass `checkpoint_id` to the run:

```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="values",
    checkpoint_id=to_replay['checkpoint_id'],
):
    ...
```

**Fork** — edit state first, then run from the returned checkpoint:

```python
forked_input = {"messages": HumanMessage(
    content="Multiply 3 and 3",
    id=to_fork['values']['messages'][0]['id'],
)}

forked_config = await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=to_fork['checkpoint_id'],
)

async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="updates",
    checkpoint_id=forked_config['checkpoint_id'],
):
    ...
```

**API vs in-process — same concepts, slightly different names:**

| Concept | In-process | SDK |
|---------|------------|-----|
| List history | `graph.get_state_history(thread)` | `client.threads.get_history(thread_id)` |
| Get state | `graph.get_state(config)` | included in history items |
| Edit state | `graph.update_state(config, patch)` | `client.threads.update_state(thread_id, patch, checkpoint_id=...)` |
| Run from checkpoint | pass `config` with `checkpoint_id` | pass `checkpoint_id=` kwarg |

---

## Why Time Travel Matters

| Use case | How time travel helps |
|----------|----------------------|
| **Debugging a bad run** | Rewind to the last-good checkpoint, inspect, fork with fixed input |
| **A/B testing prompts** | Fork the same checkpoint with different system messages |
| **Undo** in a chat UI | Roll back to the checkpoint before the last assistant reply |
| **Reproducing bugs** | Share `(thread_id, checkpoint_id)` — anyone can replay |
| **Offline analysis** | Walk the full state history, compute metrics per step |

---

## Key Takeaways

1. **`graph.get_state_history(thread)`** returns all checkpoints for a thread, newest first
2. Each `StateSnapshot` carries `.values`, `.next`, and a `.config` containing both `thread_id` **and** `checkpoint_id`
3. **Replaying** = pass a past `config` to `.stream(None, ...)` — LangGraph re-emits saved steps without re-executing nodes
4. **Forking** = call `update_state(past_config, patch)` → get a new config → stream from it → actually runs
5. **Message id trick:** include the original message's `id` in your patch to **overwrite** instead of append (thanks to `add_messages`)
6. The SDK mirrors the same API: `threads.get_history`, `threads.update_state`, and `runs.stream(..., checkpoint_id=...)`
7. Time travel + breakpoints + `update_state` = the complete toolkit for **human-in-the-loop** agents
