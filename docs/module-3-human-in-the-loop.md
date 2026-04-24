# Module 3 — Human-in-the-Loop

Streaming output, pausing graphs, editing state mid-run, and traveling back to any past checkpoint — the full toolkit for approval, debugging, and human-feedback workflows.

---

## 01 · streaming-interruption

**Notebook:** [streaming-interruption.ipynb](../module-3/01-streaming-interruption/streaming-interruption.ipynb) · **Walkthrough:** [walkthrough.md](../module-3/01-streaming-interruption/walkthrough.md)

**What it teaches**
Four levels of streaming from a LangGraph graph: full state, state deltas, low-level events (including LLM tokens), and the API's high-level `messages` mode.

**Key APIs**

| Mode | What you get |
|------|-------------|
| `stream_mode="values"` | Full state dict after each node |
| `stream_mode="updates"` | Only the delta each node returned, keyed by node name |
| `graph.astream_events(..., version="v2")` | Event stream including `on_chat_model_stream` for tokens |
| `stream_mode="messages"` (SDK only) | Typed `messages/partial` vs `messages/complete` events |

- Filter token events by `event["metadata"]["langgraph_node"]` to get tokens from one specific node
- Accept `RunnableConfig` in node signatures for token streaming on Python < 3.11

**When to use**

- Log / progress UI → `updates`
- Full-state debug → `values`
- Token-by-token typing effect → `astream_events`
- Clean chat UI over the API → `messages`

---

## 02 · breakpoints

**Notebook:** [breakpoints.ipynb](../module-3/02-breakpoints/breakpoints.ipynb) · **Walkthrough:** [walkthrough.md](../module-3/02-breakpoints/walkthrough.md)

**What it teaches**
Pause the graph at chosen nodes for **human approval**, inspection, or editing. Declared at compile time — always fires.

**Key APIs**

- `builder.compile(checkpointer=memory, interrupt_before=["tools"])` — pause before
- `interrupt_after=[...]` — pause after
- `graph.get_state(thread).next` — which node would run next
- `graph.stream(None, thread, ...)` — resume from the saved checkpoint

**When to use**
Unconditional pauses: approval before any tool call, audit logging after every node, debug breakpoints at known nodes.

**Pattern**
`stream(input) → pause → inspect state → ask user → stream(None) to continue, or abort.`

**Required**
A checkpointer — without saved state there is nothing to resume from.

---

## 03 · edit-state-human-feedback

**Notebook:** [edit-state-human-feedback.ipynb](../module-3/03-edit-state-human-feedback/edit-state-human-feedback.ipynb) · **Walkthrough:** [walkthrough.md](../module-3/03-edit-state-human-feedback/walkthrough.md)

**What it teaches**
Two ways to modify state at a breakpoint: direct `update_state()` calls, or a dedicated no-op `human_feedback` node baked into the graph topology.

**Key APIs**

- `graph.update_state(thread, patch)` — applies via normal reducers; `add_messages` appends unless id matches (then overwrite)
- `graph.update_state(thread, patch, as_node="human_feedback")` — makes the graph advance from that named node after the edit
- No-op node: `def human_feedback(state): pass` — sits in the topology, anchors the interrupt

**When to use**

- Ad-hoc debugging / tests → direct `update_state`
- Production UI with visible feedback step → `human_feedback` node + `as_node=`

**Gotcha**
Without `as_node=`, an update leaves `.next` unchanged — the graph re-pauses at the same breakpoint.

---

## 04 · dynamic-breakpoints

**Notebook:** [dynamic-breakpoints.ipynb](../module-3/04-dynamic-breakpoints/dynamic-breakpoints.ipynb) · **Walkthrough:** [walkthrough.md](../module-3/04-dynamic-breakpoints/walkthrough.md)

**What it teaches**
Pause the graph from **inside a node** based on runtime conditions. Unlike compile-time breakpoints, this is conditional and carries a reason string.

**Key APIs**

- `from langgraph.errors import NodeInterrupt`
- `raise NodeInterrupt("input too long: ...")` — inside a node
- `state.tasks[0].interrupts` — read the raised reason from outside
- Fix: `graph.update_state(thread, patch)` → `graph.stream(None, thread, ...)`

**When to use**
Input validation, budget/cost guards, suspicious tool outputs, any "pause only if" condition.

**Critical behavior**
Resuming with `None` alone **re-raises the same interrupt** — the condition is still true. You must change state first to clear the guard.

---

## 05 · time-travel

**Notebook:** [time-travel.ipynb](../module-3/05-time-travel/time-travel.ipynb) · **Walkthrough:** [walkthrough.md](../module-3/05-time-travel/walkthrough.md)

**What it teaches**
Browse, replay, and fork past checkpoints. A run is a sequence of checkpoints (one per step); the thread's history is a stack with newest at index `[0]`.

**Key APIs**

- `graph.get_state(config)` — current checkpoint
- `graph.get_state_history(thread)` — all checkpoints, newest first
- `graph.stream(None, config)` — dispatches on `config`:
  - `thread_id` only → resume from current (newest)
  - `thread_id` + `checkpoint_id` → replay/re-run from that step
- `graph.update_state(past_config, patch)` → returns a new `config` → streaming from it **re-runs** (fork)

**Replay vs Re-run (same call, different behavior)**

| | Replay | Re-run |
|---|---|---|
| Trigger | Past `checkpoint_id` whose children already exist | `checkpoint_id` from a freshly forked branch |
| LLM / tools | Not called | Called fresh |
| Output | Identical | May differ |
| Cost | Free | Tokens + tool cost |
| History | Nothing added | New checkpoints appended |

**Message overwrite trick**
Include the original message's `id` in the patch so `add_messages` **overwrites** instead of appending.

**SDK equivalents**

- `client.threads.get_history(thread_id)`
- `client.threads.update_state(thread_id, patch, checkpoint_id=...)`
- `client.runs.stream(thread_id, assistant_id="agent", input=None, checkpoint_id=...)`

**When to use**
Debugging a bad run, A/B testing prompts, implementing undo in a chat UI, sharing reproducible `(thread_id, checkpoint_id)` pairs.
