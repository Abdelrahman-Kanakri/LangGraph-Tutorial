# Walkthrough: Double Texting

> **Notebook:** `double_texting.ipynb`
> **Goal:** Handle the real-world scenario where a user sends a new message before the previous run on a thread has finished.

---

## The Big Picture

In a chat application, a user can send a second message before the first one is fully processed. If you just start a second run on the same thread while the first is still running, you get a conflict — both runs are reading and writing to the same thread state at the same time.

LangGraph Server solves this with four **multitask strategies**. You pick one per run by setting `multitask_strategy` when calling `client.runs.create`.

---

## Strategy 1: Reject

**Behaviour:** If a run is already active on the thread, refuse to start the new one and raise an error.

```python
try:
    await client.runs.create(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content=user_input_2)]},
        config=config,
        multitask_strategy="reject",
    )
except httpx.HTTPStatusError as e:
    print("Failed to start concurrent run", e)
```

The server responds with an HTTP error. The original run continues unaffected. The second message is simply dropped.

**Use when:** correctness matters more than convenience — you never want two conflicting writes to the same thread state. You handle the rejection in the client (e.g., show "please wait…" to the user).

---

## Strategy 2: Enqueue

**Behaviour:** Queue the new run and start it automatically after the current run finishes.

```python
first_run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input_1)]})
second_run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input_2)]})

await client.runs.join(thread["thread_id"], second_run["run_id"])
```

Both messages get processed in order. The thread state shows both conversations sequentially, as if the user waited.

**Use when:** you want to guarantee every message is handled, and ordering matters. Common for task-management or agentic workflows where no input should be silently discarded.

---

## Strategy 3: Interrupt

**Behaviour:** Stop the current run at the next safe checkpoint, save everything done so far, then start the new run from there.

```python
interrupted_run = await client.runs.create(
    thread["thread_id"], graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]},
    config=config,
)

await asyncio.sleep(1)  # let the first run start

second_run = await client.runs.create(
    thread["thread_id"], graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]},
    config=config,
    multitask_strategy="interrupt",
)

await client.runs.join(thread["thread_id"], second_run["run_id"])
```

After this, the interrupted run has `"status": "error"` (it was cut off). The thread state shows the partial work from run 1 plus the full result from run 2.

**Use when:** the second message supersedes the first (e.g., "never mind, do X instead") but you still want to keep whatever the first run had already written to the thread.

---

## Strategy 4: Rollback

**Behaviour:** Stop the current run, **delete it entirely** (including any state changes it made), then start fresh with the new input.

```python
rolled_back_run = await client.runs.create(
    thread["thread_id"], graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]},
    config=config,
)

second_run = await client.runs.create(
    thread["thread_id"], graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]},
    config=config,
    multitask_strategy="rollback",
)

await client.runs.join(thread["thread_id"], second_run["run_id"])

# Confirm the first run was deleted
try:
    await client.runs.get(thread["thread_id"], rolled_back_run["run_id"])
except httpx.HTTPStatusError:
    print("Original run was correctly deleted")
```

The thread state contains only the second run's result — as if the first message never existed.

**Use when:** the second message completely replaces the first. You don't want any trace of the discarded intent in the thread history.

---

## Comparison Table

| Strategy | What happens to run 1 | What happens to run 2 | Thread state |
|----------|-----------------------|----------------------|--------------|
| **reject** | Continues normally | Raises error — never starts | Only run 1 |
| **enqueue** | Runs to completion | Starts after run 1 finishes | Both, in order |
| **interrupt** | Stopped at next checkpoint, status `error` | Starts immediately | Partial run 1 + full run 2 |
| **rollback** | Stopped and deleted | Starts fresh | Only run 2 |

---

## What to Understand Here

1. **Double texting is a concurrency problem** — two runs on the same thread would write to the same state simultaneously. The strategy decides who wins.
2. **Default strategy is `enqueue`** — if you don't set `multitask_strategy`, the server queues the second run automatically. That's why basic apps "just work" without you thinking about it.
3. **Interrupt vs. rollback** — the key difference is whether partial work from run 1 stays in the thread. `interrupt` keeps it; `rollback` erases it.
4. **Checkpoints enable interrupt and rollback** — the server can stop mid-run because every node writes a checkpoint. Without persistence, neither strategy would be possible.
