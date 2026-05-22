"""Map-reduce with the `Send()` API: dynamic per-item fan-out.

Companion to ``module-4/03-map-reduce/map-reduce.ipynb``. Toy "best-joke
generator":

1. **Map** — generate N sub-topics from a topic, then spawn one parallel
   ``generate_joke`` branch per sub-topic via ``Send()``. N is decided at
   runtime by the LLM, not at compile time.
2. **Reduce** — ``best_joke`` picks the funniest joke from the merged list.

The reducer on ``jokes`` is what makes this work: each parallel branch writes
``{"jokes": [its_joke]}`` and ``operator.add`` concatenates the N single-item
lists into one final list. Strip the reducer and parallel writes hard-error.
"""

import os
import getpass
import operator
from typing import Annotated

from typing_extensions import TypedDict
from pydantic import BaseModel

from langchain_mistralai import ChatMistralAI

from langgraph.constants import Send
from langgraph.graph import END, StateGraph, START


def _set_env(var: str):
    """Prompt for an env var if it isn't already set (no-op in studio where .env supplies it)."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("MISTRAL_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""


# `temperature=0` keeps runs deterministic, which matters for the structured
# outputs below — we don't want the LLM to invent fields or vary list shapes.
model = ChatMistralAI(model="mistral-medium", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Structured output shapes (NOT state — they're response schemas for the LLM)
# ─────────────────────────────────────────────────────────────────────────────
class Subjects(BaseModel):
    """`with_structured_output(Subjects)` forces the LLM to return a `list[str]`."""
    subjects: list[str]


class BestJoke(BaseModel):
    """Forces the "best joke" pick to be an integer index, not free-form prose."""
    id: int


class Joke(BaseModel):
    """Per-branch joke shape, used inside `generate_joke`."""
    joke: str


# ─────────────────────────────────────────────────────────────────────────────
# State schemas
# ─────────────────────────────────────────────────────────────────────────────
class OverallState(TypedDict):
    """Reducer audit:
        topic              – single writer (caller).
        subjects           – single writer (`generate_topics`).
        jokes              – REDUCER REQUIRED: N parallel `generate_joke` branches.
        best_selected_joke – single writer (`best_joke`).
    """
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    """Private per-branch payload shipped by `Send()`. Doesn't have to match
    `OverallState` — `Send()` lets each branch use its own minimal schema."""
    subject: str


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def generate_topics(state: OverallState):
    """Generator step: produce the list we'll fan out over."""
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


def generate_joke(state: JokeState):
    """Map step: one joke per subject. Returns a list-of-1 so the reducer can
    concatenate contributions from every parallel branch."""
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


def best_joke(state: OverallState):
    """Reduce step: by now `state["jokes"]` is the merged list of all N jokes."""
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


def continue_to_jokes(state: OverallState):
    """Conditional edge fn that returns Send() objects instead of a node name.

    Each `Send(target, payload)` spawns one parallel `generate_joke` branch with
    its own subject. The number of branches is data-driven (len of `subjects`),
    which is exactly the case static `add_edge` can't handle.
    """
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# ─────────────────────────────────────────────────────────────────────────────
# Graph wiring
# ─────────────────────────────────────────────────────────────────────────────
graph_builder = StateGraph(OverallState)
graph_builder.add_node("generate_topics", generate_topics)
graph_builder.add_node("generate_joke", generate_joke)
graph_builder.add_node("best_joke", best_joke)

graph_builder.add_edge(START, "generate_topics")
# The third arg lists the *possible* targets. Required because LangGraph can't
# statically infer target names from a function that returns Send() objects.
graph_builder.add_conditional_edges(
    "generate_topics", continue_to_jokes, ["generate_joke"]
)
# One declared edge, but `generate_joke` runs N times in parallel. LangGraph
# fans the N invocations back into a single `best_joke` call once the `jokes`
# reducer has merged every write.
graph_builder.add_edge("generate_joke", "best_joke")
graph_builder.add_edge("best_joke", END)

graph = graph_builder.compile()
