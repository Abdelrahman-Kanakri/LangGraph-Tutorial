"""Parallelization (fan-out / fan-in): two retrievers running in the same super-step.

Companion to ``module-4/01-parallelization/parallelization.ipynb``. The graph
gathers context from Tavily web search and Wikipedia **in parallel**, then hands
the merged context to an LLM to answer the question.

The key LangGraph idea on display is the reducer on ``context`` — without it,
the two concurrent writes from ``search_web`` and ``search_wikipedia`` would
hard-error with ``INVALID_CONCURRENT_GRAPH_UPDATE``.
"""

import os
import getpass
import operator
import warnings
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader

from langgraph.graph import StateGraph, START, END


# Silences the BeautifulSoup "GuessedAtParserWarning" emitted by the Wikipedia
# loader on first call. It's noisy in studio logs and has no functional effect.
warnings.filterwarnings("ignore")


def _set_env(var: str):
    """Prompt for an env var if it isn't already set (no-op in studio where .env supplies them)."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("MISTRAL_API_KEY")
_set_env("TAVILY_API_KEY")

llm = ChatMistralAI(model="mistral-medium", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    answer: str
    # REDUCER REQUIRED: search_web and search_wikipedia both write `context` in
    # the same super-step. Without `operator.add` LangGraph raises
    # INVALID_CONCURRENT_GRAPH_UPDATE on the parallel writes.
    context: Annotated[list, operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval nodes (run in parallel)
# ─────────────────────────────────────────────────────────────────────────────
def search_web(state):
    """Retrieve docs from Tavily web search."""
    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state["question"]})
    search_docs = data.get("results", data)

    formatted_search_docs = "\n\n---\n\n".join(
        f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
        for doc in search_docs
    )
    # Returns a list of length 1 so `operator.add` concatenates contributions
    # from this branch and `search_wikipedia` into a 2-element list.
    return {"context": [formatted_search_docs]}


def search_wikipedia(state):
    """Retrieve docs from Wikipedia."""
    search_docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n'
        f"{doc.page_content}\n</Document>"
        for doc in search_docs
    )
    return {"context": [formatted_search_docs]}


# ─────────────────────────────────────────────────────────────────────────────
# Answer node (runs after the fan-in barrier)
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(state):
    """Answer the question using the merged context from both retrievers."""
    context = state["context"]
    question = state["question"]

    answer_template = "Answer the question {question} using this context: {context}"
    answer_instructions = answer_template.format(question=question, context=context)

    answer = llm.invoke(
        [SystemMessage(content=answer_instructions)]
        + [HumanMessage(content="Answer the question.")]
    )
    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# Graph wiring
# ─────────────────────────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Two edges from START → two parallel retrievers in the same super-step.
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")

# Both retrievers feed `generate_answer`; LangGraph waits for both before firing it.
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
