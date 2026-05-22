"""Sub-graphs: compose a parent graph out of two smaller compiled graphs.

Companion to ``module-4/02-sub-graph/sub-graph.ipynb``. Models a log-processing
pipeline:

- ``clean_logs`` normalizes the raw input.
- ``failure_analysis`` (sub-graph) summarizes logs that contain a failure.
- ``question_summarization`` (sub-graph) summarizes user questions and "ships"
  a report.

The two sub-graphs run in **parallel** as siblings of the parent. Communication
between parent and sub-graphs is by **overlapping state keys**: any key declared
in both schemas is shared. Each sub-graph also declares an explicit
``output_schema`` so only the keys the parent cares about leak back up.
"""

import os
import getpass
from operator import add
from typing import List, Optional, Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


def _set_env(var: str):
    """Prompt for an env var if it isn't already set (no-op in studio where .env supplies them)."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


# LangSmith is optional but the toy graph below is a good fit for tracing — every
# sub-graph step shows up as its own span. Disable both env vars if you don't
# want to send traces.
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"


# ─────────────────────────────────────────────────────────────────────────────
# Shared domain type
# ─────────────────────────────────────────────────────────────────────────────
class Log(TypedDict):
    """A single log entry. Optional fields may or may not be present —
    ``failure_analysis`` distinguishes failures by checking whether `grade` is set."""
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Sub-graph 1: Failure Analysis
# ─────────────────────────────────────────────────────────────────────────────
class FailureAnalysisState(TypedDict):
    """Internal state of the failure-analysis sub-graph. `failures` is a
    scratchpad — useful inside the sub-graph, irrelevant to the parent."""
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]


class FailureAnalysisOutputState(TypedDict):
    """Output schema — only these keys are visible to the parent. Keeps the
    internal `failures` scratchpad from leaking up."""
    fa_summary: str
    processed_logs: List[str]


def get_failures(state):
    """Filter logs down to those that contain a failure (grade is set)."""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}


def generate_summary(state):
    """Summarize the failures. Placeholder string here — swap in an LLM call in real use."""
    failures = state["failures"]
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [f"failure-analysis-on-log-{f['id']}" for f in failures],
    }


fa_builder = StateGraph(
    FailureAnalysisState, output_schema=FailureAnalysisOutputState
)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-graph 2: Question Summarization
# ─────────────────────────────────────────────────────────────────────────────
class QuestionSummarizationState(TypedDict):
    """Internal state. `qs_summary` is consumed by `send_to_slack` and discarded."""
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]


class QuestionSummarizationOutputState(TypedDict):
    """Output schema — `qs_summary` stays internal; only `report` and `processed_logs` surface."""
    report: str
    processed_logs: List[str]


# Note: a function named `generate_summary` already exists for the failure
# sub-graph above. Re-binding the name here is fine — each sub-graph captures
# its own reference when registered with `add_node`. Renaming for clarity is a
# valid alternative; keeping the names parallel is the lesson's choice.
def generate_summary(state):
    """Summarize what users were asking about. Placeholder summary."""
    cleaned_logs = state["cleaned_logs"]
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs],
    }


def send_to_slack(state):
    """Produce a 'report' from the summary. Placeholder for an actual Slack post."""
    qs_summary = state["qs_summary"]
    report = "foo bar baz"
    return {"report": report}


qs_builder = StateGraph(
    QuestionSummarizationState, output_schema=QuestionSummarizationOutputState
)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)


# ─────────────────────────────────────────────────────────────────────────────
# Parent graph
# ─────────────────────────────────────────────────────────────────────────────
class EntryGraphState(TypedDict):
    """Reducer audit:
        raw_logs        – single writer (caller); no reducer needed.
        cleaned_logs    – single writer (`clean_logs`); no reducer needed.
        fa_summary      – only failure-analysis writes it; no reducer needed.
        report          – only question-summarization writes it; no reducer needed.
        processed_logs  – REDUCER REQUIRED: both sub-graphs write it concurrently.
    """
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[List[int], add]


def clean_logs(state):
    """Stub data-cleaning step. In real use this would normalize/dedupe raw_logs."""
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}


entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
# Compiled sub-graphs plug in as nodes — from the parent's perspective they're
# indistinguishable from regular function nodes.
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
# Fan-out: both sub-graphs run in parallel after `clean_logs`.
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()
