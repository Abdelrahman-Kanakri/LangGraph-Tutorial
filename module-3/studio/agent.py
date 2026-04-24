from langchain_core.messages import SystemMessage
from langchain_mistralai import ChatMistral

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# ================= Add Tool definitions =================
def add(a: int, b: int) -> int: 
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b 

# Set the tools list
tools = [add, multiply, divide]

# Define the LLM and binds it with tools
llm = ChatMistral(
                model = "mistral-medium",
                temperature = 0, 
                )
llm_with_tools = llm.bind_tools(tools)

# Create the system message 
sys_msg = SystemMessage(content = "You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# ================= Nodes & Graph =================

# Create the Assistant Node

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]} 

# Build the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Add Edges 
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition, 
)

builder.add_edge("tools", "assistant") # We return the output of the tool node to the agent in order to give the agent:
                                    # the result then end the execution or let it decide to call another tool.

# Compile The Graph

graph = builder.compile()