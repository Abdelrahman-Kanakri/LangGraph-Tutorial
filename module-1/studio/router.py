import os, getpass
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage    
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

# Define function to set environment variable if not already set
def _set_env(var: str): 
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"Enter value for {var}: ")
        return f"{var} set successfully."

# Set the MISTRAL_API_KEY environment variable if not already set
_set_env("MISTRAL_API_KEY")


# Tool for the LLM to use to multiply two numbers.
def multiply(a: int, b: int) -> int: 
    """_summary_
    Args:
        a (int): first number
        b (int): second number

    Returns:
        int: the product of a and b
    """
    return a * b

# Initialize the LLM and bind the tool to it.
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0)
llm_with_tools = llm.bind_tools([multiply])


# Node Call Function
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the Graph
builder = StateGraph(MessagesState)

# add_nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

# add_edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    
    tools_condition,
)
builder.add_edge("tools", END)

# Compile the Graph
graph = builder.compile()



messages = [HumanMessage(content = "What is the product of 5 and 10?")]
messages = graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()