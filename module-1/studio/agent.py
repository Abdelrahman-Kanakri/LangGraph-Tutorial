from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

import os, getpass
from dotenv import load_dotenv

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set environment variables for Mistral API key and LangSmith API key, and configure LangSmith tracing and project settings.
_set_env("MISTRAL_API_KEY")
_set_env("LANGSMITH_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"


# Tools for the agent to use
def multiply(a: int, b: int) -> int: 
    """
    args: 
        a: int
        b: int
        
    returns:
        int: the product of a and b
    """
    return a * b

def add(a: int, b: int) -> int: 
    """
    args: 
        a: int
        b: int
        
    returns:
        int: the sum of a and b
    """
    return a + b

def divide(a: int, b: int) -> int: 
    """
    args: 
        a: int
        b: int
        
    returns:
        int: the quotient of a and b
    """
    return a / b

tools = [multiply, add, divide]
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)

llm_with_tools = llm.bind_tools(tools)


# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState) -> MessagesState:
    # Get the human message from the state
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

# the System message is added to the list of messages to provide the model with instructions
# on how to respond to the human message, and how to use the tools if necessary.3
# So:

# Added once per call ✅
# Added once total to state ❌ — it never goes into state at all


# ------------              Graph Building              ------------
# Build the graph
builder = StateGraph(MessagesState)


# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant", 
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")


# Compile the graph
react_graph = builder.compile()

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)



# Specify a thread_id
config = {"configurable": {"thread_id": "1"}}

#  ------------              Input 1              ------------
# Specify the input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages}, config=config)


for m in messages['messages']:
    m.pretty_print()

#  ------------              Input 2              ------------
# Another input
messages = [HumanMessage(content="Multiply that by 2")]

# Run
messages = react_graph_memory.invoke({"messages": messages}, config=config)

for m in messages['messages']:
    m.pretty_print()


#  ------------              Input 3              ------------
# Add another input
messages = [HumanMessage(content="Divide that by 5.")]

# Run
messages = react_graph_memory.invoke({"messages": messages}, config = config)


for m in messages["messages"]:
    m.pretty_print()