# Operating System Imports
import sqlite3
import os 

from dotenv import load_dotenv

load_dotenv()

os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# LangChain Imports
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage, 
    )
# LangGraph Imports
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import (
    MessagesState,
    StateGraph,
    START,
    END,
)

# Other Imports
from typing_extensions import Literal
from IPython.display import Image, display

# ============================= DEFINE THE MODEL & State =============================
# define the model
model = ChatMistralAI(model="mistral-medium", temperature=0)

# Create the Summary State
class State(MessagesState):
    summary: str

# ============================ DEFINE THE GRAPH NODES =============================
# Define the logic to call the model
def call_model(state: State): 
    # Get summary if it exists
    summary = state.get("summary", "")
    
    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

# The Summarization Logic Node
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    # Return the new summary and the messages to delete
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal ["summarize_conversation", END]:
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# ============================ DEFINE THE GRAPH =============================
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)


# ============================ COMPILE & VISUALIZE THE GRAPH =============================
# Compile
graph = workflow.compile()
# Visualize the Graph
display(Image(graph.get_graph().draw_mermaid_png()))

# # =========================== INVOKE THE GRAPH =============================
# # Create a thread
# config = {"configurable": {"thread_id": "1"}}

# # Start conversation
# input_message = HumanMessage(content="hi! I'm Abood.")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()

# input_message = HumanMessage(content="what's my name?")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()

# input_message = HumanMessage(content="i like the Counter Strike Global Offensive!")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()