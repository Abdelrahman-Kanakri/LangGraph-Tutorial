from typing import Literal
import random 
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# State
class State(TypedDict):
    graph_state: str

# Conditional edge
def decide_node(state: State) -> Literal["node2", "node3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state["graph_state"]
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5: 
        return "node2"
    else: 
        return "node3"
    
    return "node3"


# Conditional edge
def node1(state: State) -> State:
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}

def node2(state: State) -> State:
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " Happy!"}

def node3(state: State) -> State:
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " Sad!"}



# Build graph
builder = StateGraph(State)
# Add Nodes
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

# Logic (Edges)
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_node)
builder.add_edge("node2", END)
builder.add_edge("node3", END)


# Compile graph
graph = builder.compile()
