from typing_extensions import TypedDict
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    input: str

# Tools 
def step1(state: State) -> State:
    print("---Step 1---")
    return state

def step2(state: State) -> State:
    # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
    
    print("---Step 2---")
    return state
def step3(state: State) -> State:
    print("---Step 3---")
    return state

# Build the Graph

builder = StateGraph(State)

# Add nodes
builder.add_node("step_1", step1)
builder.add_node("step_2", step2)
builder.add_node("step_3", step3)

# Add Edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Compile the graph
graph = builder.compile()