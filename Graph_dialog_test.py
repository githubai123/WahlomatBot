from PIL import Image
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph,END

# Assuming `workflow` is your LangGraph object
from typing import Dict, TypedDict, Optional,Annotated,List
import operator


class GraphState(TypedDict):
    messages: Annotated[List[str], operator.add]

def example_node():
    pass


workflow = StateGraph(GraphState)

        # Define the nodes
workflow.add_node("host_introduction",example_node )  # web search
workflow.add_node("host_question",example_node )  # web search
workflow.add_node("rag_guest_1",example_node )  # web search
workflow.add_node("rag_guest_2",example_node )  # web search
workflow.add_node("evaluator",example_node )  # web search

workflow.add_node("host_verdict",example_node )  # web search

workflow.add_edge("host_introduction","host_question")
workflow.add_edge("host_question","rag_guest_1")
workflow.add_edge("rag_guest_1","rag_guest_2")
workflow.add_edge("rag_guest_2","evaluator")
workflow.add_edge("evaluator","rag_guest_1")
workflow.add_edge("host_verdict",END)


def should_continue(state: GraphState) -> str:
    if "continue" in state['messages'][-1]:
        return "continue"
    else:
        return "END"
    
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "END": "host_verdict",
        "continue": "rag_guest_1"
    }
)


workflow.set_entry_point("host_introduction")

app = workflow.compile()
graph_image = app.get_graph()
print(type(graph_image))
graph_image.draw_png("Graph_chat.png")
# Display the image
