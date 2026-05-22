from langgraph.graph import END, START, StateGraph

from node import critic_node
from node import generator_node
from node import prompt_writer_node
from node import saver_node
from node import should_continue
from node import web_search_node
from state import State


workflow = StateGraph(State)

workflow.add_node("web_search", web_search_node)
workflow.add_node("prompt_writer", prompt_writer_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)
workflow.add_node("saver", saver_node)

workflow.add_edge(START, "web_search")
workflow.add_edge("web_search", "prompt_writer")
workflow.add_edge("prompt_writer", "generator")
workflow.add_edge("generator", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "prompt_writer": "prompt_writer",
        "saver": "saver",
    },
)

workflow.add_edge("saver", END)

graph = workflow.compile()
