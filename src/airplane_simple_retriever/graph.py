from pyapp.utils.config import get_pyapp_config
from airplane_simple_retriever.agent import get_agent
from airplane_simple_retriever.schemas import Config, State
from airplane_simple_retriever.utils import message_converter

from langgraph.graph import StateGraph
from typing import Optional

from pathlib import Path
config = get_pyapp_config(Config, Path(__file__))
agent = get_agent(config.retriver.search_kwargs)



def retriver(state: State) -> State:
    result = agent(state.messages)
    return {"retriver_result": result}

def add_workflow(graph: StateGraph , end_node: Optional[str] = None, start_node: Optional[str] = None ):
    app_name = "airplane-simple-retriever"
    retriver_name = f"{app_name}.retriver"
    if start_node and end_node:
        graph.add_node(retriver_name, retriver)
        graph.add_edge(start_node, retriver_name)
        graph.add_edge(retriver_name, end_node)
    elif start_node:
        graph.add_node(retriver_name, retriver)
        graph.add_edge(start_node, retriver_name)
    elif end_node:
        graph.add_node(retriver_name, retriver)
        graph.add_edge(retriver_name, end_node)
        graph.set_entry_point(retriver_name)
    else:
        graph.add_node(retriver_name, retriver)
        graph.set_entry_point(retriver_name)
    return graph

def get_app():
    graph = StateGraph(State)
    add_workflow(graph)
    return graph.compile()









