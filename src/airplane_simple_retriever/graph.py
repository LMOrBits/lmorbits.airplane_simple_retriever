from airplane_simple_retriever.agent import retriver 
from airplane_simple_retriever.schemas import  State
from langgraph.graph import StateGraph
from typing import Optional





class Graph:
    name = "airplane-simple-retriever"
    nodes = {}
    state = State

    def __init__(self):
        retriver_name = f"{self.name}.retriver"

        self.nodes = {
            "start": retriver_name,
            "end": retriver_name,
 
        }
    
    def add_workflow(self,graph: StateGraph , end_node: Optional[str] = None, start_node: Optional[str] = None ):
        graph.add_node(self.nodes["start"], retriver)

        if start_node and end_node:
            graph.add_edge(start_node, self.nodes["start"])
            graph.add_edge(self.nodes["start"], end_node)
        elif start_node:
            graph.add_edge(start_node, self.nodes["start"])
        elif end_node:
            graph.add_edge(self.nodes["start"], end_node)
            graph.set_entry_point(self.nodes["start"])
        else:
            graph.set_entry_point(self.nodes["start"])
        return graph



def get_app():
    airplane_simple_retriever_graph = Graph()
    graph = StateGraph(State)
    graph = airplane_simple_retriever_graph.add_workflow(graph)
    return graph.compile()









