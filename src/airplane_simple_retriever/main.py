from pathlib import Path
from langchain_core.messages import HumanMessage
from airplane_simple_retriever.graph import get_app
from pyapp.observation.phoneix import traced_agent
import uuid
from airplane_simple_retriever.schemas import AnyMessage

app = get_app()


@traced_agent(name="airplane-simple-retriever")
def app_invoke(messages: list[dict|AnyMessage], session_id: str):
    state = app.invoke({"messages": messages, "session_id": session_id})
    return state["retriver_results"]

def inference():
    messages = [
        HumanMessage(content="When do I need to be at the boarding gate?")
    ]
    session_id = str(uuid.uuid4())
    return app_invoke(messages, session_id)



def get_graph():
    from langchain_core.runnables.graph import MermaidDrawMethod
    png_bytes = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    with open(Path.cwd() / "airplane_simple_retriever.png", "wb") as f:
        f.write(png_bytes)
    return app