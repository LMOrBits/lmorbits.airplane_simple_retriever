from airplane_simple_retriever.agent import get_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from airplane_simple_retriever.graph import get_app
from pyapp.observation.phoneix import traced_agent
import uuid
from airplane_simple_retriever.schemas import AnyMessage

app = get_app()


@traced_agent(name="airplane-simple-retriever")
def app_invoke(messages: list[dict|AnyMessage], session_id: str):
    state = app.invoke({"messages": messages, "session_id": session_id})
    return "\n".join([doc.page_content for doc in state["retriver_result"]])

def inference():
    messages = [
        HumanMessage(content="what is the capital of france?")
    ]
    session_id = str(uuid.uuid4())
    return app_invoke(messages, session_id)


def generate():
    try:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path=str(raw_data_path.resolve()))
        data = loader.load()
        vector_store.add_documents(data)
    finally:
        conn.close()
