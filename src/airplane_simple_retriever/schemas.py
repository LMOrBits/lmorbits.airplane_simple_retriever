from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document

class RetriverConfig(BaseModel):
    search_kwargs: dict 

class Config(BaseModel):
    retriver: RetriverConfig


class State(BaseModel):
    messages: list[AnyMessage]
    retriver_result: list[Document] = []
    session_id: str