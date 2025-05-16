from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document

class RetriverConfig(BaseModel):
    search_kwargs: dict 
    similarity_threshold: float

class DataConfig(BaseModel):
    raw_data_path: str
    vector_store_path: str
    table_name: str

class Config(BaseModel):
    retriver: RetriverConfig
    data_config: DataConfig

class State(BaseModel):
    messages: list[AnyMessage]
    retriver_result_docs: list[Document] = []
    retriver_results: list[str] = []
    session_id: str


class QA(BaseModel):
    question: str
    answer: str
    source: str
    
    def __str__(self):
        return f"**Question:** {self.question}\n**Answer:** {self.answer}"
    def markdown(self):
        return f"**Question:** {self.question}\n**Answer:** {self.answer} \n**Source:** [Link]({self.source})"