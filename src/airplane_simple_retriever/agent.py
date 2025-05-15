from functools import partial
from pyapp.observation.phoneix import traced_agent
from pyapp.model_connection.model import  get_model_embeddings_from_config_dir
from pyapp.config import find_config, get_data_dir
from pyapp.vectordb import get_vector_store
from langchain_core.vectorstores.base import VectorStoreRetriever
from pathlib import Path
from airplane_simple_retriever.schemas import AnyMessage
from loguru import logger

here = Path(__file__).resolve()
config_dir = find_config(here)
model = get_model_embeddings_from_config_dir(config_dir=config_dir)
data_dir = get_data_dir(here)

vector_store_path = data_dir / "vectordb/test.db"
raw_data_path = data_dir / "raw_data/test.csv"
table_name = "test"

logger.info(f"vector_store_path: {vector_store_path}")
logger.info(f"raw_data_path: {raw_data_path}")
logger.info(f"table_name: {table_name}")

vector_store, conn = get_vector_store(database_path= str(vector_store_path) , table_name=table_name , embedding_model=model)
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def agent_function(messages: list[AnyMessage], retriever: VectorStoreRetriever) -> str:
    question = messages[-1].content
    return retriever.invoke(question)

def get_agent(search_kwargs: dict = {"k": 5}):
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    agent = partial(agent_function, retriever=retriever)
    return agent
