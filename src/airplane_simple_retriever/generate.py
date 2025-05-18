from langchain_core.documents import Document 
from airplane_simple_retriever.config import get_vector_store, raw_data_path
# from airplane_simple_retriever.config import vectorstore , raw_data_path
import yaml
from pathlib import Path
from airplane_simple_retriever.schemas import QA
from loguru import logger

def generate(store_in_db: bool = True) -> list[Document] | None:
    yaml_dir = raw_data_path
    read_yaml = yaml.safe_load(open(Path(yaml_dir),"r"))
    qas: list[QA] = [QA(**qa,source=read_yaml["source"]) for qa in read_yaml["questions"]]
    qas_metadata = [{**qa.model_dump() , "markdown":qa.markdown()} for qa in qas]
    documents=[Document(page_content=str(qa),metadata = {**qa.model_dump() , "markdown":qa.markdown()}) for qa in qas]
    if store_in_db:
        vector_store , conn = get_vector_store()
        vector_store.add_documents(documents)
        logger.info(f"Generated {len(documents)} embeddings and stored in db")
        conn.close()
        return None

    return documents


