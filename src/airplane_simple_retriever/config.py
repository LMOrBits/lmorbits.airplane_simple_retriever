from pyapp.model_connection.model import  get_model_embeddings_from_config_dir
from pyapp.config import find_config, get_data_dir
# from pyapp.vectordb import get_vector_store as get_vector_store_from_pyapp
from pathlib import Path
from loguru import logger
from functools import partial
from airplane_simple_retriever.schemas import Config
from pyapp.utils.config import get_pyapp_config
from langchain_chroma import Chroma

here = Path(__file__).resolve()
config_dir = find_config(here)
model = get_model_embeddings_from_config_dir(config_dir=config_dir)
data_dir = get_data_dir(here)

config = get_pyapp_config(Config, Path(__file__))
vector_store_path = data_dir / config.data_config.vector_store_path
raw_data_path = data_dir / config.data_config.raw_data_path
table_name = config.data_config.table_name

logger.info(f"vector_store_path: {vector_store_path}")
logger.info(f"raw_data_path: {raw_data_path}")
logger.info(f"table_name: {table_name}")

vectorstore = Chroma(
        collection_name=table_name,
        embedding_function=model,
        persist_directory=str(vector_store_path.resolve()),
    )

# get_vector_store = partial(get_vector_store_from_pyapp, database_path= str(vector_store_path) , table_name=table_name , embedding_model=model)

