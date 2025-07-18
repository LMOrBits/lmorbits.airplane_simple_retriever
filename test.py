
from pyapp.model_connection.model import  get_model_embeddings_from_config_dir
from pyapp.config import find_config
from pathlib import Path

from airplane_simple_retriever.config import get_vector_store

here = Path(__file__).resolve()
config_dir = find_config(here)
model = get_model_embeddings_from_config_dir(config_dir=config_dir)


answer = model.embed_documents(["hi"])
print(len(answer))
answer = model.embed_documents(["hi", "hello"])
print(len(answer))
print(answer)

try:
    vectorstore , conn = get_vector_store()
    print(vectorstore.similarity_search("hi", k=1))
except Exception as e:
    print(e)
finally:
    conn.close()