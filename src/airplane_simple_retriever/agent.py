from airplane_simple_retriever.schemas import State
from langchain_core.vectorstores.base import VectorStoreRetriever
from airplane_simple_retriever.schemas import AnyMessage
# from airplane_simple_retriever.config import vectorstore
from airplane_simple_retriever.config import get_vector_store
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter , DocumentCompressorPipeline  
from airplane_simple_retriever.config import model
from airplane_simple_retriever.config import config
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever



redundant_filter = EmbeddingsRedundantFilter(embeddings=model)
relevant_filter = EmbeddingsFilter(embeddings=model, similarity_threshold=config.retriver.similarity_threshold)
# _retriever = vectorstore.as_retriever(search_kwargs=config.retriver.search_kwargs)

def pretty_print_docs(docs):
    answer = ""
    if docs:
        answer = (
            f"\n\n{'-' * 3}\n\n".join(
                [f"# Document {i}:\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
    else:
        answer = "No relevant documents found. \n if the question is about greeting answer it properly and if not mention you do not know the answer."
    return answer



def retriver(state: State) -> State:
    vectorstore , conn = get_vector_store()
    _retriever = vectorstore.as_retriever(search_kwargs=config.retriver.search_kwargs)

    pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter , relevant_filter ]
)
    compression_retriever = ContextualCompressionRetriever(
    base_retriever=_retriever,
    base_compressor=pipeline_compressor,
)
    chain = compression_retriever.with_config(run_name="Docs") 
    result = chain.invoke(state.messages[-1].content)
    conn.close()
    return {"retriver_result_docs": result , "retriver_results": pretty_print_docs(result)}

