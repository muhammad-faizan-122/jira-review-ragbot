from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from src.common.logger import log
from src.common.config import DENSE_RETRIEVED_DOCUMENTS, SPARSE_RETRIEVED_DOCUMENTS


def create_ensemble_retriever(
    vectorstore: Chroma, documents: List[Document], weights: List[float]
) -> EnsembleRetriever:
    """
    Creates and returns an EnsembleRetriever combining a dense and a sparse retriever.

    Args:
        vectorstore: The Chroma vector store for dense retrieval.
        documents: The list of documents for initializing the sparse retriever.
        weights: A list of two floats to weigh the dense and sparse retrievers.

    Returns:
        An configured EnsembleRetriever.
    """
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": DENSE_RETRIEVED_DOCUMENTS}
    )
    bm25_retriever = BM25Retriever.from_documents(
        documents, k=SPARSE_RETRIEVED_DOCUMENTS
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever], weights=weights
    )

    log.info("Ensemble retriever created successfully.")
    return ensemble_retriever
