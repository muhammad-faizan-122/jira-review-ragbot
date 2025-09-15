from abc import ABC, abstractmethod
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from src.common.utils import load_reviews_documents
from src.rag.vector_stores import load_vector_store
from src.common import config
from src.common.logger import log


class DenseRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retriever(self, vector_store):
        return


class SparseRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_retriever(self):
        return


class ChromaRetriever(DenseRetriever):
    def __init__(self):
        super().__init__()

    def get_retriever(self, vector_store):
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": config.DENSE_RETRIEVED_DOCUMENTS,
            }
        )
        return retriever


class Bm25Retriever(SparseRetriever):
    def __init__(self):
        super().__init__()

    def get_retriever(self, documents):
        retriever = BM25Retriever.from_documents(
            documents,
            k=config.SPARSE_RETRIEVED_DOCUMENTS,
        )
        return retriever


def create_ensemble_retriever() -> EnsembleRetriever:
    """
    Creates and returns an EnsembleRetriever combining a dense and a sparse retriever.

    Args:
        vectorstore: The Chroma vector store for dense retrieval.
        documents: The list of documents for initializing the sparse retriever.
        weights: A list of two floats to weigh the dense and sparse retrievers.

    Returns:
        An configured EnsembleRetriever.
    """

    sparse_retriever = Bm25Retriever().get_retriever(documents=load_reviews_documents())
    dense_retriever = ChromaRetriever().get_retriever(vector_store=load_vector_store())

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
    weights=config.ENSEMBLE_RETRIEVER_WEIGHTS,
    )

    log.info("Ensemble retriever created successfully.")
    return ensemble_retriever
