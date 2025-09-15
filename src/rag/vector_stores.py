from abc import ABC, abstractmethod
from src.common import config
from src.common.logger import log
from src.common.utils import measure_time
from langchain_community.vectorstores import Chroma
from src.rag.embeddings import HfEmbedder
import os


class VectorStores(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, embeddings):
        return


class ChromaVectorStore(VectorStores):
    def __init__(self):
        super().__init__()

    def load(self, embeddings):
        if not os.path.exists(config.DB_PERSIST_DIRECTORY):
            raise FileNotFoundError(
                f"Chroma DB directory not found at '{config.DB_PERSIST_DIRECTORY}'. "
                "Please run the ingestion script first (ingest.py)."
            )

        try:
            log.info(f"Loading vector store from {config.DB_PERSIST_DIRECTORY}...")
            with measure_time("vector db instance", log):
                vector_store = Chroma(
                    persist_directory=config.DB_PERSIST_DIRECTORY,
                    embedding_function=embeddings,
                )
                return vector_store
        except Exception as e:
            log.error(f"Failed to load the Chroma Vector store: {e}")
            raise


def load_vector_store(store_type="chroma"):
    embeddings = HfEmbedder().get_embeder()
    if store_type:
        vector_store = ChromaVectorStore().load(embeddings=embeddings)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
    return vector_store
