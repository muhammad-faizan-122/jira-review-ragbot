from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings
from src.common import config
from src.common.utils import measure_time
from src.common.logger import log


class Embedder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_embeder(self):
        return


class HfEmbedder(Embedder):
    def __init__(self):
        super().__init__()

    def get_embeder(self):
        try:
            log.info("Loading embedding model...")
            with measure_time("embedding model loading", log):
                embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL_NAME,
                    model_kwargs=config.MODEL_KWARGS,
                    encode_kwargs=config.ENCODE_KWARGS,
                )
                return embeddings
        except Exception as e:
            log.error(f"Failed to load the HuggingFace embedding Instance: {e}")
            raise
