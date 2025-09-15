from src.common.utils import measure_time
from src.rag.retriever import create_ensemble_retriever
from src.rag.chain import LcGeneration
from src.common.logger import log


class RAGExecutor:
    """
    A class to encapsulate the RAG chain for querying Jira reviews.

    This class handles the one-time initialization of models, vector stores,
    and retrievers to be used throughout the application's lifecycle.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RAGExecutor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the RAG components. This is done only once.
        """
        if self._initialized:
            return

        log.info("Initializing JiraRAGExecutor...")
        self.ensemble_retriever = create_ensemble_retriever()
        self.generator = LcGeneration()
        self._initialized = True

    def get_response(self, query: str) -> str:
        try:
            log.info(f"Invoking RAG chain with query: '{query}'")
            return self.generator.generate_response(
                retriever=self.ensemble_retriever, query=query
            )
        except Exception as e:
            log.error(f"Failed to get RAG response: {e}")
            return "An error occurred while processing your request."


jira_rag_agent = RAGExecutor()
