from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.common import config
from src.common.utils import load_reviews_documents
from src.common.logger import setup_logger
from src.common.utils import measure_time


log = setup_logger(file_name="ingest.log")


def ingest_data():
    """
    Main function to load data, create documents, and build the vector store.
    """
    log.info("Starting data ingestion process...")

    log.info(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")

    with measure_time("Embedding model loading", log):
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs=config.MODEL_KWARGS,
            encode_kwargs=config.ENCODE_KWARGS,
        )
    with measure_time("Load reveiws data", log):
        reviews_documents = load_reviews_documents()

    log.info(
        f"Creating and persisting vector store at {config.DB_PERSIST_DIRECTORY}..."
    )

    with measure_time("Data ingestion in Chroma Vector DB", log):
        Chroma.from_documents(
            documents=reviews_documents,
            embedding=embeddings,
            persist_directory=config.DB_PERSIST_DIRECTORY,
        )

    log.info("Data ingestion complete. Vector store is ready.")


if __name__ == "__main__":
    ingest_data()
