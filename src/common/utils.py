from contextlib import contextmanager
from typing import List, Dict, Any
from src.common.logger import log
from src.common import config
from langchain.docstore.document import Document
import json
import time


@contextmanager
def measure_time(label, log):
    start = time.time()
    yield
    end = time.time()
    log.info(f"{label} took {end - start:.2f} seconds")


def load_json_from_file(fp: str) -> List[Dict[str, Any]]:
    """Loads a JSON file and returns its content."""
    try:
        with open(file=fp, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Error: The file at {fp} was not found.")
        return []
    except json.JSONDecodeError:
        log.error(f"Error: The file at {fp} is not a valid JSON file.")
        return []


def convert_to_documents(data: list[dict]):
    if not data:
        log.error("No data found. Exiting.")
        raise ValueError("No dataset found.")

    documents = [
        Document(
            page_content=d.get("review_detail", ""),
            metadata={
                "author": d.get("author", "Unknown"),
                "review_date": d.get("review_date", "Unknown"),
                "rating": d.get("rating", 0),
            },
        )
        for d in data
    ]
    return documents


def load_reviews_documents():
    log.info(f"Loading reviews from {config.REVIEW_DATA_PATH}...")
    reviews = load_json_from_file(config.REVIEW_DATA_PATH)
    reviews_docs = convert_to_documents(reviews)
    log.info(f"Total reveiws documents: {len(reviews_docs)}")
    return reviews_docs
