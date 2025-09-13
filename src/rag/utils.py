import json
from typing import List, Dict, Any
from src.common.logger import log
from src.common import config
from langchain.docstore.document import Document


def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Error: The file at {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        log.error(f"Error: The file at {file_path} is not a valid JSON file.")
        return []


def load_reviews_documents():
    log.info(f"Loading reviews from {config.REVIEW_DATA_PATH}...")
    reviews = load_json_from_file(config.REVIEW_DATA_PATH)
    if not reviews:
        log.warning("No reviews found. Exiting.")
        return

    log.info("Creating Document objects from reviews...")
    documents = []
    for review in reviews:
        doc = Document(
            page_content=review.get("review_detail", ""),
            metadata={
                "author": review.get("author", "Unknown"),
                "review_date": review.get("review_date", "Unknown"),
                "rating": review.get("rating", 0),
            },
        )
        documents.append(doc)
    log.info(f"Created {len(documents)} documents.")
    return documents
