from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from operator import itemgetter
from typing import List
from src.common import config
from src.common.logger import log
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.common.utils import measure_time


load_dotenv()


def format_retrieved_document(docs: List[Document]) -> str:
    log.debug(f"--- Inspecting Retrieved Documents ---: {docs}")
    formated_docs = []
    for i, doc in enumerate(docs):
        formated_doc = f"Document-{i}:\n"
        for k, v in doc.metadata.items():
            formated_doc += f"{k}: {v}\n"
        formated_doc += f"**Review detail**:\n{doc.page_content}"

        formated_docs.append(formated_doc)

    formatted_str_docs = "\n\n".join(formated_docs)
    log.debug("\n--- Documents Updated Successfully ---\n")
    log.debug(f"Updated Document after merging metadata: {formatted_str_docs}")
    return formatted_str_docs


def log_final_prompt(prompt: ChatPromptTemplate):
    """
    A function to debug the final prompt object before it goes to the LLM.
    """
    log.debug("--- Final Prompt Sent to LLM ---", prompt.to_string())
    return prompt  # Pass the prompt through unchanged


def create_rag_chain(retriever: EnsembleRetriever) -> Runnable:
    """
    Creates and returns the main RAG chain. This chain:
    1. Retrieves documents.
    2. Allows inspection of the Document objects.
    3. Formats the documents' page_content into a single string.
    4. Assigns that string to the 'context' variable.
    5. Logs the final prompt before sending it to the LLM.
    6. Invokes the LLM and parses the output.

    Args:
        retriever: The configured EnsembleRetriever to use for fetching context.

    Returns:
        A runnable RAG chain.
    """
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME)

    with measure_time("retrieval", log):
        retrieval_and_formatting_chain = (
            itemgetter("input") | retriever | RunnableLambda(format_retrieved_document)
        )

    with measure_time("Chain initialization", log):

        answer_generation_chain = (
            config.PROMPT_TEMPLATE
            | RunnableLambda(log_final_prompt)
            | llm
            | StrOutputParser()
        )

    with measure_time("Prompt Augment + generation", log):
        rag_chain = (
            RunnablePassthrough.assign(context=retrieval_and_formatting_chain)
            | answer_generation_chain
        )

    log.info(
        "RAG chain with document formatting and prompt inspection created successfully."
    )
    return rag_chain
