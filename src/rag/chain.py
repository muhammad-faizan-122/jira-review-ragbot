from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from operator import itemgetter
from typing import List
from src.common import config
from src.common.logger import log
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.common.utils import measure_time
from abc import ABC, abstractmethod

load_dotenv()


class Generation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_response(self, query, retriever) -> str:
        return


class LcGeneration(Generation):
    """generation using Langchain module"""

    def __init__(self):
        super().__init__()

    def format_retrieved_document(self, docs: List[Document]) -> str:
        log.debug(f"--- Inspecting Retrieved Documents ---: {docs}")
        formated_docs = []
        for i, doc in enumerate(docs):
            formated_doc = f"Document-{i}:\n"
            for k, v in doc.metadata.items():
                formated_doc += f"{k}: {v}\n"
            formated_doc += f"**Review detail**:\n{doc.page_content}"
            formated_docs.append(formated_doc)

        formatted_str_docs = "\n\n".join(formated_docs)
        log.debug(
            f"Updated Document after merging metadata: {formatted_str_docs[:500]}..."
        )
        return formatted_str_docs

    def _log_final_prompt(self, prompt: ChatPromptTemplate):
        """
        A function to debug the final prompt object before it goes to the LLM.
        """
        log.debug("--- Final Prompt Sent to LLM ---", prompt.to_string())
        return prompt  # Pass the prompt through unchanged

    def generate_response(self, query, retriever) -> str:
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

        with measure_time("retrieve relevant documents", log):
            retrieval_and_formatting_chain = (
                itemgetter("input")
                | retriever
                | RunnableLambda(self.format_retrieved_document)
            )

        with measure_time("Chain initialization", log):
            answer_generation_chain = (
                config.RAG_GENERATION_PROMPT
                | RunnableLambda(self._log_final_prompt)
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
        response = rag_chain.invoke({"input": query})
        return (
            response
            if response and isinstance(response, str)
            else "Sorry I am unable to answer from Jira Knowledge base"
        )
