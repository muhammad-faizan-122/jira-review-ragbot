from langchain_core.prompts import ChatPromptTemplate

# --- Paths and Directories ---
REVIEW_DATA_PATH = "data/all_reviews.json"
DB_PERSIST_DIRECTORY = "chroma_db"

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_KWARGS = {"device": "cpu"}  # Use "cuda" for GPU
ENCODE_KWARGS = {"normalize_embeddings": False}

# --- Retriever Configuration ---
ENSEMBLE_RETRIEVER_WEIGHTS = [0.5, 0.5]  # [dense, sparse]
DENSE_RETRIEVED_DOCUMENTS = 3
SPARSE_RETRIEVED_DOCUMENTS = 3

# --- LLM and Prompt Configuration ---
LLM_MODEL_NAME = "gemini-1.5-flash"

ROUTER_PROMPT = """You are router who is responsible to select either 'rag' or 'chat'. \
If user ask query specifically related to Jira (a project managment tool) or ask related project management related things \
without specifically mentioning name of 'Jira' name, select 'rag' for answering from Jira knowledge base, \
otherwise select 'chat'.
"""

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given Jira reviews, your task is to answer the user's query "
    "based *only* on the provided review context.\n"
    "Do NOT make up any answers. If the answer is not found in the reviews, respond with: "
    "'I cannot answer this based on the provided reviews.'\n\n"
    "Format your answer as follows:\n"
    "1. Provide a concise answer using inline numbered citation references like [1], [2], etc.\n"
    "2. After the answer, include a 'Sources:' section, where each number corresponds to the reference used.\n"
    "   Each source should include: author name, rating, the date of the review, and a short phrase summarizing the relevant point.\n\n"
    "Example output format:\n"
    "Jira helps teams collaborate efficiently [1]. It supports agile methodologies like Scrum [2].\n\n"
    "Sources:\n"
    "1. Freda rated it 5.0 and mentioned this on December 2024.\n"
    "2. Rajiv rated it 5.0 and mentioned this on June 2024.\n\n"
    "Only use the following Documents of Jira Reviews as Context:\n\n"
    "{context}"
)

RAG_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


CHATBOT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's query very concisely.\nConversation history:\n{conversation_history}",
        ),
        ("user", "{user_query}"),
    ]
)

# TRIM MESSAGE CONFIGS
CONVERSATION_HISTORY_TURNS = 5
CONVERSATION_MAX_TURNS = 10
