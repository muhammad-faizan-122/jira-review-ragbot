# Jira Review RAGBot
The goal is to build a chatbot that answers user questions in the context of a Jira reviews knowledge base. When a user asks a question about Jira, the AI bot retrieves relevant documents from the knowledge base and passes them to an LLM (Large Language Model) to generate an answer based on the retrieved documents.

![alt text](images/demo.png)

## Block Diagram
![alt text](images/block_diagram.png)
## usage
run ingestion script
```
python3 src/ingest/main.py
```