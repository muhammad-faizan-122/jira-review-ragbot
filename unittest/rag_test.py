import sys

sys.path.append("../")

from src.rag.rag_executor import jira_rag_agent


rag_response = jira_rag_agent.get_rag_response(query="effect of Jira on productivity")
print("rag response: ", rag_response)
