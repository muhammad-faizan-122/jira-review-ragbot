from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from src.bot.states import State
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from src.rag.rag_executor import jira_rag_agent
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import get_buffer_string
from src.common import config
from src.common.utils import measure_time
from src.common.logger import log
from dotenv import load_dotenv


load_dotenv(override=True)


class QueryRouter(BaseModel):
    """Router for selecting the next node."""

    route: str = Field(
        description=config.ROUTER_PROMPT,
        enum=["rag", "chat"],
    )


class RAGNode:
    def __init__(self):
        pass

    def execute(self, state: State) -> dict:
        """
        Executes the RAG logic.
        """
        log.debug(f"Executing ChatbotNode with state: {state}")
        try:
            query = state["messages"][-1].content
            if not query:
                return {
                    "messages": [AIMessage(content="Please provide a valid question.")]
                }

            with measure_time("RAG answer generation", log):
                rag_response = jira_rag_agent.get_response(query=query)
            log.debug("rag_response langGraph bot: ", rag_response)
            return {"messages": [AIMessage(content=rag_response)]}
        except Exception as e:
            log.error(f"Error in RAGNode execution: {e}")
            return {
                "messages": [AIMessage(content="Failed to generate desired results")]
            }


class ChatbotNode:
    """
    Generates a response based on the conversation history.
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.history_turns = config.CONVERSATION_HISTORY_TURNS
        self.max_turns = config.CONVERSATION_MAX_TURNS

    def trim_messages(self, messages):
        """trim the old messages to avoid context window issue due to LLM limitation"""
        total_messages = len(messages)
        start_conversation = max(0, total_messages - self.history_turns - 1)
        return get_buffer_string(
            messages[start_conversation:-1]
            if total_messages > self.max_turns
            else messages[:-1]
        )

    def execute(self, state: State) -> dict:
        """
        Executes the chatbot logic. they are used as previous conversation history.
        """
        log.debug(f"Executing ChatbotNode with state: {state}")
        messages = state["messages"]
        conversation_history = self.trim_messages(messages)
        prompt = config.CHATBOT_TEMPLATE.invoke(
            {
                "conversation_history": conversation_history,
                "user_query": messages[-1].content,
            }
        )
        messages = prompt.to_messages()
        log.debug(f"LLM prompt messages with context: {messages}")
        try:
            response = self.chat_model.invoke(messages)
            log.debug(f"ChatbotNode response: {response}")
            return {"messages": [response]}

        except Exception as e:
            log.error(f"Error in ChatbotNode execution: {e}")
            return {
                "messages": [AIMessage(content="Failed to generate desired results")]
            }


class GraphBuilder:
    """
    Builds the LangGraph for the chatbot with rag search capability.
    """

    @staticmethod
    def router_function(state: State, structured_llm):
        """
        Determines the next step in the graph.
        """
        log.info("Executing router.")
        query = state["messages"][-1].content
        try:
            router_result = structured_llm.invoke(query)
            log.debug(f"Router decision: {router_result.route}")
            return router_result.route
        except Exception as e:
            log.error(f"Error in router execution: {e}")
            # Default to chatbot on error
            return "chat"

    @staticmethod
    def build_graph(model_name: str = "google_genai:gemini-1.5-flash"):
        """
        Builds and compiles the LangGraph.
        """
        chat_model = init_chat_model(model_name)

        # LLM with function calling for the router
        structured_llm = chat_model.with_structured_output(QueryRouter)

        # Initialize nodes
        chatbot_node = ChatbotNode(chat_model=chat_model)
        rag_node = RAGNode()

        bot_graph = StateGraph(State)

        # Add nodes to the graph
        bot_graph.add_node("chatbot", chatbot_node.execute)
        bot_graph.add_node("rag_search", rag_node.execute)

        # The entry point is now a conditional router
        bot_graph.add_conditional_edges(
            START,
            lambda state: GraphBuilder.router_function(state, structured_llm),
            {
                "rag": "rag_search",
                "chat": "chatbot",
            },
        )

        bot_graph.add_edge("rag_search", END)
        bot_graph.add_edge("chatbot", END)
        log.info("Graph nodes and edges defined.")

        memory = MemorySaver()

        # Compile the graph with a memory saver
        compiled_graph = bot_graph.compile(checkpointer=memory)
        log.info("Graph compiled successfully.")

        return compiled_graph
