import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Tuple
from src.common.logger import log
from src.bot.search_graph import GraphBuilder
from dotenv import load_dotenv
import os
import uuid


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in Hugging Face Space secrets."
    )

log.info("Initializing LangGraph chatbot graph...")
try:
    chatbot_graph = GraphBuilder.build_graph()
    log.info("Chatbot graph initialized successfully.")
except Exception as e:
    log.critical(f"Failed to initialize the chatbot graph: {e}")
    raise


# --- Gradio UI and Logic ---
with gr.Blocks(theme="soft", title="Jira Reviews ChatBot") as demo:

    # --- State Management ---
    # Create a state variable to hold a unique thread_id for each user session.
    # This is the key component for multi-user support.
    thread_id_state = gr.State()

    # --- UI Components ---
    gr.Markdown("# Jira Reviews ChatBot 🤖")
    gr.Markdown("Welcome! A new chat session has been started for you.")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
    )

    with gr.Row():
        chat_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter...",
            container=False,
            scale=4,
        )
        clear_button = gr.Button("🗑️ New Chat", variant="secondary", scale=1)

    # --- Core Functions ---
    def start_new_session():
        """
        Generates a new unique thread_id to start a fresh conversation.
        Returns the new thread_id and an empty chat history.
        """
        new_thread_id = str(uuid.uuid4())
        log.info(f"New user session started. Thread ID: {new_thread_id}")
        return new_thread_id, []

    def handle_chat(message: str, history: List[Tuple[str, str]], thread_id: str):
        """
        Main chat logic that streams the response from the LangGraph agent.
        Receives the unique thread_id from the session state.
        """
        if not thread_id:
            # This is a fallback, the demo.load should always provide a thread_id
            thread_id, history = start_new_session()

        log.info(f"Message received from thread '{thread_id}': {message}")

        # Append user message to history and yield immediately to update the UI
        history.append([message, ""])
        yield history, thread_id  # Always yield back the state

        response_stream = ""
        try:
            # Stream the response using the unique thread_id for memory
            for chunk in chatbot_graph.stream(
                {"messages": [HumanMessage(content=message.strip())]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="values",
            ):
                if isinstance(chunk["messages"][-1], AIMessage):
                    response_stream = chunk["messages"][-1].content
                    history[-1][1] = response_stream
                    yield history, thread_id

        except Exception as e:
            log.error(f"Error during chatbot stream for thread '{thread_id}': {e}")
            history[-1][1] = "Sorry, an error occurred. Please try again."
            yield history, thread_id

    # --- Event Handlers ---

    # 1. When the page loads, start a new session automatically.
    demo.load(
        fn=start_new_session,
        inputs=[],
        outputs=[
            thread_id_state,
            chatbot,
        ],  # Set the thread_id state and clear the chatbot UI
    )

    # 2. When the user submits a message...
    chat_input.submit(
        fn=handle_chat,
        # Pass the current message, history, and the unique thread_id
        inputs=[chat_input, chatbot, thread_id_state],
        # Return the updated chatbot history and the (unchanged) thread_id
        outputs=[chatbot, thread_id_state],
    )
    # ...also clear the input textbox after submission.
    chat_input.submit(fn=lambda: "", inputs=[], outputs=[chat_input])

    # 3. When the user clicks the "New Chat" button, start a new session.
    clear_button.click(
        fn=start_new_session,
        inputs=[],
        # Reset the thread_id state and clear the chatbot UI
        outputs=[thread_id_state, chatbot],
    )


if __name__ == "__main__":
    demo.launch(debug=True)
