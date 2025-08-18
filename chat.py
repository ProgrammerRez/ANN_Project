import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --- Streamlit App Setup ---
st.set_page_config(page_title="Gemini Chatbot", layout="wide")
st.title("💬 Chat with Gemini (Modern LCEL Version)")

with st.sidebar:
    st.markdown("## Configuration")
    api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password")
    st.markdown("---")
    st.markdown("""
    This chatbot uses the modern LangChain Expression Language (LCEL) with `RunnableWithMessageHistory` to hold a conversation. 
    Chat history is managed directly within Streamlit's session state.
    """)

# --- LangChain Conversation Setup ---

# Set up memory
# StreamlitChatMessageHistory will automatically store and retrieve messages from st.session_state
# The key "langchain_messages" is used to store the messages in the session state.
history = StreamlitChatMessageHistory(key="langchain_messages")

# Only proceed if the API key is provided
if api_key:
    # Configure the model
    # Note: 'gemini-2.5-pro' is used for compatibility. 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=1.0,
        # The convert_system_message_to_human warning is resolved by using a modern prompt structure
    )
    # Define the prompt template
    # This now uses a MessagesPlaceholder to dynamically insert the chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty and helpful assistant who explains things in simple terms. Always keep answers clear and engaging."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    # Create the main runnable chain
    chain = prompt | llm
    # Wrap the chain with RunnableWithMessageHistory
    # This class is the modern replacement for ConversationChain
    # It automatically handles loading history from and saving it to the provided `history` object
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,  # A function that returns the history object
        input_messages_key="input",
        history_messages_key="history",
    )
    # --- Chat Interface ---
    # Display past messages
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)
    # Chat input field
    user_message = st.chat_input("Type your message...")
    if user_message:
        # Display the user's message
        st.chat_message("User").write(user_message)
        # Stream the response
        with st.chat_message("Assistant"):
            with st.spinner('Thinking', show_time=True):
                # The config dictionary is crucial for streaming and session management
                config = {"configurable": {"session_id": "any_string_here"}}
                
            # Use st.write_stream to display the streaming output
            # The stream method now yields message chunks, so we access their content
            response = st.write_stream(
                (chunk.content for chunk in chain_with_history.stream({"input": user_message}, config))
            )

else:
    st.warning("Please enter your Gemini API Key in the sidebar to start chatting.")