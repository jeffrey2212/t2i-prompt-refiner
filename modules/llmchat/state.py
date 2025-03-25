"""Session state management module"""
import streamlit as st
from modules.db_utils import (
    save_message,
    save_prompt_pair,
    load_chat_history,
    clear_session_history,
    get_all_sessions,
    switch_session,
    delete_session
)

# Default system message for the LLM
DEFAULT_SYSTEM_MESSAGE = """
You are an AI assistant specialized in helping users refine their text-to-image prompts.
Your goal is to help users create better prompts for image generation models like Stable Diffusion.
Provide specific suggestions to improve clarity, detail, and style in prompts.
"""

def initialize_chat_state():
    """Initialize the chat state in the Streamlit session"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}
        ]

def load_messages_from_db():
    """Load messages from the database and update session state"""
    # Initialize chat state
    initialize_chat_state()
    
    # Load chat history from the database
    db_messages = load_chat_history()
    
    # If there are messages in the database, update the session state
    if db_messages:
        # Keep the system message
        system_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "system"]
        
        # Add the messages from the database
        st.session_state.chat_messages = system_messages + [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in db_messages
        ]

def save_chat_message(role, content):
    """Save a chat message to the database"""
    save_message(role, content)

def save_refined_prompt(original_prompt, refined_prompt):
    """Save a prompt refinement pair to the database"""
    save_prompt_pair(original_prompt, refined_prompt)
