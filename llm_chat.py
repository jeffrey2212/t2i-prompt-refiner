import os
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv

# Import database utilities
from db_utils import (
    init_db, 
    save_message, 
    save_prompt_pair, 
    load_chat_history, 
    load_prompt_history,
    clear_session_history,
    get_all_sessions,
    switch_session,
    delete_session
)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Default system message for the LLM
DEFAULT_SYSTEM_MESSAGE = """
You are an AI assistant specialized in helping users refine their text-to-image prompts.
Your goal is to help users create better prompts for image generation models like Stable Diffusion.
Provide specific suggestions to improve clarity, detail, and style in prompts.
"""

# Initialize the database
init_db()

def check_llm_server_status():
    """Check if the LLM server is available"""
    # For OpenAI API
    if openai_api_key:
        try:
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get("https://api.openai.com/v1/models", headers=headers)
            return response.status_code == 200
        except:
            return False
    return False

def check_comfyui_server_status():
    """Check if the ComfyUI server is available"""
    # Placeholder for now - will be implemented later
    # For now, just return False
    return False

def initialize_chat_state():
    """Initialize the chat state in the Streamlit session"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}
        ]

def call_openai_api(messages):
    """Call the OpenAI API with the given messages"""
    try:
        if not openai_api_key:
            return "Error: OpenAI API key not found. Please add it to your .env file."
            
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",  # Can be configured based on needs
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def display_server_status():
    """Display the server status indicators"""
    col1, col2 = st.columns(2)
    
    # LLM Server Status
    llm_status = check_llm_server_status()
    with col1:
        if llm_status:
            st.markdown("<span style='color:green'>●</span> LLM Server: Online", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>●</span> LLM Server: Offline", unsafe_allow_html=True)
    
    # ComfyUI Server Status
    comfyui_status = check_comfyui_server_status()
    with col2:
        if comfyui_status:
            st.markdown("<span style='color:green'>●</span> ComfyUI Server: Online", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>●</span> ComfyUI Server: Offline", unsafe_allow_html=True)

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

def chat_interface():
    """Main chat interface for the LLM prompt refiner"""
    # Initialize chat state
    initialize_chat_state()
    
    # Load messages from the database
    load_messages_from_db()
    
    # Display server status at the top
    display_server_status()
    
    # Main chat container
    chat_container = st.container()
    
    # Chat input at the bottom
    prompt = st.chat_input("Enter your prompt for refinement")
    
    if prompt:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Save to database
        save_message("user", prompt)
        
        # Get AI response
        response = call_openai_api(st.session_state.chat_messages)
        
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Save to database
        save_message("assistant", response)
        
        # Check if this was a prompt refinement (simple heuristic)
        if "refine" in prompt.lower():
            # Extract the refined prompt (assuming it's in the response)
            # This is a simple approach - could be improved with better parsing
            refined_prompt = response.split('\n\n')[0] if '\n\n' in response else response
            
            # Save to database
            save_prompt_pair(prompt, refined_prompt)
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] != "system":  # Don't show system messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])

def chat_sidebar():
    """Sidebar for the chat interface"""
    st.sidebar.title("Chat History")
    
    # Load chat history from the database
    chat_history = load_chat_history()
    
    # Display chat history in sidebar
    if chat_history:
        for i, entry in enumerate(chat_history):
            timestamp_str = entry["timestamp"].strftime('%H:%M')
            if entry["role"] == "user":
                with st.sidebar.expander(f"You - {timestamp_str}", expanded=False):
                    st.write(entry["content"])
            elif entry["role"] == "assistant":
                with st.sidebar.expander(f"AI - {timestamp_str}", expanded=False):
                    st.write(entry["content"])
    else:
        st.sidebar.write("No chat history yet.")
    
    # Option to clear chat history
    if st.sidebar.button("Clear Chat History"):
        # Clear the database
        clear_session_history()
        
        # Reset the session state
        st.session_state.chat_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}
        ]
        
        st.rerun()

def llm_chat_page():
    """Main function for the LLM chat page"""
    # Set up the sidebar
    chat_sidebar()
    
    # Main chat interface
    chat_interface()

if __name__ == "__main__":
    llm_chat_page()
