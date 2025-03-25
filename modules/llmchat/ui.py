"""Chat interface module"""
import streamlit as st
from .models import get_available_models, chat_with_model, save_refined_prompt
from .state import load_messages_from_db, save_chat_message

# Available SD models for prompt styles
SD_MODELS = ["Illustrious", "Flux.1 D", "Pony"]

def chat_interface():
    """Main chat interface"""
    st.title("T2I Prompt Refiner")
    
    # Create two columns for model selectors
    col1, col2 = st.columns(2)
    
    with col1:
        # LLM Model selector
        models = get_available_models()
        selected_model = st.selectbox(
            "LLM Model",
            options=models,
            index=0 if models else None,
            key="selected_model"
        )
    
    with col2:
        # SD PromptStyle selector
        selected_sd_model = st.selectbox(
            "SD PromptStyle",
            options=SD_MODELS,
            index=0,
            key="selected_sd_model"
        )
    
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        load_messages_from_db()
    
    # Create chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        chat_messages_display()
    
    # Chat input
    prompt = st.chat_input("Enter your prompt for refinement")
    
    if prompt and selected_model:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        save_chat_message("user", prompt)
        
        # Get AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for response in chat_with_model(st.session_state.chat_messages, selected_model, selected_sd_model):
                if 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
            
            # Update final response
            message_placeholder.markdown(full_response)
        
        # Save the complete response
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
        save_chat_message("assistant", full_response)
        
        # Handle prompt refinement and save to vector databases
        if "refine" in prompt.lower():
            refined_prompt = full_response.split('\n\n')[0] if '\n\n' in full_response else full_response
            save_refined_prompt(prompt, refined_prompt, selected_sd_model)

def chat_messages_display():
    """Display chat messages"""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
