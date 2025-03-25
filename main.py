from modules.datafetcher.ui import fetch_prompts_ui
import streamlit as st
from modules.db_utils import init_db
from modules.llmchat.ui import chat_interface

def main():
    st.set_page_config(layout="wide")
    # Initialize the database
    init_db()

    app_mode = "LLM Chat"
    # Sidebar setup
    with st.sidebar:
        st.title("T2I Prompt Refiner")
        app_mode = st.selectbox("Choose the app mode",
                                 ["LLM Chat", "Fetch Prompts"],
                                 index=0)
        

    if app_mode == "Fetch Prompts":
        fetch_prompts_ui()
    elif app_mode == "LLM Chat":
        chat_interface()

if __name__ == "__main__":
    main()