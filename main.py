from modules.datafetcher.ui import fetch_prompts_ui
import streamlit as st
from modules.db_utils import init_db

def main():
    st.set_page_config(layout="wide")
    # Initialize the database
    init_db()

    # Sidebar setup
    with st.sidebar:
        st.title("T2I Prompt Refiner")
        app_mode = st.selectbox("Choose the app mode",
                                 ["Fetch Prompts", "LLM Chat"])

    if app_mode == "Fetch Prompts":
        fetch_prompts_ui()
    elif app_mode == "LLM Chat":
        st.write("LLM Chat feature coming soon!")

if __name__ == "__main__":
    main()