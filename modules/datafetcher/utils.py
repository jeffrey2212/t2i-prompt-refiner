import streamlit as st
import os
from dotenv import load_dotenv
from modules.db_utils import get_db_connection

def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    civitai_api_key = os.getenv("CIVITAI_API_KEY")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    return civitai_api_key, qdrant_api_key

def clear_messages():
    st.session_state.messages = []

def save_cursor(cursor):
    """Save the cursor value to the database"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO app_settings (setting_name, setting_value)
            VALUES (?, ?) ON CONFLICT(setting_name) DO UPDATE SET setting_value = ?
        """, ("last_cursor", cursor, cursor))
        conn.commit()
        conn.close()
        return cursor
    except Exception as e:
        print(f"Error saving cursor: {e}")
        return None
