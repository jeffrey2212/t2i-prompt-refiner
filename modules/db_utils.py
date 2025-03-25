"""Database utilities module"""
import os
import sqlite3
import datetime
import uuid
import streamlit as st
from contextlib import contextmanager

# Database file path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DB_DIR, exist_ok=True)  # Create data directory if it doesn't exist
DB_PATH = os.path.join(DB_DIR, 'app.db')

# Global flag to track initialization
_DB_INITIALIZED = False

# Database version
CURRENT_DB_VERSION = 1  # Decrement this to 1 since schema has been simplified

def get_db_connection():
    """Get a database connection with proper settings"""
    conn = sqlite3.connect(DB_PATH, timeout=20)  # Add timeout for busy waiting
    conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
    conn.execute("PRAGMA busy_timeout=10000")  # Wait up to 10 seconds if db is locked
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the SQLite database with required tables and run migrations"""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Drop existing tables to ensure clean schema
            c.execute("DROP TABLE IF EXISTS chat_messages")
            c.execute("DROP TABLE IF EXISTS prompt_history")
            c.execute("DROP TABLE IF EXISTS chat_sessions")
            c.execute("DROP TABLE IF EXISTS db_version")
            
            # Create version table first
            c.execute('''
                CREATE TABLE IF NOT EXISTS db_version (
                    version INTEGER PRIMARY KEY
                )
            ''')
            
            # Get or set initial version
            c.execute("SELECT version FROM db_version")
            result = c.fetchone()
            if not result:
                c.execute("INSERT INTO db_version (version) VALUES (1)")
                current_version = 1
            else:
                current_version = result[0]
            
            # Create required tables
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS prompt_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    original TEXT NOT NULL,
                    refined TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                )
            ''')
            
            conn.commit()
            _DB_INITIALIZED = True
            print("Database initialized successfully")
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

def get_or_create_session():
    """Get the current session ID or create a new one"""
    if "chat_session_id" not in st.session_state:
        # Create a new session ID
        session_id = str(uuid.uuid4())
        
        # Create a new session in the database
        with get_db() as conn:
            c = conn.cursor()
            now = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO chat_sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now)
            )
            conn.commit()
            st.session_state.chat_session_id = session_id
    
    return st.session_state.chat_session_id

def update_session_timestamp(session_id):
    """Update the last_updated timestamp for a session"""
    with get_db() as conn:
        c = conn.cursor()
        now = datetime.datetime.now().isoformat()
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        conn.commit()

def save_message(role, content):
    """Save a message to the database"""
    session_id = get_or_create_session()
    timestamp = datetime.datetime.now().isoformat()
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, timestamp)
        )
        
        # Update the session's last_updated timestamp
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (timestamp, session_id)
        )
        
        conn.commit()

def save_prompt_pair(original, refined):
    """Save an original and refined prompt pair to the database"""
    session_id = get_or_create_session()
    with get_db() as conn:
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        
        c.execute(
            "INSERT INTO prompt_history (session_id, original, refined, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, original, refined, timestamp)
        )
        
        # Update the session's last_updated timestamp
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (timestamp, session_id)
        )
        
        conn.commit()

def load_chat_history():
    """Load chat history from the database for the current session"""
    session_id = get_or_create_session()
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        
        messages = []
        for role, content, timestamp in c.fetchall():
            timestamp_dt = datetime.datetime.fromisoformat(timestamp)
            messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp_dt
            })
        return messages

def load_prompt_history():
    """Load prompt history from the database for the current session"""
    session_id = get_or_create_session()
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT original, refined, timestamp FROM prompt_history WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        
        prompts = []
        for original, refined, timestamp in c.fetchall():
            timestamp_dt = datetime.datetime.fromisoformat(timestamp)
            prompts.append({
                "original": original,
                "refined": refined,
                "timestamp": timestamp_dt
            })
        return prompts

def get_all_sessions():
    """Get all chat sessions from the database"""
    with get_db() as conn:
        c = conn.cursor()
        
        c.execute(
            "SELECT id, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        )
        
        sessions = []
        for session_id, created_at, last_updated in c.fetchall():
            # Get the first message to use as a title
            c.execute(
                "SELECT content FROM chat_messages WHERE session_id = ? AND role = 'user' ORDER BY created_at LIMIT 1",
                (session_id,)
            )
            first_message = c.fetchone()
            title = first_message[0][:30] + "..." if first_message and len(first_message[0]) > 30 else "New Chat"
            
            sessions.append({
                "session_id": session_id,
                "title": title,
                "created_at": datetime.datetime.fromisoformat(created_at),
                "last_updated": datetime.datetime.fromisoformat(last_updated)
            })
        return sessions

def clear_session_history(session_id=None):
    """Clear the chat history for a specific session or the current session"""
    if session_id is None:
        session_id = get_or_create_session()
    
    with get_db() as conn:
        c = conn.cursor()
        
        # Delete messages for this session
        c.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        
        # Delete prompt history for this session
        c.execute("DELETE FROM prompt_history WHERE session_id = ?", (session_id,))
        
        conn.commit()
        
        # Create a new session ID if clearing the current session
        if session_id == st.session_state.get("chat_session_id"):
            st.session_state.chat_session_id = str(uuid.uuid4())
            
            # Create a new session in the database
            c.execute(
                "INSERT INTO chat_sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (st.session_state.chat_session_id, datetime.datetime.now().isoformat(), datetime.datetime.now().isoformat())
            )
            conn.commit()

def switch_session(session_id):
    """Switch to a different chat session"""
    st.session_state.chat_session_id = session_id
    
    # Update the session's last_updated timestamp
    update_session_timestamp(session_id)

def delete_session(session_id):
    """Delete a chat session and all its messages"""
    with get_db() as conn:
        c = conn.cursor()
        
        # Delete messages for this session
        c.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        
        # Delete prompt history for this session
        c.execute("DELETE FROM prompt_history WHERE session_id = ?", (session_id,))
        
        # Delete the session itself
        c.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        
        # If we deleted the current session, create a new one
        if session_id == st.session_state.get("chat_session_id"):
            st.session_state.pop("chat_session_id", None)
            get_or_create_session()

def save_cursor(cursor):
    """Save the cursor value to the database"""
    with get_db() as conn:
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        
        # Convert cursor to string if it's not already
        if cursor is None:
            cursor_str = None
        else:
            cursor_str = str(cursor).strip()  # Add strip() to remove any whitespace
            if not cursor_str:  # Check if empty after stripping
                cursor_str = None
        
        print(f"Attempting to save cursor value: {cursor_str}")
        
        # First, ensure the app_settings table exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Check if the cursor already exists
        c.execute("SELECT value FROM app_settings WHERE key = 'last_cursor'")
        existing = c.fetchone()
        
        if existing:
            # Update existing cursor
            c.execute(
                "UPDATE app_settings SET value = ?, updated_at = ? WHERE key = 'last_cursor'",
                (cursor_str, timestamp)
            )
            print("Updated existing cursor record")
        else:
            # Insert new cursor
            c.execute(
                "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                ('last_cursor', cursor_str, timestamp)
            )
            print("Inserted new cursor record")
        
        conn.commit()
        
        # Verify the save
        c.execute("SELECT value FROM app_settings WHERE key = 'last_cursor'")
        result = c.fetchone()
        saved_value = result[0] if result else None
        print(f"Verified saved cursor value: {saved_value}")
        
        return saved_value

def load_cursor():
    """Load the cursor value from the database"""
    with get_db() as conn:
        c = conn.cursor()
        
        # First, ensure the app_settings table exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        c.execute("SELECT value FROM app_settings WHERE key = 'last_cursor'")
        result = c.fetchone()
        
        # Return the cursor value as a string if it exists
        if result and result[0] and result[0].lower() != 'none':
            cursor_value = str(result[0]).strip()  # Add strip() to remove any whitespace
            if cursor_value:  # Only return if non-empty after stripping
                print(f"Loaded cursor from database: {cursor_value}")
                return cursor_value
        
        print("No valid cursor found in database")
        return None

def clear_cursor():
    """Clear the cursor from the database"""
    with get_db() as conn:
        c = conn.cursor()
        
        try:
            c.execute("DELETE FROM app_settings WHERE key = 'last_cursor'")
            conn.commit()
            print("Cursor cleared from database")
        except Exception as e:
            print(f"Error clearing cursor: {str(e)}")
            conn.rollback()

def check_db_structure():
    """Check and print the database structure"""
    with get_db() as conn:
        c = conn.cursor()
        
        try:
            # Get table schema
            c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='app_settings'")
            schema = c.fetchone()
            if schema:
                print("app_settings table schema:", schema[0])
            else:
                print("app_settings table does not exist")
                
            # Check if the table exists and get any records
            c.execute("SELECT * FROM app_settings WHERE key = 'last_cursor'")
            cursor_record = c.fetchone()
            if cursor_record:
                print("Current cursor record:", cursor_record)
            else:
                print("No cursor record found")
                
        except Exception as e:
            print(f"Error checking database: {str(e)}")

def get_total_records_count():
    """Get the total number of records in the vector database"""
    try:
        from modules.qdrant_utils import get_qdrant_client
        client = get_qdrant_client()
        collection_info = client.get_collection("prompts")
        return collection_info.points_count
    except Exception as e:
        print(f"Error getting record count: {e}")
        return 0

# Initialize the database when this module is imported
init_db()
check_db_structure()
