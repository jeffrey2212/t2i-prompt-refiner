"""Vector database operations for prompt similarity search"""
from chromadb import Client, Settings
import chromadb
import os

def get_vector_db():
    """Get or create ChromaDB client"""
    client = Client(Settings(
        persist_directory="./data/chroma",
        anonymized_telemetry=False
    ))
    
    # Create collection if it doesn't exist
    try:
        collection = client.get_collection("prompt_examples")
    except ValueError:
        collection = client.create_collection("prompt_examples")
    
    return collection

def get_similar_prompts(prompt, model_name, k=5):
    """Get similar prompts from the vector database"""
    collection = get_vector_db()
    try:
        results = collection.query(
            query_texts=[prompt],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error querying vector database: {str(e)}")
        return []

def add_prompt_to_db(prompt, model_name, metadata=None):
    """Add a prompt to the vector database"""
    collection = get_vector_db()
    try:
        collection.add(
            documents=[prompt],
            metadatas=[{"model": model_name, **(metadata or {})}],
            ids=[f"{model_name}_{len(collection.get()['ids'])}"]
        )
        return True
    except Exception as e:
        print(f"Error adding to vector database: {str(e)}")
        return False
