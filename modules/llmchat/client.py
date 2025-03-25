"""Ollama client configuration module"""
from ollama import Client

# Initialize Ollama client
client = Client(host='http://jeffaiserver:11434')

def check_server_status():
    """Check if the Ollama server is available"""
    try:
        client.list()
        return True
    except:
        return False
