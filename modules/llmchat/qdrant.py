"""Vector database operations for prompt similarity search using Qdrant"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, FieldCondition, MatchValue, Filter
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load environment variables
load_dotenv()

# Collection name and vector size
COLLECTION_NAME = "civitai_images"  # Using the same collection as images
VECTOR_SIZE = 384  # Using the same size as the image embeddings for consistency

# Initialize sentence transformer for text embeddings
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_qdrant_client():
    """Get a connection to the Qdrant vector database"""
    try:
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY not found in environment variables")
        
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            raise ValueError("QDRANT_URL not properly configured")
        
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

def get_similar_prompts(prompt, model_name, k=5):
    """Get similar prompts from the vector database"""
    try:
        print(f"\n[DEBUG] Searching Qdrant for prompts similar to: '{prompt}' for model: {model_name}")
        
        client = get_qdrant_client()
        if not client:
            return []

        # Generate embedding for the query prompt
        query_vector = model.encode(prompt).tolist()

        # Create proper filter conditions
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="model",
                    match=MatchValue(value=model_name)
                )
            ]
        )
        
        print(f"[DEBUG] Using filter: {json.dumps(query_filter.dict(), indent=2)}")

        # Search for similar prompts with model filter
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=k,
            query_filter=query_filter
        )

        # Extract prompts and metadata from search results
        similar_prompts = []
        for hit in search_result:
            if hit.payload.get("prompt"):
                prompt_data = {
                    "prompt": hit.payload["prompt"],
                    "score": hit.score,
                    "negative_prompt": hit.payload.get("negative_prompt", ""),
                    "name": hit.payload.get("name", ""),
                    "model_params": hit.payload.get("model_params", {})
                }
                similar_prompts.append(prompt_data)
        
        print(f"[DEBUG] Found {len(similar_prompts)} similar prompts")
        print("[DEBUG] First result:", json.dumps(similar_prompts[0] if similar_prompts else "None", indent=2))

        return similar_prompts

    except Exception as e:
        print(f"Error searching vector database: {e}")
        return []

def format_prompt_for_rag(similar_prompts):
    """Format similar prompts for RAG context"""
    context = []
    for idx, p in enumerate(similar_prompts, 1):
        prompt_text = f"Example {idx}:\n"
        prompt_text += f"Prompt: {p['prompt']}\n"
        if p['negative_prompt']:
            prompt_text += f"Negative Prompt: {p['negative_prompt']}\n"
        if p['model_params']:
            params = {k: v for k, v in p['model_params'].items() 
                     if k in ['steps', 'cfg_scale', 'sampler', 'seed']}
            if params:
                prompt_text += f"Parameters: {params}\n"
        prompt_text += f"Score: {p['score']:.2f}\n"
        context.append(prompt_text)
    
    formatted_context = "\n".join(context)
    print(f"\n[DEBUG] RAG Context:\n{formatted_context}\n")
    return formatted_context

def add_prompt_to_db(prompt, model_name, metadata=None):
    """This function is deprecated as we're using the existing civitai_images collection"""
    print("Warning: add_prompt_to_db is deprecated. Prompts are stored in civitai_images collection.")
