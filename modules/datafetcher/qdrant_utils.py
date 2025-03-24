from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CreateCollection, PointStruct
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Collection name constant
COLLECTION_NAME = "civitai_images"
VECTOR_SIZE = 384  # Size of the vector for storing image embeddings

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

def store_in_vector_db(item):
    """Store an item in the vector database"""
    try:
        client = get_qdrant_client()
        if not client:
            raise ValueError("Could not establish connection to Qdrant")

        # Generate a placeholder vector (replace with actual embedding later)
        vector = np.random.rand(VECTOR_SIZE).tolist()

        # Create point with payload
        point = PointStruct(
            id=item["id"],
            vector=vector,
            payload={
                "name": item["name"],
                "model": item["model"],
                "prompt": item["prompt"],
                "negative_prompt": item["negative_prompt"],
                "image_url": item["image_url"],
                "width": item["width"],
                "height": item["height"],
                "nsfw": item["nsfw"],
                "nsfw_level": item["nsfw_level"],
                "post_id": item["post_id"],
                "username": item["username"],
                "reaction_count": item["reaction_count"],
                "comment_count": item["comment_count"],
                "created_at": item["created_at"],
                "hash": item["hash"]
            }
        )

        # Upsert the point into the collection
        operation = client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return operation.status == "completed"

    except Exception as e:
        print(f"Error storing in vector DB: {e}")
        return False

def initialize_collection():
    """Initialize the Qdrant collection"""
    try:
        client = get_qdrant_client()
        if not client:
            raise ValueError("Could not establish connection to Qdrant")

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == COLLECTION_NAME for col in collections)

        if not exists:
            # Create collection with vector configuration
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {COLLECTION_NAME} created successfully")
        else:
            print(f"Collection {COLLECTION_NAME} already exists")

        return True

    except Exception as e:
        print(f"Error initializing collection: {e}")
        return False

def get_total_records_count():
    """Get the total number of records in the vector database"""
    try:
        client = get_qdrant_client()
        if not client:
            raise ValueError("Could not establish connection to Qdrant")

        collection_info = client.get_collection(COLLECTION_NAME)
        return collection_info.points_count

    except Exception as e:
        print(f"Error getting record count: {e}")
        return 0
