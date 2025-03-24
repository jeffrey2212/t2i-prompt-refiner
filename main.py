import os
import json
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# Initialize session state
if "job_status" not in st.session_state:
    st.session_state.job_status = "Idle"
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "statistics" not in st.session_state:
    st.session_state.statistics = None
if "new_images_estimate" not in st.session_state:
    st.session_state.new_images_estimate = 0
if "stored_image_ids" not in st.session_state:
    st.session_state.stored_image_ids = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "_model" not in st.session_state:
    st.session_state._model = None

# Load environment variables
load_dotenv()
civitai_api_key = os.getenv("CIVITAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not all([civitai_api_key, qdrant_url, qdrant_api_key]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# API and database setup
API_URL = "https://api.civitai.com/v1/images"
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def initialize_collection():
    """Initialize Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == "civitai_images" for c in collections)
        
        if not exists:
            qdrant_client.create_collection(
                collection_name="civitai_images",
                vectors_config=rest.VectorParams(
                    size=384,  # BGE-small embedding size
                    distance=rest.Distance.COSINE
                )
            )
            add_message("success", "Created Qdrant collection: civitai_images")
    except Exception as e:
        add_message("error", f"Failed to initialize Qdrant collection: {str(e)}")

def get_model():
    """Get or initialize the SentenceTransformer model"""
    if st.session_state._model is None:
        try:
            st.session_state._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception as e:
            add_message("error", f"Failed to initialize SentenceTransformer model: {str(e)}")
    return st.session_state._model

def add_message(msg_type, msg_text):
    """Add a message to session state, avoiding duplicates"""
    if not any(m[1] == msg_text for m in st.session_state.messages):
        st.session_state.messages.append((msg_type, msg_text))

def clear_messages():
    """Clear all messages from session state"""
    st.session_state.messages = []

def process_item(item):
    """Process a single item"""
    try:
        if not item["meta"]["prompt"]:
            return None
            
        # Check if item already exists in Qdrant
        search_result = qdrant_client.scroll(
            collection_name="civitai_images",
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="id",
                        match=rest.MatchValue(value=item["id"])
                    )
                ]
            ),
            limit=1
        )
        
        # Skip if already exists
        if search_result[0]:  # If any results found
            add_message("info", f"Item {item['id']} already exists, skipping")
            return None
            
        # Generate embedding
        model = get_model()
        embedding = model.encode(item["meta"]["prompt"])
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name="civitai_images",
            points=[{
                "id": item["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "id": item["id"],
                    "url": item["url"],
                    "baseModel": item["baseModel"],
                    "meta": item["meta"],
                    "stats": item["stats"]
                }
            }]
        )
        
        add_message("success", f"Stored item {item['id']}")
        st.session_state.stored_image_ids.add(item["id"])
        return item
            
    except Exception as e:
        add_message("error", f"Error processing item {item['id']}: {str(e)}")
        return None

def process_and_save_refined_data(data, output_file="processed_civitai_data.json"):
    """Process the raw data and save relevant information to a new JSON file"""
    processed_data = []
    
    for item in data:
        meta = item.get("meta") or {}
        prompt = meta.get("prompt", "").strip()
        
        # Skip items without prompts
        if not prompt:
            continue
            
        processed_item = {
            "id": item.get("id"),
            "url": item.get("url"),
            "baseModel": item.get("baseModel", "Unknown"),
            "meta": {
                "prompt": prompt,
                "negativePrompt": meta.get("negativePrompt", "").strip(),
                "seed": meta.get("seed"),
                "steps": meta.get("steps"),
                "sampler": meta.get("sampler"),
                "cfgScale": meta.get("cfgScale")
            },
            "stats": item.get("stats", {})
        }
        processed_data.append(processed_item)
    
    # Save to JSON file (commented out but preserved)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    return processed_data

def process_and_store(data):
    """Process and store items in Qdrant"""
    try:
        # First, process and save the refined data
        processed_data = process_and_save_refined_data(data)
        
        if not processed_data:
            add_message("error", "No valid data to process")
            return
            
        # Filter out items we've already processed
        new_items = [item for item in processed_data if item["id"] not in st.session_state.stored_image_ids]
        
        if not new_items:
            add_message("info", "No new items to process")
            st.session_state.progress = 1.0
            return
            
        add_message("info", f"Processing {len(new_items)} new items")
        
        # Process items one by one
        results = []
        total = len(new_items)
        
        for i, item in enumerate(new_items, 1):
            try:
                result = process_item(item)
                if result:
                    results.append(result)
                st.session_state.progress = i / total
                
            except Exception as e:
                add_message("error", f"Error processing item {item['id']}: {str(e)}")
                continue
        
        # Update statistics immediately if we have results
        if results:
            df = pd.DataFrame(results)
            stats = df.groupby("baseModel").agg({
                "id": "count"
            }).reset_index()
            stats.columns = ["Model", "Image Count"]
            stats = stats.sort_values("Image Count", ascending=False)
            st.session_state.statistics = stats
            add_message("success", f"Successfully processed {len(results)} items")
        else:
            add_message("warning", "No items were successfully processed")
        
        return results
        
    except Exception as e:
        add_message("error", f"Error in process_and_store: {str(e)}")
        raise

def fetch_data(target_count):
    headers = {"Authorization": f"Bearer {civitai_api_key}"}
    params = {
        "limit": 200,
        "sort": "Most Reactions",
        "period": "Month"
    }
    
    data = []
    total_fetched = 0
    cursor = None
    
    while total_fetched < target_count:
        if cursor:
            params["cursor"] = cursor
            
        response = requests.get(API_URL, headers=headers, params=params)
        if response.status_code != 200:
            add_message("error", f"API request failed with status {response.status_code}")
            break
            
        json_data = response.json()
        items = json_data.get("items", [])
        
        for item in items:
            if total_fetched >= target_count:
                break
            if item and item.get("id"):
                data.append(item)
                total_fetched += 1
                
        st.session_state.progress = total_fetched / target_count
        add_message("info", f"Fetched {total_fetched} of {target_count} images")
        
        metadata = json_data.get("metadata", {})
        cursor = metadata.get("nextCursor")
        if not cursor:
            break
            
    # Save to JSON file (commented out but preserved)
    # save_to_json(data)
    return data

def save_to_json(data, filename="civitai_data.json"):
    """Save the fetched data to a JSON file"""
    # with open(filename, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

def check_new_images(target_count):
    headers = {"Authorization": f"Bearer {civitai_api_key}"}  
    params = {"limit": 1}  
    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code == 200:
        total_items = response.json().get("metadata", {}).get("totalItems", 0)
        new_estimate = max(0, total_items - len(st.session_state.stored_image_ids))
        st.session_state.new_images_estimate = min(new_estimate, target_count)
    else:
        st.session_state.new_images_estimate = "Unable to estimate"

def get_collection_stats():
    """Get statistics about the Qdrant collection"""
    try:
        collection_info = qdrant_client.get_collection("civitai_images")
        points_count = collection_info.points_count
        
        # Get sample records if there are any points
        sample_records = []
        if points_count > 0:
            # Try to get up to 5 records
            results = qdrant_client.scroll(
                collection_name="civitai_images",
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            sample_records = results[0] if results else []
        
        return {
            "total_items": points_count,
            "sample_records": sample_records
        }
    except Exception as e:
        add_message("error", f"Failed to get collection stats: {str(e)}")
        return None

def verify_qdrant_connection():
    """Verify Qdrant connection and collection status"""
    try:
        # Check if we can connect to Qdrant
        qdrant_client.get_collections()
        
        # Try to get collection info
        collection = qdrant_client.get_collection("civitai_images")
        
        # Try to perform a simple search
        results = qdrant_client.scroll(
            collection_name="civitai_images",
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        return True, "Qdrant connection and collection verified"
    except Exception as e:
        return False, f"Qdrant verification failed: {str(e)}"

# Verify Qdrant connection at startup
qdrant_ok, qdrant_msg = verify_qdrant_connection()
if not qdrant_ok:
    st.error(qdrant_msg)
    st.stop()
else:
    add_message("success", qdrant_msg)

# Initialize Qdrant collection
initialize_collection()

# Main UI
st.title("Civitai Data Processor")

# Statistics section
if st.session_state.statistics is not None:
    st.header("Statistics")
    st.dataframe(st.session_state.statistics)

# Controls section
st.header("Controls")
col1, col2 = st.columns(2)

with col1:
    target_count = st.number_input(
        "Number of images to retrieve",
        min_value=200,
        max_value=10000,
        value=200,
        step=200,
        help="Must be divisible by 200 due to API pagination"
    )
    if st.button("Check New Images"):
        clear_messages()
        st.session_state.job_status = "Checking"
        st.session_state.last_action = "check"
        check_new_images(target_count)

with col2:
    if st.session_state.new_images_estimate != 0:
        st.write(f"Estimated new images: {st.session_state.new_images_estimate}")
        
    if st.button("Start Processing"):
        clear_messages()
        st.session_state.job_status = "Running"
        st.session_state.last_action = "process"
        try:
            data = fetch_data(target_count)
            process_and_store(data)
            st.session_state.job_status = "Complete"
            add_message("success", "Processing completed successfully!")
        except Exception as e:
            add_message("error", f"Error in processing: {str(e)}")
            st.session_state.job_status = "Error"
        st.rerun()

# Status section
st.header("Status")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Job Status:** {st.session_state.job_status}")
    if st.session_state.job_status == "Running":
        st.progress(st.session_state.progress, "Progress")

with col2:
    # Get and display collection stats
    stats = get_collection_stats()
    if stats:
        st.write(f"**Items in Database:** {stats['total_items']:,}")
        if stats['sample_records']:
            with st.expander("View Sample Records"):
                for record in stats['sample_records']:
                    st.write("---")
                    st.write(f"**ID:** {record.payload['id']}")
                    st.write(f"**Base Model:** {record.payload['baseModel']}")
                    st.write(f"**Prompt:** {record.payload['meta']['prompt'][:200]}...")
                    if record.payload['url']:
                        st.image(record.payload['url'], width=200)
        else:
            st.warning("No records found in the database")
            
    if st.button("Refresh Status"):
        if st.session_state.last_action != "refresh":
            clear_messages()
        st.session_state.last_action = "refresh"
        st.rerun()

# Messages section
if st.session_state.messages:
    st.header("Messages")
    for msg_type, msg_text in st.session_state.messages:
        if msg_type == "error":
            st.error(msg_text)
        elif msg_type == "success":
            st.success(msg_text)
        elif msg_type == "info":
            st.info(msg_text)