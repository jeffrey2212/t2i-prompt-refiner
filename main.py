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
if "last_cursor" not in st.session_state:
    st.session_state.last_cursor = None
if "fetch_mode" not in st.session_state:
    st.session_state.fetch_mode = "new"  # "new" or "continue"
if "target_models" not in st.session_state:
    st.session_state.target_models = ["Illustrious", "Flux.1 D", "Pony"]

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
        # Skip if item is None or missing required fields
        if not item or not isinstance(item, dict) or 'id' not in item:
            add_message("warning", "Skipping invalid item (missing ID)")
            return None
            
        item_id = item["id"]
        
        # Check if this item already exists in Qdrant
        search_result = qdrant_client.scroll(
            collection_name="civitai_images",
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="id",
                        match=rest.MatchValue(
                            value=item_id
                        )
                    )
                ]
            ),
            limit=1
        )
        
        # Skip if already exists
        if search_result[0]:  # If any results found
            add_message("info", f"Item {item_id} already exists, skipping")
            return None
            
        # Get the meta data safely
        meta = item.get("meta", {})
        if not meta or not isinstance(meta, dict):
            add_message("warning", f"Item {item_id} has invalid meta data, skipping")
            return None
            
        # Check for required fields
        prompt = meta.get("prompt", "").strip()
        if not prompt:
            add_message("warning", f"Item {item_id} has no prompt, skipping")
            return None
            
        # Check for base model
        base_model = meta.get("baseModel", "Unknown")
        if base_model not in st.session_state.target_models:
            add_message("info", f"Item {item_id} has non-target model {base_model}, skipping")
            return None
            
        # Generate embedding
        model = get_model()
        embedding = model.encode(prompt)
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name="civitai_images",
            points=[
                rest.PointStruct(
                    id=item_id,
                    vector=embedding.tolist(),
                    payload={
                        "id": item_id,
                        "url": item.get("url", ""),
                        "baseModel": base_model,
                        "meta": meta
                    }
                )
            ]
        )
        
        add_message("success", f"Stored item {item_id}")
        st.session_state.stored_image_ids.add(item_id)
        return item
            
    except Exception as e:
        add_message("error", f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
        return None

def process_and_save_refined_data(data, output_file="processed_civitai_data.json"):
    """Process the raw data and save relevant information to a new JSON file"""
    # Check if data is valid
    if not data or not isinstance(data, list):
        add_message("error", "Invalid data format received")
        return []
        
    processed_data = []
    
    for item in data:
        # Skip invalid items
        if not item or not isinstance(item, dict):
            continue
            
        # Get meta data safely
        meta = item.get("meta", {})
        if not meta or not isinstance(meta, dict):
            continue
            
        # Skip items without prompts
        if not meta.get("prompt"):
            continue
            
        # Get base model safely
        base_model = meta.get("baseModel", "Unknown")
        
        # Skip if not in target models
        if base_model not in st.session_state.target_models:
            continue
            
        # Ensure item has an ID
        if "id" not in item:
            continue
            
        processed_item = {
            "id": item["id"],
            "url": item.get("url", ""),
            "baseModel": base_model,
            "meta": meta
        }
        processed_data.append(processed_item)
    
    # Save to JSON file (commented out but preserved)
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    return processed_data

def process_and_store(data):
    """Process and store items in Qdrant"""
    try:
        # Check if data is valid
        if not data or not isinstance(data, list):
            add_message("error", "No valid data to process")
            return
            
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
                add_message("error", f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
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

def fetch_data(target_count, continue_from_last=False):
    """Fetch data from Civitai API"""
    headers = {"Authorization": f"Bearer {civitai_api_key}"}
    
    data = []
    total_fetched = 0
    cursor = st.session_state.last_cursor if continue_from_last and st.session_state.last_cursor else None
    
    while total_fetched < target_count:
        params = {
            "limit": 200,  # Max allowed by API
            "sort": "Most Reactions",
            "period": "Month"
        }
        
        if cursor:
            params["cursor"] = cursor
            
        try:
            response = requests.get(API_URL, headers=headers, params=params)
            if response.status_code != 200:
                add_message("error", f"API request failed with status {response.status_code}")
                break
                
            json_data = response.json()
            items = json_data.get("items", [])
            
            if not items:
                add_message("info", "No more items available from API")
                break
                
            for item in items:
                if item and isinstance(item, dict) and item not in data:  # Avoid duplicates in the current batch
                    data.append(item)
                    total_fetched += 1
                    
            st.session_state.progress = total_fetched / target_count
            add_message("info", f"Fetched {total_fetched} of {target_count} images")
            
            metadata = json_data.get("metadata", {})
            cursor = metadata.get("nextCursor")
            
            # Save the cursor for next time
            st.session_state.last_cursor = cursor
            
            if not cursor:
                add_message("info", "Reached end of available data")
                break
        except Exception as e:
            add_message("error", f"Error fetching data: {str(e)}")
            break
    
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

def delete_non_target_models():
    """Delete records from Qdrant that don't match target models"""
    try:
        # First, get all records
        results = qdrant_client.scroll(
            collection_name="civitai_images",
            limit=1000,  # Get in batches of 1000
            with_payload=True,
            with_vectors=False
        )
        
        records = results[0]
        points_to_delete = []
        
        # Find records to delete
        for record in records:
            base_model = record.payload.get("baseModel", "Unknown")
            if base_model not in st.session_state.target_models:
                points_to_delete.append(record.id)
        
        # Delete points if any found
        if points_to_delete:
            qdrant_client.delete(
                collection_name="civitai_images",
                points_selector=rest.PointIdsList(points=points_to_delete)
            )
            add_message("success", f"Deleted {len(points_to_delete)} records with non-target models")
        else:
            add_message("info", "No non-target model records found to delete")
            
        return len(points_to_delete)
    except Exception as e:
        add_message("error", f"Error deleting non-target models: {str(e)}")
        return 0

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
    
    fetch_mode = st.radio(
        "Fetch mode",
        options=["New data", "Continue from last"],
        index=0,
        help="'New data' starts from the beginning, 'Continue from last' uses the saved cursor"
    )
    st.session_state.fetch_mode = "continue" if fetch_mode == "Continue from last" else "new"
    
    # Target models selection
    target_models = st.multiselect(
        "Target models",
        options=["Illustrious", "Flux.1 D", "Pony", "SDXL 1.0", "SD 1.5", "NoobAI"],
        default=["Illustrious", "Flux.1 D", "Pony"],
        help="Only process images from these models"
    )
    st.session_state.target_models = target_models
    
    if st.button("Check New Images"):
        clear_messages()
        st.session_state.job_status = "Checking"
        st.session_state.last_action = "check"
        check_new_images(target_count)

with col2:
    if st.session_state.new_images_estimate != 0:
        st.write(f"Estimated new images: {st.session_state.new_images_estimate}")
    
    if st.session_state.last_cursor:
        st.info(f"Saved cursor available for continuing")
    
    col2a, col2b = st.columns(2)
    
    with col2a:
        if st.button("Start Processing"):
            clear_messages()
            st.session_state.job_status = "Running"
            st.session_state.last_action = "process"
            try:
                continue_from_last = st.session_state.fetch_mode == "continue"
                data = fetch_data(target_count, continue_from_last)
                process_and_store(data)
                st.session_state.job_status = "Complete"
                add_message("success", "Processing completed successfully!")
            except Exception as e:
                add_message("error", f"Error in processing: {str(e)}")
                st.session_state.job_status = "Error"
            st.rerun()
    
    with col2b:
        if st.button("Delete Non-Target Models", type="secondary"):
            clear_messages()
            st.session_state.job_status = "Deleting"
            st.session_state.last_action = "delete"
            try:
                deleted_count = delete_non_target_models()
                st.session_state.job_status = "Complete"
                if deleted_count > 0:
                    add_message("success", f"Successfully deleted {deleted_count} non-target model records")
                else:
                    add_message("info", "No non-target model records to delete")
            except Exception as e:
                add_message("error", f"Error during deletion: {str(e)}")
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