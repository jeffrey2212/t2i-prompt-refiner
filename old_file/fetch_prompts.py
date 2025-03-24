# Import statements
import os
import json
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from db_utils import save_cursor, load_cursor, clear_cursor

# Load environment variables
def load_environment_variables():
    load_dotenv()
    civitai_api_key = os.getenv("CIVITAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not all([civitai_api_key, qdrant_url, qdrant_api_key]):
        st.error("Missing required environment variables. Please check your .env file.")
        st.stop()
        
    return civitai_api_key, qdrant_url, qdrant_api_key

# Initialize Qdrant client
def initialize_qdrant_client():
    _, qdrant_url, qdrant_api_key = load_environment_variables()
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# API setup
API_URL = "https://api.civitai.com/v1/images"

# Helper functions
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

# Data processing functions
def process_item(item):
    """Process a single item"""
    qdrant_client = initialize_qdrant_client()
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

def fetch_data(target_count, continue_from_last=False):
    """Fetch data from Civitai API"""
    status = st.status("Fetching data from Civitai API...", expanded=True)
    
    civitai_api_key, _, _ = load_environment_variables()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {civitai_api_key}"
    }
    
    params = {
        "limit": 200,
        "sort": "Most Reactions",
        "period": "Month"
    }
    
    # Debug cursor state
    status.write(f"Session state cursor: {st.session_state.last_cursor}")
    status.write(f"Continue from last: {continue_from_last}")
    
    # Add cursor if continuing from last fetch
    if continue_from_last and st.session_state.last_cursor:
        cursor_value = str(st.session_state.last_cursor).strip()
        if cursor_value:
            params["cursor"] = cursor_value
            status.write(f"Using cursor: {cursor_value}")
        else:
            status.write("Cursor value was empty, starting new fetch")
    
    # Fetch data
    all_data = []
    progress_text = "Fetching images..."
    my_bar = st.progress(0, text=progress_text)
    last_cursor = None
    
    while len(all_data) < target_count:
        try:
            status.write(f"Making request with params: {params}")
            response = requests.get(API_URL, headers=headers, params=params)
            
            if response.status_code != 200:
                status.error(f"API request failed: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                status.write("No more items to fetch")
                break
                
            all_data.extend(items)
            current_progress = min(len(all_data) / target_count, 1.0)
            my_bar.progress(current_progress, text=f"{progress_text} ({len(all_data)}/{target_count})")
            status.write(f"Fetched {len(items)} items, total: {len(all_data)}")
            
            # Get next cursor
            metadata = data.get("metadata", {})
            next_cursor = metadata.get("nextCursor")
            
            # Only save cursor if we need more data
            if len(all_data) < target_count and next_cursor:
                next_cursor = str(next_cursor).strip()
                if next_cursor and next_cursor != last_cursor:  # Only update if cursor is different
                    last_cursor = next_cursor
                    saved_cursor = save_cursor(next_cursor)
                    if saved_cursor:
                        st.session_state.last_cursor = saved_cursor
                        status.write(f"New cursor saved: {saved_cursor}")
                        params["cursor"] = saved_cursor
                    else:
                        status.error("Failed to save cursor")
                        break
                else:
                    status.write("Cursor unchanged or empty, stopping fetch")
                    break
            else:
                break
            
        except Exception as e:
            status.error(f"Error in fetch: {str(e)}")
            break
    
    # Update final status
    if len(all_data) > 0:
        status.update(label=f"Fetched {len(all_data)} images", state="complete")
    else:
        status.update(label="No images fetched", state="error")
    
    # Trim to target count and return
    return all_data[:target_count]

def process_and_store(data):
    """Process and store the fetched data"""
    if not data:
        return
    
    status = st.status("Processing fetched data...", expanded=True)
    total = len(data)
    
    # Initialize progress
    progress_text = "Processing images..."
    my_bar = st.progress(0, text=progress_text)
    
    processed = 0
    stored = 0
    skipped = 0
    errors = 0
    
    try:
        for idx, item in enumerate(data):
            try:
                # Update progress
                current_progress = idx / total
                my_bar.progress(current_progress, text=f"{progress_text} ({idx}/{total})")
                
                # Process item
                processed_item = process_item(item)
                if processed_item:
                    if store_in_vector_db(processed_item):
                        stored += 1
                        status.write(f"✅ Stored: {processed_item.get('name', 'Unnamed')}")
                    else:
                        skipped += 1
                        status.write(f"⏭️ Skipped duplicate: {processed_item.get('name', 'Unnamed')}")
                
                processed += 1
                
            except Exception as e:
                errors += 1
                status.write(f"❌ Error processing item: {str(e)}")
        
        # Final progress update
        my_bar.progress(1.0, text=f"Completed processing {total} images")
        
        # Show summary
        if errors > 0:
            status.update(label="Processing complete with errors", state="error")
        else:
            status.update(label="Processing complete", state="complete")
        
        status.write("---")
        status.write("### Processing Summary")
        status.write(f"- ✅ Successfully stored: {stored}")
        status.write(f"- ⏭️ Skipped (duplicates): {skipped}")
        status.write(f"- ❌ Errors: {errors}")
        status.write(f"- 📊 Total processed: {processed}")
        
    except Exception as e:
        status.error(f"Fatal error during processing: {str(e)}")
    
    finally:
        # Update the progress bar one last time
        my_bar.progress(1.0, text=f"Processed {processed} of {total} images")

def check_new_images(continue_from_last=False):
    """Check for new images and process them"""
    try:
        # Create main container
        main_container = st.container()
        
        with main_container:
            # Create columns for status and details
            status_col, details_col = st.columns([1, 2])
            
            with status_col:
                st.write("### Status")
                status_text = st.empty()
                progress_text = st.empty()
            
            with details_col:
                st.write("### Details")
                details_area = st.empty()
        
        # Initialize
        status_text.write("🔄 Initializing...")
        clear_messages()
        st.session_state.job_status = "Running"
        
        # Get target count
        target_count = st.session_state.get("target_count", 200)
        
        # Fetch Phase
        status_text.write("🔄 Fetching Data")
        details_area.write("Setting up API connection...")
        
        # Set up API
        civitai_api_key, _, _ = load_environment_variables()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {civitai_api_key}"
        }
        
        params = {
            "limit": 200,
            "sort": "Most Reactions",
            "period": "Month"
        }
        
        # Handle cursor
        if continue_from_last and st.session_state.last_cursor:
            cursor_value = str(st.session_state.last_cursor).strip()
            if cursor_value:
                params["cursor"] = cursor_value
                details_area.write(f"Using cursor: {cursor_value}")
            else:
                details_area.write("No valid cursor, starting new fetch")
        
        # Fetch data
        all_data = []
        last_cursor = None
        fetch_log = []
        progress = 0.0
        
        while len(all_data) < target_count:
            try:
                fetch_log.append(f"Fetching: {len(all_data)}/{target_count} images")
                response = requests.get(API_URL, headers=headers, params=params)
                
                if response.status_code != 200:
                    fetch_log.append(f"❌ API Error: {response.status_code}")
                    details_area.write("\n".join(fetch_log[-5:]))  # Show last 5 logs
                    break
                
                data = response.json()
                items = data.get("items", [])
                
                if not items:
                    fetch_log.append("No more items available")
                    details_area.write("\n".join(fetch_log[-5:]))
                    break
                
                all_data.extend(items)
                progress = min(len(all_data) / target_count, 1.0)
                progress_text.write(f"Progress: {int(progress * 100)}%")
                
                # Handle cursor
                metadata = data.get("metadata", {})
                next_cursor = metadata.get("nextCursor")
                
                if len(all_data) < target_count and next_cursor:
                    next_cursor = str(next_cursor).strip()
                    if next_cursor and next_cursor != last_cursor:
                        last_cursor = next_cursor
                        saved_cursor = save_cursor(next_cursor)
                        if saved_cursor:
                            st.session_state.last_cursor = saved_cursor
                            fetch_log.append(f"✓ New cursor: {saved_cursor[:10]}...")
                            params["cursor"] = saved_cursor
                        else:
                            fetch_log.append("❌ Failed to save cursor")
                            details_area.write("\n".join(fetch_log[-5:]))
                            break
                else:
                    break
                
                details_area.write("\n".join(fetch_log[-5:]))  # Show last 5 logs
                    
            except Exception as e:
                fetch_log.append(f"❌ Error: {str(e)}")
                details_area.write("\n".join(fetch_log[-5:]))
                break
        
        # Processing Phase
        if all_data:
            status_text.write("🔄 Processing Images")
            process_log = []
            
            total = len(all_data)
            processed = 0
            stored = 0
            skipped = 0
            errors = 0
            
            for idx, item in enumerate(all_data):
                try:
                    progress = idx / total
                    progress_text.write(f"Progress: {int(progress * 100)}%")
                    
                    processed_item = process_item(item)
                    if processed_item:
                        if store_in_vector_db(processed_item):
                            stored += 1
                            process_log.append(f"✅ Stored: {processed_item.get('name', 'Unnamed')}")
                        else:
                            skipped += 1
                            process_log.append(f"⏭️ Skipped: {processed_item.get('name', 'Unnamed')}")
                    
                    processed += 1
                    details_area.write("\n".join(process_log[-5:]))  # Show last 5 logs
                    
                except Exception as e:
                    errors += 1
                    process_log.append(f"❌ Error: {str(e)}")
                    details_area.write("\n".join(process_log[-5:]))
            
            # Final progress
            progress_text.write("Completed!")
            
            # Show summary in main container
            with main_container:
                st.write("---")
                st.write("### Processing Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Statistics")
                    st.write(f"- ✅ Stored: {stored}")
                    st.write(f"- ⏭️ Skipped: {skipped}")
                with col2:
                    st.write("#### Status")
                    st.write(f"- 📊 Total: {processed}")
                    st.write(f"- ❌ Errors: {errors}")
            
            if errors > 0:
                status_text.write("❌ Completed with errors")
            else:
                status_text.write("✅ Processing complete")
            
            st.session_state.job_status = "Complete"
        else:
            status_text.write("❌ No data to process")
            st.session_state.job_status = "Error"
    
    except Exception as e:
        if 'status_text' in locals():
            status_text.write(f"❌ Error: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
        st.session_state.job_status = "Error"

def fetch_prompts_ui():
    """UI for the fetch prompts functionality"""
    # Initialize session state variables
    initialize_session_state()
    
    st.title("Fetch Prompts")
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Target count input
        target_count = st.number_input(
            "Number of images to fetch",
            min_value=200,
            max_value=10000,
            value=200,
            step=200,
            help="Number of images to fetch and process"
        )
        st.session_state.target_count = target_count
        
        # Fetch mode selection
        fetch_mode = st.radio(
            "Fetch mode",
            ["New fetch", "Continue from last"],
            index=0 if st.session_state.fetch_mode == "new" else 1,
            help="'New fetch' starts from the beginning. 'Continue from last' continues from where you left off."
        )
        st.session_state.fetch_mode = "new" if fetch_mode == "New fetch" else "continue"
        
        # Show current cursor status
        if st.session_state.last_cursor:
            st.info(f"Current cursor: {st.session_state.last_cursor[:10]}...")
        else:
            st.info("No saved cursor")
        
        # Database stats
        total_records = get_total_records_count()
        st.metric(
            label="Database Records",
            value=f"{total_records:,}",
            help="Total number of records in the vector database"
        )
    
    with col2:
        # Clear cursor button
        if st.button("Clear Cursor", type="secondary", help="Clear the saved cursor and start fresh"):
            clear_saved_cursor()
        
        # Fetch button
        if st.button("Fetch Images", type="primary"):
            continue_from_last = (fetch_mode == "Continue from last")
            check_new_images(continue_from_last=continue_from_last)
        
        # Show current job status if running
        if st.session_state.job_status == "Running":
            st.spinner("Processing in progress...")
    
    # Show any error messages
    for msg in st.session_state.messages:
        if msg[0] == "error":
            st.error(msg[1])
        elif msg[0] == "warning":
            st.warning(msg[1])
        else:
            st.info(msg[1])
    
    # Clear old messages
    st.session_state.messages = []

def initialize_collection():
    """Initialize Qdrant collection if it doesn't exist"""
    qdrant_client = initialize_qdrant_client()
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
            add_message("success", "Created Qdrant collection 'civitai_images'")
        return True
    except Exception as e:
        add_message("error", f"Failed to initialize Qdrant collection: {str(e)}")
        return False

def get_collection_stats():
    """Get statistics about the Qdrant collection"""
    qdrant_client = initialize_qdrant_client()
    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name="civitai_images")
        total_items = collection_info.vectors_count
        
        # Get sample records
        sample_records = []
        if total_items > 0:
            sample_result = qdrant_client.scroll(
                collection_name="civitai_images",
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            sample_records = sample_result[0]
        
        return {
            "total_items": total_items,
            "sample_records": sample_records
        }
    except Exception as e:
        add_message("error", f"Error getting collection stats: {str(e)}")
        return None

def verify_qdrant_connection():
    """Verify connection to Qdrant"""
    qdrant_client = initialize_qdrant_client()
    try:
        # Try to list collections to verify connection
        qdrant_client.get_collections()
        return True, "Qdrant connection verified"
    except Exception as e:
        return False, f"Qdrant verification failed: {str(e)}"

def delete_non_target_models():
    """Delete records from Qdrant that don't match target models"""
    qdrant_client = initialize_qdrant_client()
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
            # Delete if not in target models (including 'Unknown')
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

def display_vector_db_samples():
    """Display sample records from the vector database"""
    try:
        qdrant_client = initialize_qdrant_client()
        
        # Get a sample of records from Qdrant
        search_result = qdrant_client.scroll(
            collection_name="civitai_images",
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        if not search_result[0]:
            st.warning("No records found in the database.")
            return
        
        # Display the samples
        st.subheader("Sample Records")
        
        for i, point in enumerate(search_result[0]):
            with st.expander(f"Record {i+1} - ID: {point.id}"):
                payload = point.payload
                
                # Display image if URL is available
                if "url" in payload and payload["url"]:
                    st.image(payload["url"], width=300)
                
                # Display metadata
                st.write("**Base Model:**", payload.get("baseModel", "Unknown"))
                
                # Display prompt
                meta = payload.get("meta", {})
                if meta and "prompt" in meta:
                    st.text_area("Prompt", meta["prompt"], height=100)
                
                # Display other metadata
                st.write("**Full Metadata:**")
                st.json(payload)
    
    except Exception as e:
        st.error(f"Error displaying vector DB samples: {str(e)}")

def get_total_records_count():
    """Get the total number of records in the Qdrant database"""
    try:
        qdrant_client = initialize_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name="civitai_images")
        return collection_info.points_count
    except Exception as e:
        st.error(f"Error getting record count: {str(e)}")
        return 0

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
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
    if "target_count" not in st.session_state:
        st.session_state.target_count = 200
    if "fetch_mode" not in st.session_state:
        st.session_state.fetch_mode = "new"
    
    # Always check for cursor in database
    if "last_cursor" not in st.session_state:
        db_cursor = load_cursor()
        st.session_state.last_cursor = db_cursor
        if db_cursor:
            # Only add message if it's a new session (no messages yet)
            if not st.session_state.messages:
                add_message("info", f"Loaded cursor from database: {db_cursor[:10]}...")
                print(f"Loaded cursor into session state: {db_cursor}")

def clear_saved_cursor():
    """Clear the saved cursor from both session state and database"""
    try:
        # Clear from session state
        st.session_state.last_cursor = None
        
        # Clear from database
        clear_cursor()
        
        # Add message
        add_message("info", "Cursor cleared successfully")
        print("Cursor cleared from both session state and database")
        
        # Reset progress
        st.session_state.progress = 0
        
    except Exception as e:
        add_message("error", f"Error clearing cursor: {str(e)}")
        print(f"Error clearing cursor: {str(e)}")
