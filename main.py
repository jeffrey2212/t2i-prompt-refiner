import streamlit as st
import requests
import pandas as pd
import threading
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()
civitai_api_key = os.getenv("CIVITAI_API_KEY")

# Initialize embedder globally
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize session state for job tracking
if "job_status" not in st.session_state:
    st.session_state.job_status = "Idle"
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "statistics" not in st.session_state:
    st.session_state.statistics = None
if "new_images_estimate" not in st.session_state:
    st.session_state.new_images_estimate = 0

# API and database setup
API_URL = "https://api.civitai.com/v1/images"  
qdrant_client = QdrantClient(":memory:")  

stored_image_ids = set()  

def save_to_json(data, filename="civitai_data.json"):
    """Save the fetched data to a JSON file"""
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
            break
            
        json_data = response.json()
        items = json_data.get("items", [])
        
        for item in items:
            if total_fetched >= target_count:
                break
            if item and item.get("id"):
                data.append(item)
                total_fetched += 1
                
        st.session_state.progress = min(100, int((total_fetched / target_count) * 100))
        
        metadata = json_data.get("metadata", {})
        cursor = metadata.get("nextCursor")
        if not cursor:
            break
            
    save_to_json(data)
    return data

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
    
    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    return processed_data

def process_and_store(data):
    # First, process and save the refined data
    processed_data = process_and_save_refined_data(data)
    
    # Then create DataFrame for statistics
    df = pd.DataFrame(processed_data)
    if not df.empty:
        # Create embeddings for all items since we know they all have prompts now
        df["embedding"] = [embedder.encode(item["meta"]["prompt"]) for item in processed_data]
        
        for idx, row in df.iterrows():
            qdrant_client.upsert(
                collection_name="civitai_images",
                points=[{"id": row["id"], "vector": row["embedding"], "payload": row.to_dict()}]
            )
        
        # Update statistics based on baseModel
        stats = df.groupby("baseModel").agg({
            "id": "count"
        }).reset_index()
        stats.columns = ["Model", "Image Count"]
        stats = stats.sort_values("Image Count", ascending=False)
        st.session_state.statistics = stats

def background_job(target_count):
    st.session_state.job_status = "Running"
    data = fetch_data(target_count)
    process_and_store(data)
    st.session_state.job_status = "Completed"
    st.session_state.progress = 100

def check_new_images(target_count):
    headers = {"Authorization": f"Bearer {civitai_api_key}"}  
    params = {"limit": 1}  
    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code == 200:
        total_items = response.json().get("metadata", {}).get("totalItems", 0)
        new_estimate = max(0, total_items - len(stored_image_ids))
        st.session_state.new_images_estimate = min(new_estimate, target_count)
    else:
        st.session_state.new_images_estimate = "Unable to estimate"

st.title("Civitai Image Data Retriever")
st.write("Retrieve, process, and store image data from Civitai.")

col1, col2 = st.columns(2)

with col1:
    target_count = st.number_input("Number of Images to Fetch (must be divisible by 200)", 
                                 min_value=200,
                                 max_value=10000,
                                 value=200,
                                 step=200)

if st.button("Check New Images"):
    check_new_images(target_count)
    st.write(f"Estimated new images: {st.session_state.new_images_estimate}")

if st.button("Start Retrieval"):
    if st.session_state.job_status != "Running":
        thread = threading.Thread(
            target=background_job,
            args=(target_count,)
        )
        thread.start()
        st.write("Retrieval started in the background.")
    else:
        st.write("Job already running.")

st.write(f"**Job Status:** {st.session_state.job_status}")
if st.session_state.job_status == "Running":
    st.progress(st.session_state.progress)

if st.session_state.statistics is not None:
    st.write("**Statistics:**")
    st.dataframe(st.session_state.statistics)

if st.button("↻ Refresh Status"):
    st.rerun()