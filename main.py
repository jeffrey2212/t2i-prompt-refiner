import streamlit as st
import requests
import pandas as pd
import threading
import time
from qdrant_client import QdrantClient

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize session state for job tracking
if "job_status" not in st.session_state:
    st.session_state.job_status = "Idle"
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "statistics" not in st.session_state:
    st.session_state.statistics = None
if "new_images_estimate" not in st.session_state:
    st.session_state.new_images_estimate = 0
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# API and database setup
API_URL = "https://api.civitai.com/v1/images"  # Replace with actual endpoint
qdrant_client = QdrantClient(":memory:")  # Use persistent storage in production

def initialize_embedder():
    """Initialize the sentence transformer model if not already initialized"""
    if st.session_state.embedder is None:
        from sentence_transformers import SentenceTransformer
        st.session_state.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return st.session_state.embedder

# Simulated database of stored image IDs (replace with actual DB query in production)
stored_image_ids = set()  # Example: load from Qdrant or a file

# Function to fetch data from Civitai API (simplified)
def fetch_data(target_count, model_filter, rating_filter):
    headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Replace with your key
    params = {"limit": 200}  # API pagination limit
    if model_filter != "All":
        params["model"] = model_filter
    data = []
    total_fetched = 0
    page = 1
    while total_fetched < target_count:
        params["page"] = page
        response = requests.get(API_URL, headers=headers, params=params)
        if response.status_code != 200:
            break
        json_data = response.json()
        items = json_data.get("items", [])
        for item in items:
            if total_fetched >= target_count:
                break
            if item.get("rating", 0) >= rating_filter and item["id"] not in stored_image_ids:
                # Extract model information from the API response
                model_info = item.get("meta", {}).get("Model", "Unknown")
                item['model_name'] = model_info if isinstance(model_info, str) else model_info.get("name", "Unknown")
                data.append(item)
                total_fetched += 1
        if not json_data.get("metadata", {}).get("nextPage"):
            break
        page += 1
        st.session_state.progress = min(100, int((total_fetched / target_count) * 100))
    return data

# Function to process and store data
def process_and_store(data):
    # Example processing: create a DataFrame and store in Qdrant
    df = pd.DataFrame(data)
    if not df.empty:
        embedder = initialize_embedder()  # Get or initialize the embedder
        df["embedding"] = df["description"].apply(lambda x: embedder.encode(x) if x else [0]*384)
        # Store in Qdrant (simplified)
        for idx, row in df.iterrows():
            qdrant_client.upsert(
                collection_name="civitai_images",
                points=[{"id": row["id"], "vector": row["embedding"], "payload": row.to_dict()}]
            )
    # Compute statistics using the model_name field
    if not df.empty:
        stats = df.groupby("model_name").agg({
            "id": "count",
            "rating": "mean"
        }).reset_index()
        stats.columns = ["Model", "Image Count", "Average Rating"]
        stats = stats.sort_values("Image Count", ascending=False)
        st.session_state.statistics = stats

# Background job function
def background_job(target_count, model_filter, rating_filter):
    st.session_state.job_status = "Running"
    data = fetch_data(target_count, model_filter, rating_filter)
    process_and_store(data)
    st.session_state.job_status = "Completed"
    st.session_state.progress = 100

# Function to estimate new images
def check_new_images(target_count, model_filter, rating_filter):
    headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Replace with your key
    params = {"limit": 1}  # Minimal fetch to get metadata
    if model_filter != "All":
        params["model"] = model_filter
    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code == 200:
        total_items = response.json().get("metadata", {}).get("totalItems", 0)
        new_estimate = max(0, total_items - len(stored_image_ids))
        st.session_state.new_images_estimate = min(new_estimate, target_count)
    else:
        st.session_state.new_images_estimate = "Unable to estimate"

# UI Layout
st.title("Civitai Image Data Retriever")
st.write("Retrieve, process, and store image data from Civitai with customizable filters.")

# Input parameters
target_count = st.slider("Target Count", 100, 10000, 1000)
model_filter = st.selectbox("Model Filter", ["All", "Stable Diffusion", "Flux", "Pony"])
rating_filter = st.slider("Minimum Rating", 0.0, 5.0, 4.0)
like_filter = st.checkbox("Filter by Likes (e.g., >10)", False)  # Example additional filter

# Store filters in session state
st.session_state.target_count = target_count
st.session_state.model_filter = model_filter
st.session_state.rating_filter = rating_filter

# Check New Images button
if st.button("Check New Images"):
    check_new_images(target_count, model_filter, rating_filter)
    st.write(f"Estimated new images: {st.session_state.new_images_estimate}")

# Start Retrieval button
if st.button("Start Retrieval"):
    if st.session_state.job_status != "Running":
        # Start background job with current parameter values
        thread = threading.Thread(
            target=background_job,
            args=(target_count, model_filter, rating_filter)
        )
        thread.start()
        st.write("Retrieval started in the background.")
    else:
        st.write("Job already running.")

# Refresh Data button
if st.button("Refresh Data"):
    st.write("Data refreshed.")

# Display job status and progress
st.write(f"**Job Status:** {st.session_state.job_status}")
if st.session_state.job_status == "Running":
    st.progress(st.session_state.progress)

# Display statistics
if st.session_state.statistics is not None:
    st.write("**Statistics:**")
    st.dataframe(st.session_state.statistics)

# Add a rerun button for manual refresh
if st.button("↻ Refresh Status"):
    st.rerun()