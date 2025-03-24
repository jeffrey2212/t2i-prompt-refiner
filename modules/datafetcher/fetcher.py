import requests
import os
from dotenv import load_dotenv
from modules.db_utils import save_cursor, load_cursor, clear_cursor

# Load environment variables
load_dotenv()

def fetch_data(target_count, continue_from_last=False):
    """Fetch data from the Civitai API"""
    try:
        api_key = os.getenv("CIVITAI_API_KEY")
        if not api_key:
            raise ValueError("CIVITAI_API_KEY not found in environment variables")

        # Get cursor if continuing from last
        cursor = None
        if continue_from_last:
            cursor = load_cursor()

        # Initialize variables
        total_fetched = 0
        page_size = min(100, target_count)  # Max 100 per request

        while total_fetched < target_count:
            # Prepare request
            url = "https://civitai.com/api/v1/images"
            params = {
                "limit": page_size,
                "sort": "Most Reactions",
                "period": "Month"
            }
            if cursor:
                params["cursor"] = cursor

            headers = {"Authorization": f"Bearer {api_key}"}

            # Make request
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Process response
            items = data.get("items", [])
            if not items:
                break

            # Update cursor for next request
            metadata = data.get("metadata", {})
            cursor = metadata.get("nextCursor")
            if cursor:
                save_cursor(cursor)

            # Calculate progress
            total_fetched += len(items)
            progress = min(100, int(total_fetched * 100 / target_count))

            # Yield the batch and progress
            yield items, progress

            if not cursor:
                break

    except Exception as e:
        print(f"Error fetching data: {e}")
        raise
