import streamlit as st
from modules.qdrant_utils import store_in_vector_db
from modules.refined_data_utils import process_item

def process_and_store(data):
    """Processes and stores the fetched data."""
    if not data:
        return 0, 0, 0, 0

    processed = 0
    stored = 0
    skipped = 0
    errors = 0

    for item in data:
        try:
            # Skip None or empty items
            if not item:
                skipped += 1
                continue

            # Process item
            processed_item = process_item(item)
            if processed_item:
                # Store processed item
                if store_in_vector_db(processed_item):
                    stored += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            processed += 1

        except Exception as e:
            print(f"Error processing item: {e}")
            print(f"Raw item data: {item}")
            errors += 1

    print(f"Processing summary: Total={processed}, Stored={stored}, Skipped={skipped}, Errors={errors}")
    return processed, stored, skipped, errors
