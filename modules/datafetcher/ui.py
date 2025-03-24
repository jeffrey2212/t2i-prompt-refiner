import streamlit as st
from modules.datafetcher.fetcher import fetch_data
from modules.datafetcher.processor import process_and_store
from modules.datafetcher.utils import clear_messages, load_environment_variables
from modules.datafetcher.qdrant_utils import initialize_collection, get_total_records_count
from modules.db_utils import load_cursor

# Initialize fetch status


def fetch_prompts_ui():
    """UI for the fetch prompts functionality"""
    # Initialize session state variables
    if "target_count" not in st.session_state:
        st.session_state.target_count = 200
    if "last_cursor" not in st.session_state:
        st.session_state.last_cursor = load_cursor()
    if "collection_initialized" not in st.session_state:
        st.session_state.collection_initialized = initialize_collection()
    if "fetch_status" not in st.session_state:
        st.session_state.fetch_status = False
    if "stop_fetch" not in st.session_state:
        st.session_state.stop_fetch = False

    st.title("Fetch Prompts")

    # Create two columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Target count input
        target_count = st.number_input(
            "Number of images to fetch",
            min_value=200,
            max_value=10000,
            value=st.session_state.target_count,
            step=200,
            help="Number of images to fetch and process"
        )
        st.session_state.target_count = target_count

        # Fetch mode selection
        fetch_mode = st.radio(
            "Fetch mode",
            ["New fetch", "Continue from last"],
            index=0 if st.session_state.last_cursor is None else 1,
            help="'New fetch' starts from the beginning. 'Continue from last' continues from where you left off."
        )
        st.session_state.fetch_mode = "new" if fetch_mode == "New fetch" else "continue"

    with col2:
        # Create placeholders for status updates
        status_placeholder = st.empty()
        details_placeholder = st.empty()
        
        # Stop button (only show when fetch is running)
        stop_button = st.empty()
        
        # Fetch button and stop button container
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("Fetch Prompts", type="primary", disabled=st.session_state.fetch_status):
                st.session_state.stop_fetch = False
                continue_from_last = (fetch_mode == "Continue from last")
                check_new_images(
                    continue_from_last=continue_from_last,
                    status_placeholder=status_placeholder,
                    details_placeholder=details_placeholder,
                    summary_container=st.container()
                )
        
        with button_col2:
            if st.session_state.fetch_status:
                if st.button("Stop", type="secondary"):
                    st.session_state.stop_fetch = True
                    status_placeholder.write("⏹️ Stopping...")

    # Status section below the main controls (static)
    st.write("\n---\n")  # Add line break before status
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.write("### Database Status")
        # Show current cursor status
        if st.session_state.last_cursor:
            st.info(f"Current cursor: {st.session_state.last_cursor[:10]}...")
        else:
            st.info("No saved cursor")

    with status_col2:
        st.write("### Statistics")
        # Database stats
        try:
            total_records = get_total_records_count()
            st.metric(
                label="Total Records",
                value=f"{total_records:,}",
                help="Total number of records in the vector database"
            )
        except Exception as e:
            st.warning("Unable to fetch record count")


def check_new_images(continue_from_last=False, status_placeholder=None, details_placeholder=None, summary_container=None):
    """Check for new images and process them"""
    try:
        # Update session state
        st.session_state.fetch_status = True

        # Use the passed placeholders
        status_text = status_placeholder
        details_area = details_placeholder

        # Initialize
        status_text.write("🔄 Initializing...")
        
        clear_messages()
        st.session_state.job_status = "Running"

        # Get target count
        target_count = st.session_state.get("target_count", 200)

        # Fetch Phase
        status_text.write("🔄 Fetching Data")
        details_area.write("Setting up API connection...")

        try:
            # Fetch data with progress updates
            all_data = []
            for batch_data, progress in fetch_data(target_count, continue_from_last):
                if st.session_state.stop_fetch:
                    status_text.write("⏹️ Stopped by user")
                    st.session_state.fetch_status = False
                    return
                
                all_data.extend(batch_data)
                status_text.write(f"🔄 Fetching Data ({progress}%)")
                details_area.write(f"Fetched {len(all_data)} images so far...")
                st.rerun()

            # Processing Phase
            if all_data:
                status_text.write("🔄 Processing Images")
                total = len(all_data)

                # Process in batches for better UI updates
                processed = 0
                stored = 0
                skipped = 0
                errors = 0
                batch_size = 10

                for i in range(0, len(all_data), batch_size):
                    if st.session_state.stop_fetch:
                        status_text.write("⏹️ Stopped by user")
                        st.session_state.fetch_status = False
                        return

                    batch = all_data[i:i+batch_size]
                    batch_processed, batch_stored, batch_skipped, batch_errors = process_and_store(batch)
                    
                    processed += batch_processed
                    stored += batch_stored
                    skipped += batch_skipped
                    errors += batch_errors
                    
                    progress = min(100, int((i + batch_size) * 100 / total))
                    status_text.write(f"🔄 Processing Images ({progress}%)")
                    details_area.write(f"Processed: {processed}, Stored: {stored}, Skipped: {skipped}, Errors: {errors}")
                    st.rerun()

                # Show final summary
                with summary_container:
                    st.write("---")
                    st.write("### Processing Summary")
                    summary_column1, summary_column2 = st.columns(2)
                    with summary_column1:
                        st.write("#### Statistics")
                        st.write(f"- ✅ Stored: {stored}")
                        st.write(f"- ⏭️ Skipped: {skipped}")
                    with summary_column2:
                        st.write("#### Status")
                        st.write(f"- 📊 Total: {processed}")
                        st.write(f"- ❌ Errors: {errors}")

                if errors > 0:
                    status_text.write("❌ Completed with errors")
                else:
                    status_text.write("✅ Processing complete")
            else:
                status_text.write("❌ No data to process")

        except Exception as e:
            status_text.write(f"❌ Error: {str(e)}")
            raise e

    except Exception as e:
        if 'status_text' in locals():
            status_text.write(f"❌ Error: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
    finally:
        st.session_state.fetch_status = False
        st.session_state.stop_fetch = False


# Run the UI
if __name__ == "__main__":
    fetch_prompts_ui()