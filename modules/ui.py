import streamlit as st
from modules.fetcher import fetch_data
from modules.processor import process_and_store
from modules.utils import clear_messages, load_environment_variables
from modules.qdrant_utils import initialize_collection, get_total_records_count

def fetch_prompts_ui():
    """UI for the fetch prompts functionality"""
    # Initialize session state variables
    if "target_count" not in st.session_state:
        st.session_state.target_count = 200
    if "fetch_mode" not in st.session_state:
        st.session_state.fetch_mode = "new"
    if "last_cursor" not in st.session_state:
        st.session_state.last_cursor = None
    if "collection_initialized" not in st.session_state:
        st.session_state.collection_initialized = initialize_collection()

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

    with col2:
        # Fetch button
        if st.button("Fetch Images", type="primary"):
            continue_from_last = (fetch_mode == "Continue from last")
            check_new_images(continue_from_last=continue_from_last)
    
    # Status section below the main controls
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

def check_new_images(continue_from_last=False):
    """Check for new images and process them"""
    try:
        
        # Create main container
        main_container = st.container()
        main_container.columns(1)

        with main_container:
            # Create columns for status and details
            status_column, details_column = st.columns(2)

            with status_column:
                st.write("### Status")
                status_text = st.empty()
                progress_text = st.empty()

            with details_column:
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

        # Fetch data
        all_data = fetch_data(target_count, continue_from_last)

        # Processing Phase
        if all_data:
            status_text.write("🔄 Processing Images")
            total = len(all_data)

            processed, stored, skipped, errors = process_and_store(all_data)

            # Show summary in main container
            with main_container:
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

            st.session_state.job_status = "Complete"
        else:
            status_text.write("❌ No data to process")
            st.session_state.job_status = "Error"

    except Exception as e:
        if 'status_text' in locals():
            status_text.write(f"❌ Error: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
        st.session_state.job_status = "Error"
