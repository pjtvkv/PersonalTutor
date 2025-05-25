import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Embeddings Viewer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Get embeddings directory
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_OUTPUT_DIR', '../embeddings_output')

# Sidebar
st.sidebar.title("Embeddings Viewer")

# Display directory path
st.sidebar.info(f"Embeddings Directory: {EMBEDDINGS_DIR}")

# Main content
st.title("Embeddings Data Viewer")

# Load all JSON files
if os.path.exists(EMBEDDINGS_DIR):
    json_files = list(Path(EMBEDDINGS_DIR).glob("*.json"))
    if json_files:
        # Load all files into a single DataFrame
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                st.error(f"Error reading {file_path.name}: {str(e)}")
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Display basic info
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total Files: {len(json_files)}")
                st.write(f"Total Chunks: {len(df)}")
            with col2:
                st.write(f"Files Processed:")
                st.write(df['file_name'].unique())
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df[['file_name', 'chunk_id', 'chunk_content']].head(10))
            
            # Display full data with filtering
            st.subheader("Full Data")
            
            # Add filtering options
            file_filter = st.selectbox(
                "Filter by File",
                ["All Files"] + list(df['file_name'].unique())
            )
            
            if file_filter != "All Files":
                df = df[df['file_name'] == file_filter]
            
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="embeddings_data.csv",
                mime='text/csv'
            )
    else:
        st.info("No JSON files found in the embeddings directory")
else:
    st.error(f"Embeddings directory {EMBEDDINGS_DIR} does not exist")
