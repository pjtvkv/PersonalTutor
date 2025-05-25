import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Constants
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_OUTPUT_DIR', '../embeddings_output')
TOP_N_RESULTS = 5

# Set page configuration
st.set_page_config(
    page_title="Embeddings Search",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
st.sidebar.title("Embeddings Search")

# Display directory path
st.sidebar.info(f"Embeddings Directory: {EMBEDDINGS_DIR}")

# Main content
st.title("Embeddings Data Viewer")

# Load embeddings data
@st.cache_data
def load_embeddings_data():
    if os.path.exists(EMBEDDINGS_DIR):
        json_files = list(Path(EMBEDDINGS_DIR).glob("*.json"))
        if json_files:
            all_data = []
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_data.extend(data)
                except Exception as e:
                    logging.warning(f"Error reading {file_path.name}: {str(e)}")
            return pd.DataFrame(all_data)
        else:
            return None
    return None

# Load embeddings data
st.session_state.data = load_embeddings_data()

# Main content
st.title("Embeddings Search")

if st.session_state.data is None:
    st.error(f"No embeddings data found in directory: {EMBEDDINGS_DIR}")
else:
    # Display basic info
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Total Files: {len(st.session_state.data['file_name'].unique())}")
        st.write(f"Total Chunks: {len(st.session_state.data)}")
    with col2:
        st.write(f"Files Processed:")
        st.write(st.session_state.data['file_name'].unique())

    # Search interface
    st.subheader("Search")
    
    # Get user query
    query = st.text_input("Enter your search query:")
    
    if query:
        try:
            # Get query embedding
            query_embedding = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            ).data[0].embedding
            
            # Calculate cosine similarity
            embeddings = np.array(st.session_state.data['embedding'].tolist())
            query_embedding = np.array(query_embedding)
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top N results
            top_indices = np.argsort(similarities)[-TOP_N_RESULTS:][::-1]
            
            # Display results
            st.subheader("Search Results")
            for idx in top_indices:
                row = st.session_state.data.iloc[idx]
                with st.expander(f"Relevance Score: {similarities[idx]:.3f}"):
                    st.write(f"**File:** {row['file_name']}")
                    st.write(f"**Chunk ID:** {row['chunk_id']}")
                    st.write(f"**Content:**")
                    st.write(row['chunk_content'])
                    
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")

    # Add file filtering
    st.subheader("Filter by File")
    file_filter = st.selectbox(
        "Select a file to view its chunks",
        ["All Files"] + list(st.session_state.data['file_name'].unique())
    )
    
    if file_filter != "All Files":
        filtered_data = st.session_state.data[st.session_state.data['file_name'] == file_filter]
        st.subheader(f"Chunks from {file_filter}")
        st.dataframe(filtered_data[['chunk_id', 'chunk_content']])

    # Download button
    csv = st.session_state.data.to_csv(index=False)
    st.download_button(
        label="Download All Data",
        data=csv,
        file_name="embeddings_data.csv",
        mime='text/csv'
    )
