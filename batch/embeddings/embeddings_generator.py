import os
import logging
from pathlib import Path
import json
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load OpenAI API key from environment
load_dotenv()
client = OpenAI()

# Constants
CHUNK_SIZE = 1000  # Maximum tokens per chunk
MAX_RETRIES = 3    # Maximum retries for API calls


def read_text_file(file_path: str) -> str:
    """Read text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int) -> list:
    """Split text into chunks of approximately chunk_size tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word.split())
        if current_length + word_length <= chunk_size:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embedding(text: str) -> list:
    """Get embedding for a text chunk using OpenAI API."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            retries += 1
            logging.warning(f"API error on attempt {retries}: {str(e)}")

    logging.error(f"Failed to get embedding after {MAX_RETRIES} attempts")
    return None

def process_text_files():
    """Process all text files and generate embeddings."""
    text_dir = os.getenv('TEXT_OUTPUT_DIR', '/text_output')
    embeddings_dir = os.getenv('EMBEDDINGS_OUTPUT_DIR', '/embeddings_output')
    
    if not os.path.exists(text_dir):
        logging.error(f"Text output directory {text_dir} does not exist")
        return
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
        logging.info(f"Created embeddings output directory: {embeddings_dir}")
    
    # Process each text file
    for txt_file in Path(text_dir).glob("*.txt"):
        try:
            # Read file content
            content = read_text_file(str(txt_file))
            if not content:
                continue
                
            # Split into chunks
            chunks = chunk_text(content, CHUNK_SIZE)
            
            # Store embeddings data
            embeddings_data = []
            
            # Process each chunk
            for chunk_id, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                if embedding:
                    embeddings_data.append({
                        "file_name": txt_file.name,
                        "chunk_id": chunk_id,
                        "chunk_content": chunk,
                        "embedding": embedding
                    })
            
            if embeddings_data:
                # Create output filename
                output_filename = os.path.join(embeddings_dir, f"{txt_file.stem}_embeddings.json")
                
                # Write to JSON file
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Saved embeddings for {txt_file.name} to {output_filename}")
                
        except Exception as e:
            logging.error(f"Error processing {txt_file.name}: {str(e)}")

def main():
    logging.info("Starting embeddings generation process")
    process_text_files()
    logging.info("Embeddings generation completed")

if __name__ == "__main__":
    main()
