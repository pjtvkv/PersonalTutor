# Embeddings Viewer Web App

A Streamlit-based web application to view and analyze embeddings data generated from text files.

## Features

- View embeddings data from JSON files
- Display file information and chunk contents
- Preview data in a table format
- Download data as CSV
- Modern and responsive UI

## Prerequisites

- Docker and Docker Compose
- Python 3.10+

## Running the App

### Using Docker

1. Build the Docker image:
```bash
docker build -t rag-app .
```

2. Run the container:
```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/../batch/embeddings_output:/embeddings_output \
  -v $(pwd):/app \
  --name rag-app \
  rag-app
```

3. Open your browser and navigate to `http://localhost:8501`

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Environment Variables

- `EMBEDDINGS_OUTPUT_DIR`: Directory containing the JSON files (default: `/embeddings_output`)

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration
- `README.md`: Documentation
