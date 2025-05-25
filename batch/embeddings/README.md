# Text Embeddings Generator

This batch job processes text files and generates embeddings using OpenAI's API.

## Prerequisites

1. OpenAI API key
2. Docker installed
3. Python 3.10 or higher

## Setup

1. Create a `.env` file in the embeddings directory with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. Build the Docker image:
```bash
docker build -t embeddings-generator .
```

3. Run the container:
```bash
docker run -v $(pwd)/../text_output:/input \
          -v $(pwd):/app \
          -v $(pwd)/.env:/app/.env \
          embeddings-generator
```

## Features

- Processes all text files in the input directory
- Splits text into chunks of appropriate size
- Generates embeddings using OpenAI's text-embedding-ada-002 model
- Handles API rate limits and errors gracefully
- Logs processing status and errors

## Error Handling

- API rate limit handling
- Retry mechanism for failed API calls
- File processing errors
- Logging of all operations

## Logging

The application logs all operations to stdout with timestamps, including:
- File processing status
- Chunk processing
- API call status
- Errors encountered


