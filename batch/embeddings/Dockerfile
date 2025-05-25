FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for input, app code, and embeddings output
RUN mkdir -p /input /app /embeddings_output

# Set environment variables
ENV TEXT_OUTPUT_DIR=/input
ENV EMBEDDINGS_OUTPUT_DIR=/embeddings_output

# Set default command
CMD ["python", "/app/embeddings_generator.py"]
