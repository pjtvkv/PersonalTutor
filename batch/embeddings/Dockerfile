FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for input and app code
RUN mkdir -p /input /app

# Set environment variables
ENV TEXT_OUTPUT_DIR=/input

# Set default command
CMD ["python", "/app/embeddings_generator.py"]
