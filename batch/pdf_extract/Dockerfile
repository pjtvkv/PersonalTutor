# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pdf_extractor.py .

# Create directories for input and output
RUN mkdir -p /input /output

# Set environment variables for input/output directories
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

# Set default command to run the Python script
CMD ["python", "pdf_extractor.py"]
