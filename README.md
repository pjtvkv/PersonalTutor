# PersonalTutor
LLM Based Personal Tutor for Students based in India

## Features

- PDF Content Extraction Batch Job
  - Extracts text from PDF files
  - Containerized Python application
  - Volume mounting for input and output directories
  - Robust error handling and logging

## PDF Extractor Usage

1. Build the Docker image:
```bash
docker build -t pdf-extractor .
```

2. Run the container with mounted volumes:
```bash
docker run -v $(pwd)/batch/input:/input \
          -v $(pwd)/batch/output:/output \
          -v $(pwd)/batch:/app \
          pdf-extractor
```

Replace `/path/to/input`, `/path/to/output`, and `/path/to/app` with your actual directory paths.

## Directory Structure

- `/input`: Directory containing PDF files to process
- `/output`: Directory where extracted text files will be saved
- `/app`: Application code directory

## Prerequisites

- Docker installed on your system
- Python 3.10 or higher

## Error Handling

The application includes:
- File processing errors
- PDF reading errors
- Text extraction failures
- Logging of all operations

## Logging

The application logs all operations to stdout with timestamps, including:
- Successful file processing
- Errors encountered
- Progress information
