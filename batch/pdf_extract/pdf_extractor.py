import os
from PyPDF2 import PdfReader
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdf_files(input_dir: str, output_dir: str):
    """Process all PDF files in the input directory and save text to output directory."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each PDF file
    for pdf_file in Path(input_dir).glob("*.pdf"):
        try:
            # Extract text
            text = extract_text_from_pdf(str(pdf_file))
            
            if text:
                # Create output filename (replace .pdf with .txt)
                output_file = pdf_file.with_suffix('.txt').name
                output_path = os.path.join(output_dir, output_file)
                
                # Write text to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                logging.info(f"Successfully processed {pdf_file.name}")
            else:
                logging.warning(f"No text extracted from {pdf_file.name}")
                
        except Exception as e:
            logging.error(f"Error processing {pdf_file.name}: {str(e)}")

def main():
    input_dir = os.getenv('INPUT_DIR', '/input')
    output_dir = os.getenv('OUTPUT_DIR', '/output')
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist")
        return
    
    logging.info(f"Starting PDF extraction from {input_dir}")
    process_pdf_files(input_dir, output_dir)
    logging.info("PDF extraction completed")

if __name__ == "__main__":
    main()
