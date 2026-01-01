import mimetypes
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
dockling_api_url = os.getenv("DOCKLING_API_URL")
input_folder = "input"
output_folder = "output"

# Supported extensions based on Docling documentation
supported_extensions = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", 
    ".md", ".adoc", ".xml", ".json", ".csv",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"
}

def convert_pdfs():
    # Ensure input and output directories exist
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"[{input_folder}] directory not found. Creating it...")
        input_path.mkdir(parents=True, exist_ok=True)
        print(f"Please place your files in the '{input_folder}' directory and run this script again.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Process files with supported extensions recursively
    files_to_process = []
    for ext in supported_extensions:
        files_to_process.extend(input_path.rglob(f"*{ext}"))
    
    # Remove duplicates if any (though rglob with distinct extensions shouldn't overlap)
    files_to_process = list(set(files_to_process))

    if not files_to_process:
        print(f"No supported files found in '{input_folder}' or its subdirectories.")
        print(f"Supported extensions: {', '.join(sorted(supported_extensions))}")
        return

    print(f"Found {len(files_to_process)} file(s) to process.")

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}...")
        
        try:
            # Determine relative path to maintain structure
            relative_path = file_path.relative_to(input_path)
            # Create corresponding output directory structure
            target_output_dir = output_path / relative_path.parent
            target_output_dir.mkdir(parents=True, exist_ok=True)

            # Guess mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                # Default fallback if unknown
                mime_type = 'application/octet-stream'

            with open(file_path, 'rb') as f:
                # Prepare the multipart upload
                # The API expects the field name to be 'files' (plural)
                files = {
                    'files': (file_path.name, f, mime_type)
                }
                # Optional: Add data parameters if needed, e.g., for OCR options
                # data = {'do_ocr': 'true'} 
                
                response = requests.post(dockling_api_url, files=files)
                
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    data = response.json()
                    # Parse response structure: {'document': {'md_content': '...'}, ...}
                    document = data.get("document", {})
                    markdown_content = document.get("md_content")
                    
                    if not markdown_content:
                        # Fallback: check other common keys or dump full JSON
                        markdown_content = data.get("markdown") or data.get("content") or data.get("text")
                    
                    if not markdown_content:
                        # Final Fallback
                        print(f"  Warning: Could not identify 'md_content' field in JSON response. Saving full JSON.")
                        markdown_content = str(data)
                    
                    output_filename = file_path.stem + ".md"
                    output_file = target_output_dir / output_filename
                    
                    with open(output_file, "w", encoding="utf-8") as out_f:
                        out_f.write(markdown_content)
                        
                    print(f"  Success! Saved to '{output_file}'")

                except requests.exceptions.JSONDecodeError:
                    # If response is not JSON, assume it's raw text/markdown
                    print("  Response is not JSON. Saving as raw text.")
                    output_filename = file_path.stem + ".md"
                    output_file = target_output_dir / output_filename
                    
                    with open(output_file, "w", encoding="utf-8") as out_f:
                        out_f.write(response.text)
                    print(f"  Success! Saved to '{output_file}'")
            
            else:
                print(f"  Error: API returned status code {response.status_code}")
                # Print first 200 chars of response to avoid spamming console
                print(f"  Response: {response.text[:200]}...")

        except Exception as e:
            print(f"  Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    convert_pdfs()
