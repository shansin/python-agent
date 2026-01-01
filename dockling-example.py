from yt_dlp.utils import base_url
import os
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline_options import AcceleratorOptions

# Configure accelerator options for GPU
pipeline_options.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,  # or AcceleratorDevice.AUTO
)

#todo: https://docling-project.github.io/docling/examples/gpu_standard_pipeline/

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

def convert_documents():
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
    
    # Remove duplicates
    files_to_process = list(set(files_to_process))

    if not files_to_process:
        print(f"No supported files found in '{input_folder}' or its subdirectories.")
        print(f"Supported extensions: {', '.join(sorted(supported_extensions))}")
        return

    print(f"Found {len(files_to_process)} file(s) to process.")

    # Initialize DocumentConverter with remote URL
    # Assuming user's snippet is correct for connecting to remote server
    converter = DocumentConverter()

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}...")
        
        try:
            # Determine relative path to maintain structure
            relative_path = file_path.relative_to(input_path)
            # Create corresponding output directory structure
            target_output_dir = output_path / relative_path.parent
            target_output_dir.mkdir(parents=True, exist_ok=True)

            # Convert single file
            result = converter.convert(file_path)

            if result.status == "success":
                markdown_content = result.document.export_to_markdown()
                
                output_filename = file_path.stem + ".md"
                output_file = target_output_dir / output_filename
                
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(markdown_content)
                    
                print(f"  Success! Saved to '{output_file}'")
            else:
                print(f"  Error converting {file_path.name}: {result.errors}")

        except Exception as e:
            print(f"  Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    convert_documents()
