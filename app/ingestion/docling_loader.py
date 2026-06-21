import os

# Fix HuggingFace symlink issue on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HOME"] = r"D:\Personal\Deeplearning\Project\Prototype\hf_cache"

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DoclingLoader:

    def __init__(self):
        # Configure pipeline for lower memory usage
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False           # Disable OCR (saves ~1-2 GB)
        pipeline_options.do_table_structure = False  # Keep table parsing
        pipeline_options.images_scale = 0.25       # Lower image resolution
        pipeline_options.generate_page_images = False    # Don't extract page images
        pipeline_options.generate_picture_images = False  # Don't extract embedded images
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def load_pdf(self, pdf_path):
        result = self.converter.convert(pdf_path)
        return result.document