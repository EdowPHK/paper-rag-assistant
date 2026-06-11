from typing import List
from schemas import PdfPage
import os
import logging
import pymupdf

def parse_pdf_to_pages(pdf_path: str) -> List[PdfPage]:
    try:
        source_name = os.path.basename(pdf_path)
        with pymupdf.open(pdf_path) as doc:
            pages: List[PdfPage] = []
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    text = text.strip()
                pages.append({
                    "source": source_name,
                    "page_id": index,
                    "text": text,
                })
            return pages
    except (ValueError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Failed to parse PDF: %s", pdf_path)
        raise ValueError(f"Cannot parse PDF: {pdf_path}") from exc
    
