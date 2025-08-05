import os
from typing import List, Dict, Any
from markitdown import MarkItDown
from models import DocumentChunk


class DocumentProcessor:
    """Handles document processing using markitdown."""
    
    def __init__(self):
        self.markitdown = MarkItDown()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using markitdown.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = self.markitdown.convert(pdf_path)
        return result.text_content
    
    def split_into_pages(self, text: str) -> List[str]:
        """
        Split text into pages based on common page break indicators.
        
        Args:
            text: Full document text
            
        Returns:
            List of page texts
        """
        # Common page break patterns
        page_breaks = ['\f', '\n\n---\n\n', '\n\nPage ', '\n\n\n\n']
        
        pages = [text]
        for break_pattern in page_breaks:
            new_pages = []
            for page in pages:
                new_pages.extend(page.split(break_pattern))
            pages = new_pages
        
        # Filter out empty pages and strip whitespace
        pages = [page.strip() for page in pages if page.strip()]
        
        return pages
    
    def create_chunks_with_overlap(
        self, 
        pages: List[str], 
        chunk_size: int = 1000, 
        overlap_size: int = 200
    ) -> List[DocumentChunk]:
        """
        Create overlapping chunks from pages.
        
        Args:
            pages: List of page texts
            chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_id = 0
        
        for page_num, page_text in enumerate(pages, 1):
            if len(page_text) <= chunk_size:
                # Page fits in one chunk
                chunk = DocumentChunk(
                    id=f"chunk_{chunk_id}",
                    content=page_text,
                    page_number=page_num,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(page_text),
                    metadata={"source_page": page_num}
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split page into overlapping chunks
                start = 0
                chunk_index = 0
                
                while start < len(page_text):
                    end = min(start + chunk_size, len(page_text))
                    chunk_content = page_text[start:end]
                    
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_id}",
                        content=chunk_content,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={"source_page": page_num}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    chunk_index += 1
                    
                    # Move start position for next chunk with overlap
                    start = end - overlap_size
                    if start >= len(page_text):
                        break
        
        return chunks
    
    def process_document(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000, 
        overlap_size: int = 200
    ) -> List[DocumentChunk]:
        """
        Complete document processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        print(f"Processing document: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters")
        
        # Split into pages
        pages = self.split_into_pages(text)
        print(f"Split into {len(pages)} pages")
        
        # Create chunks with overlap
        chunks = self.create_chunks_with_overlap(pages, chunk_size, overlap_size)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
