from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Pydantic model for document chunks with metadata."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the chunk")
    page_number: int = Field(..., description="Page number from source document")
    chunk_index: int = Field(..., description="Index of chunk within the page")
    start_char: int = Field(..., description="Starting character position in original text")
    end_char: int = Field(..., description="Ending character position in original text")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True


class RetrievalResult(BaseModel):
    """Pydantic model for retrieval results."""
    
    chunk: DocumentChunk
    similarity_score: float = Field(..., description="Cosine similarity score")
    
    class Config:
        arbitrary_types_allowed = True


class RAGResponse(BaseModel):
    """Pydantic model for RAG agent response."""
    
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Generated response")
    retrieved_chunks: List[RetrievalResult] = Field(..., description="Retrieved chunks used for generation")
    total_chunks_in_db: int = Field(..., description="Total number of chunks in database")
