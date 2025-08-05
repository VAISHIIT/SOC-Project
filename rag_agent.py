import os
from typing import List, Optional
from models import DocumentChunk, RetrievalResult, RAGResponse
from document_processor import DocumentProcessor
from embedding_model import EmbeddingModel
from vector_database import VectorDatabase
from llm_generator import LLMGenerator


class RAGAgent:
    """Main RAG Agent that orchestrates all components."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "microsoft/DialoGPT-medium",
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the RAG Agent with all components.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the language model
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        print("Initializing RAG Agent...")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_db = VectorDatabase(collection_name, persist_directory)
        self.llm_generator = LLMGenerator(llm_model_name)
        
        print("RAG Agent initialized successfully!")
    
    def load_document(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000, 
        overlap_size: int = 200,
        clear_existing: bool = False
    ) -> List[DocumentChunk]:
        """
        Load and process a PDF document into the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks
            clear_existing: Whether to clear existing data in the database
            
        Returns:
            List of processed DocumentChunk objects
        """
        print(f"\n{'='*50}")
        print(f"LOADING DOCUMENT: {pdf_path}")
        print(f"{'='*50}")
        
        if clear_existing:
            print("Clearing existing database...")
            self.vector_db.clear_collection()
        
        # Process document
        chunks = self.document_processor.process_document(pdf_path, chunk_size, overlap_size)
        
        # Add to vector database
        self.vector_db.add_chunks(chunks, self.embedding_model)
        
        total_chunks = self.vector_db.get_collection_count()
        print(f"Total chunks in database: {total_chunks}")
        
        return chunks
    
    def query(self, query: str, top_k: int = 2) -> RAGResponse:
        """
        Query the RAG system and get a response.
        
        Args:
            query: User query
            top_k: Number of top similar chunks to retrieve
            
        Returns:
            RAGResponse object containing the response and metadata
        """
        print(f"\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        # Check if database has any chunks
        total_chunks = self.vector_db.get_collection_count()
        if total_chunks == 0:
            return RAGResponse(
                query=query,
                response="No documents have been loaded into the database. Please load a document first.",
                retrieved_chunks=[],
                total_chunks_in_db=0
            )
        
        # Retrieve similar chunks
        retrieval_results = self.vector_db.search_similar_chunks(query, self.embedding_model, top_k)
        
        if not retrieval_results:
            return RAGResponse(
                query=query,
                response="No relevant information found in the database.",
                retrieved_chunks=[],
                total_chunks_in_db=total_chunks
            )
        
        # Display retrieved chunks
        print(f"\nRETRIEVED CHUNKS:")
        print("-" * 30)
        context_chunks = []
        for i, result in enumerate(retrieval_results, 1):
            print(f"Chunk {i} (Similarity: {result.similarity_score:.4f}):")
            print(f"Page {result.chunk.page_number}, Chunk {result.chunk.chunk_index}")
            print(f"Content: {result.chunk.content[:200]}...")
            print("-" * 30)
            context_chunks.append(result.chunk.content)
        
        # Generate response
        print("Generating response...")
        response = self.llm_generator.generate_response(query, context_chunks)
        
        return RAGResponse(
            query=query,
            response=response,
            retrieved_chunks=retrieval_results,
            total_chunks_in_db=total_chunks
        )
    
    def get_database_stats(self) -> dict:
        """Get statistics about the vector database."""
        total_chunks = self.vector_db.get_collection_count()
        chunk_ids = self.vector_db.list_all_chunks()
        
        return {
            "total_chunks": total_chunks,
            "sample_chunk_ids": chunk_ids[:5] if chunk_ids else [],
            "embedding_dimension": self.embedding_model.get_embedding_dimension(),
            "collection_name": self.vector_db.collection_name
        }
    
    def clear_database(self):
        """Clear all data from the vector database."""
        self.vector_db.clear_collection()
        print("Database cleared successfully!")


def main():
    """Example usage of the RAG Agent."""
    # Initialize RAG Agent
    rag_agent = RAGAgent()
    
    # Load a document (replace with your PDF path)
    pdf_path = "practical_guide_to_building_agents_notes.pdf"
    
    if os.path.exists(pdf_path):
        print(f"Loading document: {pdf_path}")
        chunks = rag_agent.load_document(pdf_path, chunk_size=800, overlap_size=150)
        
        # Show database stats
        stats = rag_agent.get_database_stats()
        print(f"\nDatabase Stats: {stats}")
        
        # Example queries
        queries = [
            "What are the key concepts discussed in this document?",
            "How does the system work?",
            "What are the main benefits?",
        ]
        
        for query in queries:
            response = rag_agent.query(query)
            print(f"\nResponse: {response.response}")
            print(f"Retrieved {len(response.retrieved_chunks)} chunks")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please place your PDF file in the same directory as this script.")


if __name__ == "__main__":
    main()
