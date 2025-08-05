import chromadb
import numpy as np
from typing import List, Optional, Tuple
from models import DocumentChunk, RetrievalResult
from embedding_model import EmbeddingModel


class VectorDatabase:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        print(f"Initializing ChromaDB at: {persist_directory}")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.collection_name = collection_name
        print(f"ChromaDB collection '{collection_name}' initialized")
    
    def add_chunks(self, chunks: List[DocumentChunk], embedding_model: EmbeddingModel):
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of DocumentChunk objects
            embedding_model: EmbeddingModel instance for generating embeddings
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to database...")
        
        # Extract texts and generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.encode_texts(texts)
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = texts
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(chunks)} chunks to database")
    
    def search_similar_chunks(
        self, 
        query: str, 
        embedding_model: EmbeddingModel, 
        top_k: int = 2
    ) -> List[RetrievalResult]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query: Search query
            embedding_model: EmbeddingModel instance
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        print(f"Searching for top {top_k} similar chunks...")
        
        # Generate query embedding
        query_embedding = embedding_model.encode_text(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert results to RetrievalResult objects
        retrieval_results = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances, convert to similarity scores
                distance = results['distances'][0][i]
                similarity_score = 1 - distance  # Convert distance to similarity
                
                # Reconstruct DocumentChunk
                metadata = results['metadatas'][0][i]
                chunk = DocumentChunk(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    page_number=metadata['page_number'],
                    chunk_index=metadata['chunk_index'],
                    start_char=metadata['start_char'],
                    end_char=metadata['end_char'],
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['page_number', 'chunk_index', 'start_char', 'end_char']}
                )
                
                retrieval_result = RetrievalResult(
                    chunk=chunk,
                    similarity_score=similarity_score
                )
                retrieval_results.append(retrieval_result)
        
        print(f"Found {len(retrieval_results)} similar chunks")
        return retrieval_results
    
    def get_collection_count(self) -> int:
        """Get the total number of chunks in the database."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all data from the collection."""
        # Delete the collection and recreate it
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")
    
    def list_all_chunks(self) -> List[str]:
        """List all chunk IDs in the database."""
        results = self.collection.get()
        return results['ids'] if results['ids'] else []
