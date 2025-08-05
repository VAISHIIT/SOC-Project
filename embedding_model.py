from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    """Handles text embeddings using a lightweight Hugging Face model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
                       Default: all-MiniLM-L6-v2 (lightweight, fast, good performance)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Embedding model loaded successfully")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode([text])[0]
    
    def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return [embedding for embedding in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
