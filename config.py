"""
Configuration settings for the RAG Agent
Modify these settings to customize the behavior of your RAG system.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Embedding Model Settings
EMBEDDING_MODEL = {
    "name": "all-MiniLM-L6-v2",  # Fast and lightweight
    "alternatives": [
        "all-MiniLM-L12-v2",     # Better quality, slightly slower
        "paraphrase-MiniLM-L6-v2",  # Good for paraphrase detection
        "all-mpnet-base-v2",     # High quality, larger model
    ]
}

# Language Model Settings
LANGUAGE_MODEL = {
    "name": "microsoft/DialoGPT-medium",  # Good for conversation
    "alternatives": [
        "distilgpt2",            # Lighter and faster
        "gpt2",                  # Standard GPT-2
        "microsoft/DialoGPT-small",  # Smaller version
    ]
}

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# =============================================================================

CHUNKING_CONFIG = {
    "chunk_size": 1000,      # Maximum characters per chunk
    "overlap_size": 200,     # Overlap between consecutive chunks
    "min_chunk_size": 100,   # Minimum chunk size to keep
}

# =============================================================================
# VECTOR DATABASE CONFIGURATION
# =============================================================================

VECTOR_DB_CONFIG = {
    "collection_name": "rag_documents",
    "persist_directory": "./chroma_db",
    "similarity_metric": "cosine",  # cosine, euclidean, or manhattan
}

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

RETRIEVAL_CONFIG = {
    "top_k": 2,              # Number of chunks to retrieve
    "min_similarity": 0.1,   # Minimum similarity threshold
    "max_chunks_display": 3, # Maximum chunks to display in results
}

# =============================================================================
# GENERATION CONFIGURATION
# =============================================================================

GENERATION_CONFIG = {
    "max_response_length": 300,  # Maximum length of generated response
    "temperature": 0.7,          # Sampling temperature (0.1 = conservative, 1.0 = creative)
    "do_sample": True,           # Whether to use sampling
    "top_p": 0.9,               # Nucleus sampling parameter
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

SYSTEM_CONFIG = {
    "device": "auto",           # "cuda", "cpu", or "auto"
    "cache_dir": "./model_cache",  # Directory to cache downloaded models
    "log_level": "INFO",        # Logging level: DEBUG, INFO, WARNING, ERROR
    "max_memory_gb": 8,         # Maximum memory to use (in GB)
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    "rag_prompt": """Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:""",
    
    "no_context_prompt": """Answer the following question to the best of your ability:

Question: {query}

Answer:""",
    
    "system_message": "You are a helpful AI assistant that provides accurate and informative responses based on the given context."
}

# =============================================================================
# FILE PATHS
# =============================================================================

DEFAULT_PATHS = {
    "default_pdf": "practical_guide_to_building_agents_notes.pdf",
    "output_dir": "./output",
    "logs_dir": "./logs",
    "temp_dir": "./temp",
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

PERFORMANCE_CONFIG = {
    "batch_size": 32,           # Batch size for embedding generation
    "max_workers": 4,           # Number of worker threads
    "enable_gpu": True,         # Whether to use GPU if available
    "precision": "float16",     # Model precision: float16, float32
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

VALIDATION_CONFIG = {
    "validate_chunks": True,     # Validate chunk format
    "check_similarity": True,    # Validate similarity scores
    "max_chunk_length": 2000,   # Maximum allowed chunk length
    "min_query_length": 3,      # Minimum query length
}
