# RAG Agent - Retrieval-Augmented Generation System

A modular and lightweight RAG (Retrieval-Augmented Generation) agent that processes PDF documents and provides intelligent question-answering capabilities using open-source models.

## Features

- 📄 **PDF Text Extraction**: Uses `markitdown` to extract text from PDF documents
- 🧩 **Smart Chunking**: Creates overlapping chunks with Pydantic models for structured data
- 🔍 **Semantic Search**: Uses lightweight Hugging Face embedding models for vector search
- 💾 **Vector Storage**: ChromaDB for persistent vector storage with cosine similarity
- 🤖 **Text Generation**: Free, open-source LLMs from Hugging Face
- 📊 **Retrieval Display**: Shows top-k relevant chunks with similarity scores
- 🏗️ **Modular Design**: Clean, maintainable code structure
- 💻 **Interactive CLI**: Easy-to-use command-line interface

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Document  │ -> │ Document         │ -> │ Text Chunks     │
│                 │    │ Processor        │    │ (with overlap)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         |
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │ Embedding        │ -> │ Vector          │
│                 │    │ Model            │    │ Embeddings      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         |
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final         │ <- │ LLM Generator    │ <- │ ChromaDB        │
│   Response      │    │                  │    │ (Vector Store)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Clone or download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your PDF document** in the project directory (e.g., `practical_guide_to_building_agents_notes.pdf`)

## Usage

### Interactive CLI (Recommended)

Run the interactive command-line interface:

```bash
python cli.py
```

Available commands:
- `load <pdf_path>` - Load a PDF document into the system
- `query <question>` - Ask a question about the loaded document
- `stats` - Show database statistics
- `clear` - Clear the database
- `help` - Show available commands
- `quit` - Exit the program

### Example CLI Session

```
rag> load practical_guide_to_building_agents_notes.pdf
Processing document: practical_guide_to_building_agents_notes.pdf
Extracted 50000 characters
Split into 25 pages
Created 75 chunks
Successfully loaded 75 chunks

rag> query What are the main concepts in this document?
Query: What are the main concepts in this document?
Response: Based on the document, the main concepts include...

Retrieved 2 chunks:
--- Chunk 1 (Similarity: 0.8234) ---
Page: 5
Content: The fundamental concepts of building agents include...

--- Chunk 2 (Similarity: 0.7891) ---
Page: 12
Content: Key principles for agent development involve...
```

### Programmatic Usage

```python
from rag_agent import RAGAgent

# Initialize the RAG agent
rag_agent = RAGAgent()

# Load a document
chunks = rag_agent.load_document("your_document.pdf")

# Query the system
response = rag_agent.query("What are the key points?")

print(f"Response: {response.response}")
print(f"Retrieved {len(response.retrieved_chunks)} relevant chunks")

# Display retrieved chunks
for i, result in enumerate(response.retrieved_chunks):
    print(f"Chunk {i+1} (Score: {result.similarity_score:.4f})")
    print(f"Content: {result.chunk.content[:200]}...")
```

## Configuration

### Model Customization

You can customize the models used by the RAG agent:

```python
rag_agent = RAGAgent(
    embedding_model_name="all-MiniLM-L6-v2",  # Fast, lightweight embedding model
    llm_model_name="microsoft/DialoGPT-medium",  # Conversational LLM
    collection_name="my_documents",
    persist_directory="./my_vector_db"
)
```

### Recommended Models

**Embedding Models** (from Hugging Face):
- `all-MiniLM-L6-v2` - Default, good balance of speed and quality
- `all-MiniLM-L12-v2` - Better quality, slightly slower
- `paraphrase-MiniLM-L6-v2` - Good for paraphrase detection

**Language Models** (from Hugging Face):
- `microsoft/DialoGPT-medium` - Default, good for conversational responses
- `distilgpt2` - Lighter, faster alternative
- `gpt2` - Standard GPT-2 model

### Chunking Parameters

```python
chunks = rag_agent.load_document(
    "document.pdf",
    chunk_size=1000,    # Maximum characters per chunk
    overlap_size=200,   # Overlap between consecutive chunks
    clear_existing=True # Clear existing data in database
)
```

## Components

### 1. Document Processor (`document_processor.py`)
- Extracts text from PDFs using `markitdown`
- Splits text into pages
- Creates overlapping chunks with metadata

### 2. Embedding Model (`embedding_model.py`)
- Loads sentence transformer models from Hugging Face
- Generates vector embeddings for text
- Calculates cosine similarity

### 3. Vector Database (`vector_database.py`)
- Uses ChromaDB for persistent vector storage
- Implements cosine similarity search
- Manages document chunks and metadata

### 4. LLM Generator (`llm_generator.py`)
- Loads language models from Hugging Face
- Generates responses based on query and context
- Handles prompt formatting and response cleaning

### 5. RAG Agent (`rag_agent.py`)
- Main orchestrator class
- Coordinates all components
- Provides high-level API

### 6. Data Models (`models.py`)
- Pydantic models for type safety
- `DocumentChunk`: Represents a text chunk with metadata
- `RetrievalResult`: Search result with similarity score
- `RAGResponse`: Complete response with metadata

## File Structure

```
RAG Agent/
├── cli.py                    # Interactive command-line interface
├── rag_agent.py             # Main RAG agent orchestrator
├── document_processor.py    # PDF processing and chunking
├── embedding_model.py       # Text embedding functionality
├── vector_database.py       # ChromaDB vector storage
├── llm_generator.py         # Language model for generation
├── models.py                # Pydantic data models
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── chroma_db/              # ChromaDB data directory (auto-created)
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- At least 2GB RAM (for model loading)
- GPU recommended but not required

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory issues**: If models are too large, try smaller alternatives:
   ```python
   # Use lighter models
   rag_agent = RAGAgent(
       embedding_model_name="all-MiniLM-L6-v2",
       llm_model_name="distilgpt2"
   )
   ```

3. **PDF processing errors**: Ensure your PDF is text-based (not scanned images)

4. **Slow performance**: Consider using GPU if available, or reduce chunk sizes

### Performance Tips

- Use GPU acceleration if available
- Reduce `chunk_size` for faster processing
- Use lighter models for better speed
- Increase `overlap_size` for better context retention

## Contributing

Feel free to contribute by:
- Adding support for more document types
- Implementing better chunking strategies
- Adding more embedding/LLM model options
- Improving the user interface
- Adding evaluation metrics

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- **markitdown**: For PDF text extraction
- **ChromaDB**: For vector database functionality
- **Hugging Face**: For pre-trained models
- **Sentence Transformers**: For embedding models
- **Transformers**: For language models
- **Pydantic**: For data validation and modeling
