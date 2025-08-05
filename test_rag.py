#!/usr/bin/env python3
"""
Test script for the RAG Agent
This script runs basic tests to verify all components work correctly.
"""

import os
import sys
import traceback
from typing import List


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        from models import DocumentChunk, RetrievalResult, RAGResponse
        print("✅ Models imported successfully")
        
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        from embedding_model import EmbeddingModel
        print("✅ EmbeddingModel imported successfully")
        
        from vector_database import VectorDatabase
        print("✅ VectorDatabase imported successfully")
        
        from llm_generator import LLMGenerator
        print("✅ LLMGenerator imported successfully")
        
        from rag_agent import RAGAgent
        print("✅ RAGAgent imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def test_models():
    """Test Pydantic models."""
    print("\n📋 Testing Pydantic models...")
    
    try:
        from models import DocumentChunk, RetrievalResult, RAGResponse
        
        # Test DocumentChunk
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is a test chunk content.",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=28,
            metadata={"test": "data"}
        )
        print(f"✅ DocumentChunk created: {chunk.id}")
        
        # Test RetrievalResult
        result = RetrievalResult(
            chunk=chunk,
            similarity_score=0.85
        )
        print(f"✅ RetrievalResult created with score: {result.similarity_score}")
        
        # Test RAGResponse
        response = RAGResponse(
            query="Test query",
            response="Test response",
            retrieved_chunks=[result],
            total_chunks_in_db=1
        )
        print(f"✅ RAGResponse created for query: '{response.query}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing error: {e}")
        traceback.print_exc()
        return False


def test_document_processor():
    """Test document processor with mock data."""
    print("\n📄 Testing document processor...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test text splitting
        sample_text = "This is page 1.\n\nThis is page 2.\n\nThis is page 3."
        pages = processor.split_into_pages(sample_text)
        print(f"✅ Text split into {len(pages)} pages")
        
        # Test chunking
        chunks = processor.create_chunks_with_overlap(pages, chunk_size=50, overlap_size=10)
        print(f"✅ Created {len(chunks)} chunks with overlap")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentProcessor testing error: {e}")
        traceback.print_exc()
        return False


def test_embedding_model():
    """Test embedding model (may take time on first run)."""
    print("\n🔢 Testing embedding model...")
    
    try:
        from embedding_model import EmbeddingModel
        
        print("   Loading model (this may take a moment)...")
        model = EmbeddingModel("all-MiniLM-L6-v2")
        print("✅ EmbeddingModel loaded successfully")
        
        # Test single text encoding
        text = "This is a test sentence."
        embedding = model.encode_text(text)
        print(f"✅ Single text encoded, dimension: {len(embedding)}")
        
        # Test multiple texts encoding
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = model.encode_texts(texts)
        print(f"✅ Multiple texts encoded: {len(embeddings)} embeddings")
        
        # Test similarity calculation
        similarity = model.calculate_similarity(embeddings[0], embeddings[1])
        print(f"✅ Similarity calculated: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingModel testing error: {e}")
        traceback.print_exc()
        return False


def test_vector_database():
    """Test vector database operations."""
    print("\n💾 Testing vector database...")
    
    try:
        from vector_database import VectorDatabase
        from embedding_model import EmbeddingModel
        from models import DocumentChunk
        
        # Initialize components
        db = VectorDatabase("test_collection", "./test_chroma_db")
        model = EmbeddingModel("all-MiniLM-L6-v2")
        print("✅ VectorDatabase and EmbeddingModel initialized")
        
        # Clear any existing data
        db.clear_collection()
        print("✅ Database cleared")
        
        # Create test chunks
        chunks = [
            DocumentChunk(
                id="test_1",
                content="The sky is blue and beautiful.",
                page_number=1,
                chunk_index=0,
                start_char=0,
                end_char=29
            ),
            DocumentChunk(
                id="test_2", 
                content="The ocean is deep and mysterious.",
                page_number=1,
                chunk_index=1,
                start_char=30,
                end_char=63
            )
        ]
        
        # Add chunks to database
        db.add_chunks(chunks, model)
        print(f"✅ Added {len(chunks)} chunks to database")
        
        # Test search
        results = db.search_similar_chunks("blue sky", model, top_k=1)
        print(f"✅ Search completed, found {len(results)} results")
        
        if results:
            print(f"   Top result similarity: {results[0].similarity_score:.4f}")
        
        # Test count
        count = db.get_collection_count()
        print(f"✅ Database contains {count} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorDatabase testing error: {e}")
        traceback.print_exc()
        return False


def test_llm_generator():
    """Test LLM generator (may take time on first run)."""
    print("\n🤖 Testing LLM generator...")
    
    try:
        from llm_generator import LLMGenerator
        
        print("   Loading LLM (this may take a moment)...")
        generator = LLMGenerator("microsoft/DialoGPT-medium")
        print("✅ LLMGenerator loaded successfully")
        
        # Test simple response generation
        prompt = "Hello, how are you?"
        response = generator.generate_simple_response(prompt, max_length=50)
        print(f"✅ Simple response generated: '{response[:50]}...'")
        
        # Test context-based response
        query = "What is the sky?"
        context = ["The sky is blue.", "It contains clouds."]
        response = generator.generate_response(query, context, max_length=50)
        print(f"✅ Context-based response generated: '{response[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ LLMGenerator testing error: {e}")
        traceback.print_exc()
        return False


def test_full_rag_system():
    """Test the complete RAG system with mock data."""
    print("\n🔄 Testing complete RAG system...")
    
    try:
        from rag_agent import RAGAgent
        from models import DocumentChunk
        
        # Initialize RAG agent
        rag_agent = RAGAgent(
            collection_name="test_rag",
            persist_directory="./test_rag_db"
        )
        print("✅ RAG Agent initialized")
        
        # Clear existing data
        rag_agent.clear_database()
        print("✅ Database cleared")
        
        # Create mock chunks (simulating a loaded document)
        mock_chunks = [
            DocumentChunk(
                id="mock_1",
                content="Artificial intelligence is the simulation of human intelligence in machines.",
                page_number=1,
                chunk_index=0,
                start_char=0,
                end_char=77
            ),
            DocumentChunk(
                id="mock_2",
                content="Machine learning is a subset of AI that focuses on learning from data.",
                page_number=1,
                chunk_index=1,
                start_char=78,
                end_char=148
            ),
            DocumentChunk(
                id="mock_3",
                content="Natural language processing helps computers understand human language.",
                page_number=2,
                chunk_index=0,
                start_char=0,
                end_char=69
            )
        ]
        
        # Add chunks to vector database
        rag_agent.vector_db.add_chunks(mock_chunks, rag_agent.embedding_model)
        print(f"✅ Added {len(mock_chunks)} mock chunks")
        
        # Test query
        query = "What is artificial intelligence?"
        response = rag_agent.query(query, top_k=2)
        print(f"✅ Query processed successfully")
        print(f"   Query: {response.query}")
        print(f"   Response: {response.response[:100]}...")
        print(f"   Retrieved chunks: {len(response.retrieved_chunks)}")
        
        # Test database stats
        stats = rag_agent.get_database_stats()
        print(f"✅ Database stats retrieved: {stats['total_chunks']} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Full RAG system testing error: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting RAG Agent Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Tests", test_models),
        ("Document Processor Tests", test_document_processor),
        ("Embedding Model Tests", test_embedding_model),
        ("Vector Database Tests", test_vector_database),
        ("LLM Generator Tests", test_llm_generator),
        ("Full RAG System Tests", test_full_rag_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG Agent is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)
