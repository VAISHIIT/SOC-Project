#!/usr/bin/env python3
"""
Example usage of the RAG Agent
This script demonstrates how to use the RAG system programmatically.
"""

import os
from rag_agent import RAGAgent


def run_example():
    """Run a complete example of the RAG system."""
    
    print("🤖 RAG Agent Example")
    print("=" * 50)
    
    # Initialize the RAG Agent
    print("1. Initializing RAG Agent...")
    rag_agent = RAGAgent(
        embedding_model_name="all-MiniLM-L6-v2",  # Lightweight and fast
        llm_model_name="microsoft/DialoGPT-medium",  # Good for conversation
        collection_name="example_docs",
        persist_directory="./example_chroma_db"
    )
    
    # Check if the example PDF exists
    pdf_path = "practical_guide_to_building_agents_notes.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please place your PDF file in the project directory.")
        return
    
    # Load the document
    print(f"\n2. Loading document: {pdf_path}")
    try:
        chunks = rag_agent.load_document(
            pdf_path,
            chunk_size=800,      # Moderate chunk size
            overlap_size=150,    # Good overlap for context
            clear_existing=True  # Start fresh
        )
        print(f"✅ Successfully loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return
    
    # Show database statistics
    print(f"\n3. Database Statistics:")
    stats = rag_agent.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Example queries
    queries = [
        "What are the main topics covered in this document?",
        "How does the system work?",
        "What are the key benefits mentioned?",
        "Can you explain the methodology?",
        "What are the conclusions or recommendations?"
    ]
    
    print(f"\n4. Running Example Queries:")
    print("-" * 30)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        try:
            response = rag_agent.query(query, top_k=2)
            
            print(f"📝 Response: {response.response}")
            
            if response.retrieved_chunks:
                print(f"\n📚 Retrieved Chunks ({len(response.retrieved_chunks)}):")
                for j, result in enumerate(response.retrieved_chunks, 1):
                    print(f"\n  Chunk {j} (Similarity: {result.similarity_score:.4f}):")
                    print(f"  📄 Page {result.chunk.page_number}, Section {result.chunk.chunk_index}")
                    print(f"  📖 Preview: {result.chunk.content[:150]}...")
            else:
                print("   No relevant chunks found.")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "="*70)
    
    print(f"\n🎉 Example completed successfully!")
    print(f"Total chunks in database: {rag_agent.get_database_stats()['total_chunks']}")


def run_interactive_demo():
    """Run an interactive demo where users can ask custom questions."""
    
    print("\n🔄 Interactive Demo Mode")
    print("=" * 50)
    print("You can now ask custom questions about the loaded document.")
    print("Type 'quit' to exit the interactive mode.")
    
    # Initialize RAG agent (reuse the same database)
    rag_agent = RAGAgent(
        collection_name="example_docs",
        persist_directory="./example_chroma_db"
    )
    
    # Check if database has content
    stats = rag_agent.get_database_stats()
    if stats['total_chunks'] == 0:
        print("❌ No documents found in database. Please run the example first.")
        return
    
    print(f"✅ Found {stats['total_chunks']} chunks in database.")
    
    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Processing query: {query}")
            response = rag_agent.query(query, top_k=2)
            
            print(f"\n💬 Response:")
            print(f"   {response.response}")
            
            if response.retrieved_chunks:
                print(f"\n📚 Source Information:")
                for i, result in enumerate(response.retrieved_chunks, 1):
                    print(f"   {i}. Page {result.chunk.page_number} (Similarity: {result.similarity_score:.3f})")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        # Run the basic example
        run_example()
        
        # Ask if user wants interactive mode
        print("\n" + "="*70)
        choice = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
