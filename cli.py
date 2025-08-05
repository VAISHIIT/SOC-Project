#!/usr/bin/env python3
"""
Interactive CLI for the RAG Agent
"""

import os
import sys
from rag_agent import RAGAgent


def print_header():
    print("=" * 60)
    print("             RAG AGENT - Interactive CLI")
    print("=" * 60)
    print()


def print_menu():
    print("\nAvailable Commands:")
    print("1. load <pdf_path>     - Load a PDF document")
    print("2. query <question>    - Ask a question")
    print("3. stats              - Show database statistics")
    print("4. clear              - Clear the database")
    print("5. help               - Show this menu")
    print("6. quit               - Exit the program")
    print()


def main():
    print_header()
    
    # Initialize RAG Agent
    print("Initializing RAG Agent...")
    try:
        rag_agent = RAGAgent()
    except Exception as e:
        print(f"Error initializing RAG Agent: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return
    
    print("RAG Agent ready!")
    print_menu()
    
    while True:
        try:
            user_input = input("rag> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == "quit" or command == "exit":
                print("Goodbye!")
                break
            
            elif command == "help":
                print_menu()
            
            elif command == "load":
                if len(parts) < 2:
                    print("Usage: load <pdf_path>")
                    continue
                
                pdf_path = parts[1]
                if not os.path.exists(pdf_path):
                    print(f"Error: File not found - {pdf_path}")
                    continue
                
                try:
                    chunks = rag_agent.load_document(pdf_path, clear_existing=True)
                    print(f"Successfully loaded {len(chunks)} chunks from {pdf_path}")
                except Exception as e:
                    print(f"Error loading document: {e}")
            
            elif command == "query":
                if len(parts) < 2:
                    print("Usage: query <your question>")
                    continue
                
                question = parts[1]
                try:
                    response = rag_agent.query(question)
                    
                    print(f"\nQuery: {response.query}")
                    print(f"Response: {response.response}")
                    print(f"\nRetrieved {len(response.retrieved_chunks)} chunks:")
                    
                    for i, result in enumerate(response.retrieved_chunks, 1):
                        print(f"\n--- Chunk {i} (Similarity: {result.similarity_score:.4f}) ---")
                        print(f"Page: {result.chunk.page_number}")
                        print(f"Content: {result.chunk.content[:300]}...")
                    
                except Exception as e:
                    print(f"Error processing query: {e}")
            
            elif command == "stats":
                try:
                    stats = rag_agent.get_database_stats()
                    print(f"\nDatabase Statistics:")
                    print(f"Total chunks: {stats['total_chunks']}")
                    print(f"Embedding dimension: {stats['embedding_dimension']}")
                    print(f"Collection name: {stats['collection_name']}")
                    if stats['sample_chunk_ids']:
                        print(f"Sample chunk IDs: {', '.join(stats['sample_chunk_ids'])}")
                except Exception as e:
                    print(f"Error getting stats: {e}")
            
            elif command == "clear":
                try:
                    rag_agent.clear_database()
                    print("Database cleared successfully!")
                except Exception as e:
                    print(f"Error clearing database: {e}")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see available commands")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
