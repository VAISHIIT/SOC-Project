#!/usr/bin/env python3
"""
Setup script for the RAG Agent
This script helps with initial setup and dependency installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    print("=" * 60)
    print("          RAG AGENT SETUP SCRIPT")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    major, minor = sys.version_info[:2]
    
    if major < 3 or (major == 3 and minor < 8):
        print(f"❌ Python {major}.{minor} is not supported.")
        print("   Please upgrade to Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {major}.{minor} is supported.")
    return True


def check_pip():
    """Check if pip is available."""
    print("\n📦 Checking pip availability...")
    
    try:
        import pip
        print("✅ pip is available.")
        return True
    except ImportError:
        print("❌ pip is not available.")
        print("   Please install pip to continue.")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n⬇️  Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found.")
        return False
    
    try:
        # Use the current Python executable to ensure correct environment
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully.")
            return True
        else:
            print("❌ Failed to install dependencies.")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "./chroma_db",
        "./model_cache", 
        "./output",
        "./logs",
        "./temp"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created/verified directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True


def check_pdf_file():
    """Check for the example PDF file."""
    print("\n📄 Checking for example PDF...")
    
    pdf_file = Path("practical_guide_to_building_agents_notes.pdf")
    
    if pdf_file.exists():
        print(f"✅ Found example PDF: {pdf_file}")
        return True
    else:
        print(f"⚠️  Example PDF not found: {pdf_file}")
        print("   You can place your own PDF file in this directory.")
        return False


def test_installation():
    """Test if the installation works."""
    print("\n🧪 Testing installation...")
    
    try:
        # Try importing key modules
        from models import DocumentChunk
        from rag_agent import RAGAgent
        print("✅ Core modules import successfully.")
        
        print("   Running basic functionality test...")
        
        # Quick test of model creation
        chunk = DocumentChunk(
            id="test",
            content="Test content",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=12
        )
        print("✅ Pydantic models work correctly.")
        
        print("✅ Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("   Some dependencies might not be installed correctly.")
        return False


def show_next_steps():
    """Show next steps after setup."""
    print("\n🚀 Next Steps:")
    print("-" * 40)
    print("1. Place your PDF file in the project directory")
    print("2. Run the test script: python test_rag.py")
    print("3. Try the example: python example.py")
    print("4. Use the CLI: python cli.py")
    print("5. Or import RAGAgent in your own script")
    print()
    print("📚 Documentation: See README.md for detailed usage")
    print()


def show_system_info():
    """Display system information."""
    print("💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Platform: {platform.machine()}")
    print(f"   Working Directory: {os.getcwd()}")
    print()


def main():
    """Main setup function."""
    print_header()
    
    show_system_info()
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        return False
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed during directory creation.")
        return False
    
    # Check for PDF file
    check_pdf_file()
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during installation test.")
        return False
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    show_next_steps()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("🎉 Your RAG Agent is ready to use!")
        else:
            print("💥 Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        sys.exit(1)
