"""
Quick Start Script for Knowledge Graph Agent

This script helps you get started quickly with the Knowledge Graph Agent.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly configured."""
    print("Checking environment...")
    
    issues = []
    
    # Check .env file
    if not Path(".env").exists():
        issues.append("❌ .env file not found. Copy .env.example to .env and add your API keys.")
    else:
        print("✓ .env file found")
        
        # Check for required API keys
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("GOOGLE_API_KEY"):
            issues.append("⚠️  GOOGLE_API_KEY not set in .env")
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("⚠️  OPENAI_API_KEY not set in .env")
    
    # Check spaCy model
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("✓ spaCy model found")
        except OSError:
            issues.append("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
    except ImportError:
        issues.append("❌ spaCy not installed. Run: pip install -r requirements.txt")
    
    # Check other dependencies (use importable module names)
    missing_deps = []
    for dep in [
        "sentence_transformers",
        "networkx",
        "bs4",           # installed via beautifulsoup4
        "pdfplumber",
        "requests",
    ]:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        issues.append(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        issues.append("   Run: pip install -r requirements.txt")
    else:
        print("✓ All dependencies installed")
    
    return issues


def run_setup():
    """Run initial setup."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH AGENT - QUICK START")
    print("="*60 + "\n")
    
    # Check environment
    issues = check_environment()
    
    if issues:
        print("\n⚠️  Setup Issues Found:\n")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n📝 Setup Steps:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API keys to .env")
        print("  3. Install dependencies: pip install -r requirements.txt")
        print("  4. Download spaCy model: python -m spacy download en_core_web_sm")
        print("\nRun this script again after completing setup.\n")
        return False
    
    print("\n✅ Environment is properly configured!\n")
    
    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path(".cache").mkdir(exist_ok=True)
    print("✓ Created output directories")
    
    # Test imports
    print("\nTesting imports...")
    try:
        from src.pipeline import KnowledgeGraphPipeline
        from src.core.graph_schema import KnowledgeGraph
        print("✓ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Run basic test
    print("\nRunning basic test...")
    try:
        kg = KnowledgeGraph()
        from src.core.graph_schema import KGNode
        node = KGNode(id="test", label="Test", properties={"name": "Test"})
        kg.add_node(node)
        print("✓ Basic functionality working")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    
    print("\n🚀 Next Steps:")
    print("\n1. Run example with Wikipedia:")
    print("   python example_wikipedia.py")
    print("\n2. Run example with PDF files:")
    print("   python example_pdf.py")
    print("\n3. Build custom pipeline:")
    print("   from src.pipeline import KnowledgeGraphPipeline")
    print("   pipeline = KnowledgeGraphPipeline()")
    print("   pipeline.build_graph(sources)")
    
    print("\n📚 Documentation:")
    print("   README.md - Full documentation")
    print("   MIGRATION.md - Migration guide")
    print("   RESTRUCTURING_SUMMARY.md - Project overview")
    
    print("\n✨ Happy graph building!\n")
    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
