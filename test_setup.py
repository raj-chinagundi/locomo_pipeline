"""
Quick test to verify setup is working correctly.
Run this after installation to check that core components work.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import yaml
        print("  ✓ PyYAML")
    except ImportError:
        print("  ✗ PyYAML - run: uv add pyyaml")
        return False
    
    try:
        from langchain_core.documents import Document
        print("  ✓ LangChain Core")
    except ImportError:
        print("  ✗ LangChain - run: uv sync")
        return False
    
    try:
        from langchain_community.retrievers import BM25Retriever
        print("  ✓ LangChain Community")
    except ImportError:
        print("  ✗ LangChain Community - run: uv sync")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("  ✓ Sentence Transformers")
    except ImportError:
        print("  ✗ Sentence Transformers - run: uv sync")
        return False
    
    try:
        import faiss
        print("  ✓ FAISS")
    except ImportError:
        print("  ✗ FAISS - run: uv sync")
        return False
    
    print("\n✓ All required imports successful!\n")
    return True


def test_modules():
    """Test that custom modules can be imported."""
    print("Testing custom modules...")
    
    try:
        from dataloader import DataLoader
        print("  ✓ dataloader.py")
    except ImportError as e:
        print(f"  ✗ dataloader.py - {e}")
        return False
    
    try:
        from retriever import Retriever, RETRIEVER_INFO
        print("  ✓ retriever.py")
    except ImportError as e:
        print(f"  ✗ retriever.py - {e}")
        return False
    
    try:
        from llm_helper import get_llm
        print("  ✓ llm_helper.py")
    except ImportError as e:
        print(f"  ✗ llm_helper.py - {e}")
        return False
    
    print("\n✓ All custom modules loaded!\n")
    return True


def test_config():
    """Test that config file exists and is valid."""
    print("Testing configuration...")
    
    import yaml
    from pathlib import Path
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("  ✗ config.yaml not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("  ✓ config.yaml is valid")
    except Exception as e:
        print(f"  ✗ config.yaml has errors: {e}")
        return False
    
    # Check required keys
    required_keys = ['data', 'retrievers', 'evaluation', 'llm', 'output']
    for key in required_keys:
        if key not in config:
            print(f"  ✗ Missing required key: {key}")
            return False
    
    print("  ✓ All required config keys present")
    
    # Check dataset exists
    json_path = Path(config['data']['json_path'])
    if not json_path.exists():
        print(f"  ⚠️  Dataset not found: {json_path}")
        print("     (This is OK if you haven't downloaded it yet)")
    else:
        print(f"  ✓ Dataset found: {json_path}")
    
    print("\n✓ Configuration valid!\n")
    return True


def test_basic_functionality():
    """Test basic functionality with dummy data."""
    print("Testing basic functionality...")
    
    try:
        from langchain_core.documents import Document
        from retriever import Retriever
        
        # Create dummy documents
        dummy_docs = [
            Document(page_content="Hello world", metadata={"dia_id": "D1:1"}),
            Document(page_content="Python programming", metadata={"dia_id": "D1:2"}),
            Document(page_content="Machine learning", metadata={"dia_id": "D1:3"}),
        ]
        
        # Initialize retriever
        retriever = Retriever(dummy_docs, config={'top_k': 2, 'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'})
        print("  ✓ Retriever initialized")
        
        # Test BM25
        bm25 = retriever.bm25
        results = bm25.invoke("hello")
        print(f"  ✓ BM25 working ({len(results)} results)")
        
        print("\n✓ Basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MEMAGENT Setup Verification")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Custom Modules", test_modules),
        ("Configuration", test_config),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✓ All tests passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Review config.yaml")
        print("  2. Run: uv run python run_pipeline.py")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

