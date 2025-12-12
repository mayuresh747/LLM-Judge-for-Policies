import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.ingestion import load_document, split_documents
from src.utils.vectorstore import create_vectorstore
from dotenv import load_dotenv

# Load env vars
load_dotenv()

def test_pipeline():
    print("Testing Ingestion Pipeline...")
    
    # Check API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Skipping VectorStore test: GOOGLE_API_KEY not found.")
        # We can still test loading and splitting
    
    file_path = "data/test_doc.txt"
    
    # 1. Load
    print(f"Loading {file_path}...")
    docs = load_document(file_path)
    print(f"Loaded {len(docs)} documents.")
    assert len(docs) > 0
    
    # 2. Split
    print("Splitting documents...")
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    print(f"Created {len(chunks)} chunks.")
    assert len(chunks) > 0
    
    # 3. VectorStore (only if key exists)
    if os.getenv("GOOGLE_API_KEY"):
        print("Creating VectorStore...")
        vectorstore = create_vectorstore(chunks)
        print("VectorStore created successfully.")
        
        # Simple search test
        print("Testing similarity search...")
        results = vectorstore.similarity_search("Streamlit", k=1)
        print(f"Search Result: {results[0].page_content}")
        assert "Streamlit" in results[0].page_content
    else:
        print("Warning: GOOGLE_API_KEY missing. Cannot test Embedding/VectorStore.")

if __name__ == "__main__":
    test_pipeline()
