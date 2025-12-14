import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

# Batch size for embedding (to avoid memory issues)
EMBEDDING_BATCH_SIZE = 100

def create_vectorstore(documents: List[Document]) -> FAISS:
    """Creates a FAISS vector store from a list of documents using local Ollama embeddings."""
    
    # Use Ollama embeddings (local, no rate limits)
    # nomic-embed-text is a good general-purpose embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Batch documents to avoid memory issues with large document sets
    if len(documents) <= EMBEDDING_BATCH_SIZE:
        # Small enough to embed in one call
        return FAISS.from_documents(documents, embeddings)
    
    # Process in batches
    print(f"Embedding {len(documents)} documents in batches of {EMBEDDING_BATCH_SIZE}...")
    vectorstore = None
    
    for i in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[i:i + EMBEDDING_BATCH_SIZE]
        batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
        total_batches = (len(documents) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} docs)...")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Create temp vectorstore and merge
            temp_vs = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(temp_vs)
    
    print(f"Embedding complete. Total documents: {len(documents)}")
    return vectorstore

