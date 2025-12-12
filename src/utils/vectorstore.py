import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

def create_vectorstore(documents: List[Document]) -> FAISS:
    """Creates a FAISS vector store from a list of documents."""
    # Use GitHub Models OpenAI token for embeddings
    api_key = os.getenv("GITHUB_TOKEN_OPENAI")
    if not api_key:
        raise ValueError("GITHUB_TOKEN_OPENAI environment variable not set")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url="https://models.inference.ai.azure.com"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

