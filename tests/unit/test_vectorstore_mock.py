import unittest
from unittest.mock import patch, MagicMock
import os
from langchain_core.documents import Document
from src.utils.vectorstore import create_vectorstore

from langchain_community.vectorstores import FAISS

class TestVectorStore(unittest.TestCase):

    @patch('src.utils.vectorstore.FAISS')
    @patch('src.utils.vectorstore.OpenAIEmbeddings')
    def test_create_vectorstore_with_mock(self, mock_openai_embeddings, mock_faiss):
        # Set the environment variable for the test
        os.environ["GITHUB_TOKEN_OPENAI"] = "test_api_key"

        # Mock the return value of the embeddings and FAISS
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore_instance = MagicMock(spec=FAISS)
        mock_faiss.from_documents.return_value = mock_vectorstore_instance

        # Create some dummy documents
        documents = [Document(page_content="This is a test document.")]

        # Call the function to be tested
        vectorstore = create_vectorstore(documents)

        # Assert that OpenAIEmbeddings was called with the correct parameters
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test_api_key",
            base_url="https://models.inference.ai.azure.com"
        )

        # Assert that FAISS.from_documents was called with the documents and the mocked embeddings
        mock_faiss.from_documents.assert_called_once_with(documents, mock_embeddings_instance)

        # Assert that the returned vectorstore is the mocked one
        self.assertIs(vectorstore, mock_vectorstore_instance)

        # Clean up the environment variable
        del os.environ["GITHUB_TOKEN_OPENAI"]
