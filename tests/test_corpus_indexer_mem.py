import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
import qdrant_client

# Import the class to test
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.corpus_indexer_mem import CorpusIndexer


class TestCorpusIndexer(unittest.TestCase):
    """Unit tests for the CorpusIndexer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_corpus_path = os.path.join(self.temp_dir.name, "test_corpus.json")
        self.test_export_path = os.path.join(self.temp_dir.name, "test_export.json")

        # Create sample test data
        self.test_corpus = [
            {
                "content": "Project management course covering agile methodologies",
                "type": "course",
                "source": "internal",
            },
            {
                "content": "Communication skills workshop for technical teams",
                "type": "workshop",
                "source": "external",
            },
        ]

        # Write test data to file
        with open(self.test_corpus_path, "w", encoding="utf-8") as f:
            json.dump(self.test_corpus, f)

        # Create an instance with mocked API key
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            self.indexer = CorpusIndexer(openai_api_key="test-key")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("src.corpus_indexer_mem.OpenAIEmbeddings")  # Changed mock path
    def test_init(self, mock_embeddings):
        """Test initialization of CorpusIndexer."""
        # Arrange & Act
        indexer = CorpusIndexer(openai_api_key="test-key")
        
        # Assert
        self.assertIsNotNone(indexer)
        self.assertIsNone(indexer.qdrant_client)
        self.assertIsNone(indexer.vectorstore)
        self.assertEqual(indexer.corpus_documents, [])
        
        # Check that the mock was called without worrying about exact parameters
        mock_embeddings.assert_called_once()
        
        # Or check just that it was called with the key we expect
        args, kwargs = mock_embeddings.call_args
        self.assertEqual(kwargs.get("openai_api_key"), "test-key")
        self.assertEqual(kwargs.get("model"), "text-embedding-3-small")

    def test_load_corpus(self):
        """Test loading corpus from file."""
        # Act
        documents = self.indexer.load_corpus(self.test_corpus_path)

        # Assert
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(self.indexer.corpus_documents), 2)
        self.assertEqual(documents[0].page_content, self.test_corpus[0]["content"])
        self.assertEqual(documents[1].page_content, self.test_corpus[1]["content"])
        self.assertEqual(documents[0].metadata["type"], self.test_corpus[0]["type"])
        self.assertEqual(documents[1].metadata["source"], self.test_corpus[1]["source"])
        self.assertEqual(documents[0].metadata["id"], "doc_1")
        self.assertEqual(documents[1].metadata["id"], "doc_2")

    @patch("qdrant_client.QdrantClient")
    @patch("langchain_community.vectorstores.Qdrant.from_documents")
    def test_create_index(self, mock_from_documents, mock_qdrant_client):
        """Test creating index in memory."""
        # Arrange
        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore
        documents = [
            Document(page_content="Test content 1", metadata={"id": "doc_1"}),
            Document(page_content="Test content 2", metadata={"id": "doc_2"}),
        ]

        # Act
        result = self.indexer.create_index(documents, "test_collection")

        # Assert
        self.assertEqual(result, mock_vectorstore)
        self.assertEqual(self.indexer.vectorstore, mock_vectorstore)
        mock_qdrant_client.assert_called_once_with(location=":memory:")
        mock_from_documents.assert_called_once_with(
            documents=documents,
            embedding=self.indexer.embeddings,
            collection_name="test_collection",
            location=":memory:",
        )

    def test_save_index(self):
        """Test saving index to file."""
        # Arrange
        self.indexer.corpus_documents = [
            Document(page_content="Test content 1", metadata={"id": "doc_1"}),
            Document(page_content="Test content 2", metadata={"id": "doc_2"}),
        ]
        mock_vectorstore = MagicMock()
        mock_vectorstore.collection_name = "test_collection"
        self.indexer.vectorstore = mock_vectorstore

        # Act
        self.indexer.save_index(self.test_export_path)

        # Assert
        self.assertTrue(os.path.exists(self.test_export_path))
        with open(self.test_export_path, "r", encoding="utf-8") as f:
            export_data = json.load(f)
            self.assertEqual(len(export_data["documents"]), 2)
            self.assertEqual(export_data["collection_name"], "test_collection")
            self.assertEqual(export_data["documents"][0]["content"], "Test content 1")
            self.assertEqual(export_data["documents"][1]["metadata"]["id"], "doc_2")

    def test_save_index_no_documents(self):
        """Test saving index when no documents are present."""
        # Arrange
        self.indexer.corpus_documents = []
        self.indexer.vectorstore = None

        # Act
        self.indexer.save_index(self.test_export_path)

        # Assert
        self.assertFalse(os.path.exists(self.test_export_path))

    @patch("qdrant_client.QdrantClient")
    def test_close(self, mock_client):
        """Test closing the client."""
        # Arrange
        self.indexer.qdrant_client = mock_client

        # Act
        self.indexer.close()

        # Assert
        self.assertIsNone(self.indexer.qdrant_client)

    def test_close_no_client(self):
        """Test closing when no client exists."""
        # Arrange
        self.indexer.qdrant_client = None

        # Act & Assert (should not raise exception)
        self.indexer.close()
        self.assertIsNone(self.indexer.qdrant_client)

    @patch("qdrant_client.QdrantClient")
    @patch("langchain_community.vectorstores.Qdrant.from_documents")
    def test_integration(self, mock_from_documents, mock_qdrant_client):
        """Test the complete workflow."""
        # Arrange
        mock_vectorstore = MagicMock()
        mock_vectorstore.collection_name = "formations"
        mock_from_documents.return_value = mock_vectorstore

        # Act
        documents = self.indexer.load_corpus(self.test_corpus_path)
        self.indexer.create_index(documents)
        self.indexer.save_index(self.test_export_path)
        self.indexer.close()

        # Assert
        self.assertTrue(os.path.exists(self.test_export_path))
        with open(self.test_export_path, "r", encoding="utf-8") as f:
            export_data = json.load(f)
            self.assertEqual(len(export_data["documents"]), 2)
            self.assertEqual(export_data["collection_name"], "formations")


if __name__ == "__main__":
    unittest.main()
