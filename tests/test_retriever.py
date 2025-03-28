import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retriever import Retriever
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant


class TestRetriever(unittest.TestCase):
    """Unit tests for the Retriever class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_index_path = os.path.join(self.temp_dir.name, "test_index.json")
        self.test_input_path = os.path.join(self.temp_dir.name, "test_employees.json")
        self.test_output_path = os.path.join(self.temp_dir.name, "test_enriched.json")

        # Sample index data
        self.test_index_data = {
            "collection_name": "formations",
            "documents": [
                {
                    "content": "Advanced project management training",
                    "metadata": {"id": "doc_1", "type": "course", "source": "internal"},
                },
                {
                    "content": "Communication skills workshop",
                    "metadata": {
                        "id": "doc_2",
                        "type": "workshop",
                        "source": "external",
                    },
                },
                {
                    "content": "Leadership and team management",
                    "metadata": {
                        "id": "doc_3",
                        "type": "seminar",
                        "source": "external",
                    },
                },
            ],
        }

        # Sample employees data
        self.test_employees = [
            {
                "employe": "John Doe",
                "evaluation": "Needs to improve project management skills",
                "score": 70,
                "keywords": ["project management"],
            },
            {
                "employe": "Jane Smith",
                "evaluation": "Great leadership but communication needs work",
                "score": 85,
                "keywords": ["leadership", "communication"],
            },
        ]

        # Write test data to files
        with open(self.test_index_path, "w", encoding="utf-8") as f:
            json.dump(self.test_index_data, f)

        with open(self.test_input_path, "w", encoding="utf-8") as f:
            json.dump(self.test_employees, f)

        # Create a patcher for OpenAI embeddings
        self.embeddings_patcher = patch("src.retriever.OpenAIEmbeddings")
        self.mock_embeddings = self.embeddings_patcher.start()

        # Create retriever instance
        self.retriever = Retriever()

    def tearDown(self):
        """Clean up test fixtures."""
        self.embeddings_patcher.stop()
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of Retriever."""
        # Assert
        self.assertIsNotNone(self.retriever)
        self.assertIsNone(self.retriever.vectorstore)
        self.mock_embeddings.assert_called_once()

    @patch("langchain_community.vectorstores.Qdrant.from_documents")
    def test_load_index(self, mock_from_documents):
        """Test loading index from a file."""
        # Arrange
        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore

        # Act
        result = self.retriever.load_index(self.test_index_path)

        # Assert
        self.assertEqual(result, mock_vectorstore)
        self.assertEqual(self.retriever.vectorstore, mock_vectorstore)
        mock_from_documents.assert_called_once()

        # Check the documents were created correctly
        args, kwargs = mock_from_documents.call_args
        documents = kwargs.get("documents")
        self.assertEqual(len(documents), 3)
        self.assertEqual(
            documents[0].page_content, "Advanced project management training"
        )
        self.assertEqual(documents[1].metadata["id"], "doc_2")

    @patch("langchain_community.vectorstores.Qdrant.similarity_search_with_score")
    def test_search_formations(self, mock_search):
        """Test searching for relevant training programs."""
        # Arrange
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = mock_search  # This line is new
        self.retriever.vectorstore = mock_vectorstore

        # Create mock documents with scores
        doc1 = Document(
            page_content="Advanced project management training",
            metadata={"id": "doc_1", "type": "course", "source": "internal"},
        )
        doc2 = Document(
            page_content="Communication skills workshop",
            metadata={"id": "doc_2", "type": "workshop", "source": "external"},
        )

        mock_search.return_value = [(doc1, 0.8), (doc2, 0.6)]

        # Act
        formations = self.retriever.search_formations("project management", top_k=2)

        # Assert
        self.assertEqual(len(formations), 2)
        mock_search.assert_called_once_with("project management", k=2)

        # Check formations are correctly formatted
        self.assertEqual(formations[0]["id"], "doc_1")
        self.assertEqual(
            formations[0]["content"], "Advanced project management training"
        )
        self.assertEqual(formations[0]["score"], 0.8)
        self.assertEqual(formations[1]["type"], "workshop")

    def test_search_formations_no_vectorstore(self):
        """Test searching without loading index first."""
        # Arrange
        self.retriever.vectorstore = None

        # Act & Assert
        with self.assertRaises(ValueError):
            self.retriever.search_formations("project management")

    def test_process_employee(self):
        """Test processing a single employee."""
        # Arrange
        self.retriever.vectorstore = MagicMock()

        # Mock search_formations to return predefined formations
        mock_formations = [
            {
                "id": "doc_1",
                "score": 0.8,
                "type": "course",
                "source": "internal",
                "content": "Advanced project management training",
            }
        ]

        self.retriever.search_formations = MagicMock(return_value=mock_formations)

        employee = self.test_employees[0]

        # Act
        result = self.retriever.process_employee(employee)

        # Assert
        self.retriever.search_formations.assert_called_once()
        self.assertEqual(result["employe"], "John Doe")
        self.assertEqual(result["formations_recommandees"], mock_formations)

        # Check query construction with keywords
        query_arg = self.retriever.search_formations.call_args[0][0]
        self.assertIn("project management", query_arg)

    @patch("json.dump")
    def test_process_employees(self, mock_dump):
        """Test processing all employees."""
        # Arrange
        self.retriever.vectorstore = MagicMock()

        # Mock process_employee to return predefined results
        def mock_process(employee):
            result = employee.copy()
            result["formations_recommandees"] = [{"id": "mock_formation"}]
            return result

        self.retriever.process_employee = MagicMock(side_effect=mock_process)

        # Act
        results = self.retriever.process_employees(
            self.test_input_path, self.test_output_path
        )

        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(self.retriever.process_employee.call_count, 2)
        mock_dump.assert_called_once()

        # Check each employee was processed
        self.assertIn("formations_recommandees", results[0])
        self.assertIn("formations_recommandees", results[1])

    @patch("langchain_community.vectorstores.Qdrant.from_documents")
    @patch(
        "src.retriever.Retriever.search_formations"
    )  # Changed from mocking Qdrant directly
    def test_integration(self, mock_search_formations, mock_from_documents):
        """Test the complete workflow."""
        # Arrange
        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore

        # Instead of mocking similarity_search_with_score, mock the search_formations method
        mock_search_formations.return_value = [
            {
                "id": "doc_1",
                "score": 0.8,
                "type": "course",
                "source": "internal",
                "content": "Advanced project management training",
            },
            {
                "id": "doc_2",
                "score": 0.6,
                "type": "workshop",
                "source": "external",
                "content": "Communication skills workshop",
            },
        ]

        # Act
        self.retriever.load_index(self.test_index_path)
        results = self.retriever.process_employees(
            self.test_input_path, self.test_output_path
        )

        # Assert
        mock_from_documents.assert_called_once()
        self.assertEqual(mock_search_formations.call_count, 2)  # Once for each employee
        self.assertEqual(len(results), 2)
        self.assertTrue(os.path.exists(self.test_output_path))


if __name__ == "__main__":
    unittest.main()
