import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generator import Generator


class TestGenerator(unittest.TestCase):
    """Unit tests for the Generator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_input_path = os.path.join(self.temp_dir.name, "test_enriched.json")
        self.test_output_path = os.path.join(
            self.temp_dir.name, "test_recommendations.json"
        )

        # Sample enriched employee data
        self.test_employees = [
            {
                "employe": "John Doe",
                "evaluation": "Needs to improve project management skills",
                "score": 70,
                "keywords": ["project management"],
                "formations_recommandees": [
                    {
                        "id": "F001",
                        "score": 0.85,
                        "type": "course",
                        "source": "internal",
                        "content": "Advanced Project Management Certification",
                    },
                    {
                        "id": "F002",
                        "score": 0.75,
                        "type": "workshop",
                        "source": "external",
                        "content": "Agile Project Management Fundamentals",
                    },
                ],
            },
            {
                "employe": "Jane Smith",
                "evaluation": "Great leadership but communication needs work",
                "score": 85,
                "keywords": ["leadership", "communication"],
                "formations_recommandees": [
                    {
                        "id": "F003",
                        "score": 0.90,
                        "type": "seminar",
                        "source": "internal",
                        "content": "Advanced Communication Skills for Leaders",
                    }
                ],
            },
        ]

        # Write test data to file
        with open(self.test_input_path, "w", encoding="utf-8") as f:
            json.dump(self.test_employees, f)

        # Patch the OpenAI chat model
        self.openai_patcher = patch("src.generator.ChatOpenAI")
        self.mock_openai = self.openai_patcher.start()

        # Create mock response
        mock_response = MagicMock()
        mock_response.content = "Mock recommendation report"

        # Set up the chat model to return the mock response
        self.mock_chat = MagicMock()
        self.mock_chat.invoke.return_value = mock_response
        self.mock_openai.return_value = self.mock_chat

        # Create generator instance
        self.generator = Generator(model_name="gpt-3.5-turbo")

    def tearDown(self):
        """Clean up test fixtures."""
        self.openai_patcher.stop()
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of Generator."""
        # Assert
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.prompt_template)
        self.mock_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def test_format_formations(self):
        """Test formatting of training programs."""
        # Arrange
        formations = [
            {
                "id": "F001",
                "score": 0.85,
                "type": "course",
                "source": "internal",
                "content": "Advanced Project Management",
            },
            {
                "id": "F002",
                "score": 0.75,
                "type": "workshop",
                "source": "external",
                "content": "Communication Skills",
            },
        ]

        # Act
        formatted = self.generator.format_formations(formations)

        # Assert
        self.assertIn("Training 1:", formatted)
        self.assertIn("ID: F001", formatted)
        self.assertIn("Training 2:", formatted)
        self.assertIn("Type: workshop", formatted)
        self.assertIn("Relevance score: 0.75", formatted)

    def test_generate_recommendation(self):
        """Test generating a recommendation for an employee."""
        # Arrange
        employee = self.test_employees[0]

        # Act
        result = self.generator.generate_recommendation(employee)

        # Assert
        self.mock_chat.invoke.assert_called_once()
        self.assertEqual(result["employe"], "John Doe")
        self.assertEqual(result["recommendation"], "Mock recommendation report")

        # Check that the prompt was properly constructed
        prompt_args = self.mock_chat.invoke.call_args[0][0]
        prompt_content = str(prompt_args)
        self.assertIn("John Doe", prompt_content)
        self.assertIn("project management skills", prompt_content)
        self.assertIn("70", prompt_content)

    def test_generate_recommendation_no_formations(self):
        """Test generating a recommendation when no training programs are available."""
        # Arrange
        employee = {
            "employe": "Alex Johnson",
            "evaluation": "Needs to improve technical skills",
            "score": 65,
            "keywords": ["technical skills"],
            # No formations_recommandees key
        }

        # Act
        result = self.generator.generate_recommendation(employee)

        # Assert
        self.assertEqual(result["employe"], "Alex Johnson")
        self.assertEqual(result["recommendation"], "Mock recommendation report")

    @patch("json.dump")
    def test_process_employees(self, mock_dump):
        """Test processing all employees."""
        # Act
        results = self.generator.process_employees(
            self.test_input_path, self.test_output_path
        )

        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(self.mock_chat.invoke.call_count, 2)
        mock_dump.assert_called_once()

        # Check each employee was processed
        self.assertIn("recommendation", results[0])
        self.assertIn("recommendation", results[1])

    def test_process_employees_real_output(self):
        """Test that the output file is actually created and contains the expected data."""
        # Act
        results = self.generator.process_employees(
            self.test_input_path, self.test_output_path
        )

        # Assert
        self.assertTrue(os.path.exists(self.test_output_path))

        # Load the output file
        with open(self.test_output_path, "r", encoding="utf-8") as f:
            saved_results = json.load(f)

        # Check structure and content
        self.assertEqual(len(saved_results), 2)
        self.assertEqual(saved_results[0]["employe"], "John Doe")
        self.assertEqual(saved_results[1]["employe"], "Jane Smith")
        self.assertEqual(
            saved_results[0]["recommendation"], "Mock recommendation report"
        )

    @patch("builtins.open", side_effect=IOError("Mock file error"))
    def test_process_employees_file_error(self, mock_open):
        """Test handling of file errors during processing."""
        # Act & Assert
        with self.assertRaises(IOError):
            self.generator.process_employees(
                "nonexistent_file.json", self.test_output_path
            )


if __name__ == "__main__":
    unittest.main()
