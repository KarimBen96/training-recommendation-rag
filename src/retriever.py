"""
retriever.py - Information retrieval module

Searches for relevant training programs in the document corpus
to enrich employee evaluations.
"""

import json
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document

# Load environment variables
load_dotenv()


class Retriever:
    """Retrieval of relevant training programs for employee evaluations."""

    def __init__(self):
        """Initialize the retriever with OpenAI embeddings."""
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("MODEL_NAME_EMBEDDING"), openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # The vectorstore will be loaded later
        self.vectorstore = None

    def load_index(self, export_path="data/qdrant_export.json"):
        """Load the index from the export file."""
        # Load data
        with open(export_path, "r", encoding="utf-8") as f:
            export_data = json.load(f)

        # Recreate documents
        documents = []
        for doc_data in export_data["documents"]:
            doc = Document(
                page_content=doc_data["content"], metadata=doc_data["metadata"]
            )
            documents.append(doc)

        # Recreate the vectorstore
        self.vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="formations",
            location=":memory:",
        )

        print(f"Index loaded with {len(documents)} documents")
        return self.vectorstore

    def search_formations(self, query, top_k=3):
        """Search for the most relevant training programs."""
        if not self.vectorstore:
            raise ValueError("Please first load the index with load_index()")

        # Similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        # Format results
        formations = []
        for doc, score in results:
            formation = {
                "id": doc.metadata.get("id"),
                "score": float(score),
                "type": doc.metadata.get("type"),
                "source": doc.metadata.get("source"),
                "content": doc.page_content,
            }
            formations.append(formation)

        return formations

    def process_employee(self, employee):
        """Enrich an employee with recommended training programs."""
        # Create a query from the evaluation
        query = employee["evaluation"]
        if "keywords" in employee:
            query += " " + " ".join(employee["keywords"])

        # Search for relevant training programs
        formations = self.search_formations(query)

        # Enrich employee data
        result = employee.copy()
        result["formations_recommandees"] = formations

        return result

    def process_employees(self, input_file, output_file):
        """Process all employees and save the results."""
        # Load data
        with open(input_file, "r", encoding="utf-8") as f:
            employees = json.load(f)

        # Process each employee
        results = []
        for employee in employees:
            enriched = self.process_employee(employee)
            results.append(enriched)

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Recommendations saved to {output_file}")
        return results


# Usage example
if __name__ == "__main__":
    retriever = Retriever()
    retriever.load_index("data/qdrant_export.json")
    retriever.process_employees(
        input_file="data/employe_preprocessed.json",
        output_file="data/employe_enriched.json",
    )
