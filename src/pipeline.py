"""
pipeline.py - Main pipeline of the training recommendation system

Orchestrates the entire process of generating personalized training
recommendations.
"""

import os
import argparse
import json
import time
from dotenv import load_dotenv

# Import system modules
from preprocessor import Preprocessor
from corpus_indexer_mem import CorpusIndexer
from retriever import Retriever
from generator import Generator

# Load environment variables
load_dotenv()


class RAGPipeline:
    """Complete pipeline for training recommendations."""

    def __init__(self, data_dir="data"):
        """Initialize the pipeline with file paths."""
        # Define file paths
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.employees_file = os.path.join(data_dir, "employe.json")
        self.corpus_file = os.path.join(data_dir, "formation.json")
        self.preprocessed_file = os.path.join(data_dir, "preprocessed_employees.json")
        self.index_export_file = os.path.join(data_dir, "qdrant_export.json")
        self.enriched_file = os.path.join(data_dir, "enriched_employees.json")
        self.recommendations_file = os.path.join(data_dir, "recommendations.json")

        print(f"Pipeline initialized with directory {data_dir}")

    def run(self, force_indexing=False):
        """Run the complete pipeline."""
        start_time = time.time()
        print("Starting pipeline...")

        # Step 1: Preprocess evaluations
        print("\n--- STEP 1: PREPROCESSING EVALUATIONS ---")
        preprocessor = Preprocessor()
        preprocessor.process_employees(
            input_file=self.employees_file, output_file=self.preprocessed_file
        )

        # Step 2: Index the corpus (if needed)
        print("\n--- STEP 2: CORPUS INDEXING ---")
        index_exists = os.path.exists(self.index_export_file)

        if force_indexing or not index_exists:
            indexer = CorpusIndexer(openai_api_key=os.getenv("OPENAI_API_KEY"))
            documents = indexer.load_corpus(self.corpus_file)
            indexer.create_index(documents=documents, collection_name="formations")
            indexer.save_index(self.index_export_file)
            indexer.close()
        else:
            print(
                f"Existing index found in {self.index_export_file}, skipping indexing"
            )

        # Step 3: Search for relevant training programs
        print("\n--- STEP 3: RETRIEVING RELEVANT TRAINING PROGRAMS ---")
        retriever = Retriever()
        # Add this line to load the index before searching
        retriever.load_index(self.index_export_file)  # THIS LINE WAS MISSING
        retriever.process_employees(
            input_file=self.preprocessed_file, output_file=self.enriched_file
        )

        # Step 4: Generate recommendations
        print("\n--- STEP 4: GENERATING RECOMMENDATIONS ---")
        generator = Generator()
        generator.process_employees(
            input_file=self.enriched_file, output_file=self.recommendations_file
        )

        # Display execution time
        elapsed_time = time.time() - start_time
        print(f"\nPipeline completed in {elapsed_time:.2f} seconds")

        # Display results summary
        self.show_summary()

        return self.recommendations_file

    def show_summary(self):
        """Display a summary of the generated recommendations."""
        try:
            # Load recommendations
            with open(self.recommendations_file, "r", encoding="utf-8") as f:
                recommendations = json.load(f)

            print("\n" + "=" * 70)
            print(f"RECOMMENDATION SUMMARY ({len(recommendations)} employees)")
            print("=" * 70)

            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['employe']}")
                print(f"   Score: {rec['score']}/100")
                print(f"   Evaluation: {rec['evaluation']}")

                # Display the beginning of the recommendation
                recommendation = rec.get("recommendation", "Not available")
                excerpt = (
                    recommendation[:100] + "..."
                    if len(recommendation) > 100
                    else recommendation
                )
                print(f"   Excerpt: {excerpt}")

            print("\n" + "=" * 70)
            print(f"Recommendations file: {self.recommendations_file}")
            print("=" * 70)

        except Exception as e:
            print(f"Error displaying summary: {e}")


# Command line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for training recommendations"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--force-indexing", action="store_true", help="Force corpus reindexing"
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(data_dir=args.data_dir)
    pipeline.run(force_indexing=args.force_indexing)
