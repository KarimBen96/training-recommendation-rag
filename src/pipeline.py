"""
pipeline.py - Main pipeline of the training recommendation system

Orchestrates the entire process of generating personalized training
recommendations using multiple specialized agents.
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
from agent import Agent

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
        self.multi_agent_enriched_file = os.path.join(
            data_dir, "multi_agent_enriched.json"
        )
        self.recommendations_file = os.path.join(data_dir, "recommendations.json")
        self.multi_agent_recommendations_file = os.path.join(
            data_dir, "recommendations_multi_agent.json"
        )

        print(f"Pipeline initialized with directory {data_dir}")

    def run(self, force_indexing=False, use_multi_agents=True):
        """
        Run the complete pipeline.

        Args:
            force_indexing: Whether to force re-indexing of the corpus
            use_multi_agents: Whether to use the multi-agent approach
        """
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

        if use_multi_agents:
            self._run_multi_agent_pipeline()
        else:
            self._run_single_agent_pipeline()

        # Display execution time
        elapsed_time = time.time() - start_time
        print(f"\nPipeline completed in {elapsed_time:.2f} seconds")

        # Display results summary
        self.show_summary()

        return self.recommendations_file

    def _run_single_agent_pipeline(self):
        """Run the original single-agent pipeline."""
        # Step 3: Search for relevant training programs
        print("\n--- STEP 3: RETRIEVING RELEVANT TRAINING PROGRAMS (SINGLE AGENT) ---")
        retriever = Retriever()
        retriever.load_index(self.index_export_file)
        retriever.process_employees(
            input_file=self.preprocessed_file, output_file=self.enriched_file
        )

        # Step 4: Generate recommendations
        print("\n--- STEP 4: GENERATING RECOMMENDATIONS (SINGLE AGENT) ---")
        generator = Generator()
        generator.process_employees(
            input_file=self.enriched_file, output_file=self.recommendations_file
        )

    def _run_multi_agent_pipeline(self):
        """Run the multi-agent pipeline with specialized agents."""
        # Step 3: Process with specialized agents
        print("\n--- STEP 3: RÉCUPÉRATION AVEC AGENTS SPÉCIALISÉS ---")

        # Initialize retriever (shared by all agents)
        retriever = Retriever()
        retriever.load_index(self.index_export_file)

        # Initialize agents
        print("Initialisation des agents spécialisés...")
        training_agent = Agent(
            document_type="programme de formation", retriever=retriever
        )
        practices_agent = Agent(
            document_type="meilleures pratiques", retriever=retriever
        )
        case_study_agent = Agent(document_type="étude de cas", retriever=retriever)

        # Load preprocessed employees
        with open(self.preprocessed_file, "r", encoding="utf-8") as f:
            employees = json.load(f)

        # Process with each agent
        print(f"Traitement de {len(employees)} employés avec les agents spécialisés...")
        enriched_employees = []
        for employee in employees:
            print(f"Traitement de l'employé(e) : {employee['employe']}")
            # Get recommendations from each agent
            result = employee.copy()
            result = training_agent.process_employee(result)
            result = practices_agent.process_employee(result)
            result = case_study_agent.process_employee(result)
            enriched_employees.append(result)
            print(f"Employé(e) {employee['employe']} traité(e) avec succès")

        # Save enriched data
        with open(self.multi_agent_enriched_file, "w", encoding="utf-8") as f:
            json.dump(enriched_employees, f, ensure_ascii=False, indent=2)

        print(
            f"Données enrichies par agents multiples sauvegardées dans {self.multi_agent_enriched_file}"
        )

        print(f"Multi-agent enriched data saved to {self.multi_agent_enriched_file}")

        # Step 4: Generate combined recommendations
        print("\n--- STEP 4: GENERATING COMBINED RECOMMENDATIONS ---")
        generator = Generator()

        # Load enriched employees data
        with open(self.multi_agent_enriched_file, "r", encoding="utf-8") as f:
            enriched_employees = json.load(f)

        # Generate combined recommendations for each employee
        final_recommendations = []
        for employee in enriched_employees:
            recommendation = generator.generate_combined_recommendation(employee)
            final_recommendations.append(recommendation)

        # Save the final recommendations
        with open(self.multi_agent_recommendations_file, "w", encoding="utf-8") as f:
            json.dump(final_recommendations, f, ensure_ascii=False, indent=2)

        print(
            f"Recommandations finales sauvegardées dans {self.multi_agent_recommendations_file}"
        )

    def show_summary(self):
        """Display a summary of the generated recommendations."""
        try:
            # Determine which file to load based on the mode
            if os.path.exists(self.multi_agent_recommendations_file):
                recommendations_file = self.multi_agent_recommendations_file
            else:
                recommendations_file = self.recommendations_file

            # Load recommendations
            with open(recommendations_file, "r", encoding="utf-8") as f:
                recommendations = json.load(f)

            print("\n" + "=" * 70)
            print(f"RÉSUMÉ DES RECOMMANDATIONS ({len(recommendations)} employés)")
            print("=" * 70)

            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['employe']}")
                print(f"   Score : {rec['score']}/100")
                print(f"   Évaluation : {rec['evaluation']}")

                # Display the beginning of the recommendation
                recommendation = rec.get("recommendation", "Non disponible")
                excerpt = (
                    recommendation[:100] + "..."
                    if len(recommendation) > 100
                    else recommendation
                )
                print(f"   Extrait : {excerpt}")

            print("\n" + "=" * 70)
            print(f"Fichier de recommandations : {recommendations_file}")
            print("=" * 70)

        except Exception as e:
            print(f"Erreur lors de l'affichage du résumé : {e}")


# Command line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for training recommendations"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--force-indexing", action="store_true", help="Force corpus reindexing"
    )
    parser.add_argument(
        "--single-agent",
        action="store_true",
        help="Use single agent instead of multi-agent",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(data_dir=args.data_dir)
    pipeline.run(
        force_indexing=args.force_indexing, use_multi_agents=not args.single_agent
    )
