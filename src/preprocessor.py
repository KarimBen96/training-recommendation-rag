import json
from langchain.schema import Document
from typing import Dict, List, Tuple, Any
import os


class Preprocessor:
    """
    A class to preprocess employee data.

    Attributes
    ----------
    model : SentenceTransformer
        SentenceTransformer model that can be used to map sentences/text to embeddings.
    translator : GoogleTranslator
        Translator object to translate keywords that are not in English.

    Methods
    -------
    process(keyword, article):
        Process keyword by creating its query based on a template and by embedding it.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the Preprocessor object.
        """
        self.competences_keywords = {
            "project management": ["project", "planning", "organization"],
            "communication": ["communication", "expression", "clarity"],
            "leadership": ["leadership", "management", "initiative", "decision"],
            "teamwork": ["team", "collaboration", "interpersonal"],
            "technical skills": ["technical", "tool", "computing"],
            "time management": ["time", "deadline", "priority"],
            "stress management": ["stress", "pressure", "emotion"],
            "autonomy": ["autonomy", "independence", "initiative"],
            "innovation": ["innovation", "creativity", "idea"]
        }

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts relevant keywords from an evaluation text.

        Args:
            text: Evaluation text

        Returns:
            List of keywords
        """
        text_lower = text.lower()
        keywords = []

        # Search for keywords related to competencies
        for competence, terms in self.competences_keywords.items():
            if any(term in text_lower for term in terms):
                keywords.append(competence)

        return keywords

    def process_evaluation(self, evaluations_filename: str) -> List[Document]:
        """
        Processes evaluations from a JSON file.
        
        Args:
            evaluations_filename: Path to the JSON file containing employee evaluations
            
        Returns:
            List of processed evaluations
        """
        with open(evaluations_filename, "r", encoding="utf-8") as f:
            evaluations = json.load(f)

        processed_evaluations = []
        for evaluation in evaluations:
            # Extract main information
            employee_name = evaluation.get("employe", "")
            evaluation_text = evaluation.get("evaluation", "")
            score = evaluation.get("score", 0)

            # Enrich data by text analysis
            keywords = self.extract_keywords(evaluation_text)
            # categories = self.categorize_evaluation(evaluation_text)
            # sentiment = self.analyze_sentiment(evaluation_text, score)

            # Add enriched data
            processed_eval = {
                "employe": employee_name,
                "evaluation": evaluation_text,
                "score": score,
                "keywords": keywords,
                # "priority": "high" if score < 65 else ("medium" if score < 75 else "low")
            }
            processed_evaluations.append(processed_eval)

        return processed_evaluations

    def to_documents(
        self, processed_evaluations: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Converts enriched evaluations to LangChain Documents.

        Args:
            processed_evaluations: List of preprocessed evaluations

        Returns:
            List of LangChain Documents
        """
        documents = []

        for eval_data in processed_evaluations:
            doc = Document(
                page_content=eval_data["evaluation"],
                metadata={
                    "employe": eval_data["employe"],
                    "score": eval_data["score"],
                    "keywords": eval_data["keywords"],
                    # "priority": eval_data["priority"],
                    "type": "evaluation",
                },
            )
            documents.append(doc)

        return documents

    def save_processed_data(
        self, processed_data: List[Dict[str, Any]], output_path: str
    ) -> None:
        """
        Saves preprocessed data to a JSON file.

        Args:
            processed_data: Preprocessed data to save
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Preprocessed data saved to: {output_path}")


    def process_employees(self, input_file: str, output_file: str) -> None:
        """
        Process all employees data from the input file and save to output file.
        
        Args:
            input_file: Path to the input JSON file with employee evaluations
            output_file: Path to save the processed data
        """
        print(f"Processing employee data from {input_file}")
        
        # Process evaluations
        processed_data = self.process_evaluation(input_file)
        
        # Save the processed data
        self.save_processed_data(processed_data, output_file)
        
        print(f"Processed {len(processed_data)} employee records.")
        
        return processed_data


if __name__ == "__main__":
    # File paths
    input_file = "data/employe.json"
    output_file = "data/employe_preprocessed.json"
    
    # Create preprocessor
    preprocessor = Preprocessor()
    
    # Process data
    processed_data = preprocessor.process_evaluation(input_file)
    
    # Save preprocessed data
    preprocessor.save_processed_data(processed_data, output_file)
    
    # Convert to LangChain documents
    documents = preprocessor.to_documents(processed_data)
    
    print(f"Preprocessing completed. {len(documents)} documents created.")
    
    # Display an example
    if documents:
        print("\nExample of preprocessed evaluation:")
        print(f"Employee: {documents[0].metadata['employe']}")
        print(f"Evaluation: {documents[0].page_content}")
        print(f"Score: {documents[0].metadata['score']}")
        # print(f"Keywords: {documents[0].metadata['keywords']}")
        # print(f"Priority: {documents[0].metadata['priority']}")