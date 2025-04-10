"""
generator.py - Module for generating personalized recommendations

Generates personalized training recommendations
using enriched employee data.
"""

import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


class Generator:
    """Generates personalized training recommendations using an LLM."""

    def __init__(self, model_name="gpt-3.5-turbo"):
        """Initialize the generator with the language model."""
        # Initialize the LLM model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Vous êtes un expert RH spécialisé dans le développement professionnel.
            
            Informations sur l'employé(e) :
            - Nom : {nom}
            - Évaluation : {evaluation}
            - Score : {score}/100
            - Priorité de formation : {priorite}
            
            Programmes de formation pertinents disponibles :
            {formations}
            
            Rédigez un rapport de recommandation qui :
            1. Résume les besoins de développement identifiés
            2. Propose un plan de formation personnalisé basé sur les programmes disponibles
            3. Explique comment ces programmes répondent aux besoins spécifiques de l'employé(e)
            4. Suggère un calendrier de formation approprié
            
            Le rapport doit être professionnel, concis et directement applicable.
            Assurez-vous que la réponse soit en français.
            """
        )

        print(f"Generator initialized with model {model_name}")

    def format_formations(self, formations):
        """Format the training programs for the prompt."""
        formatted = ""
        for i, formation in enumerate(formations, 1):
            formatted += f"\nFormation {i}:\n"
            formatted += f"- ID: {formation.get('id', 'Non spécifié')}\n"
            formatted += f"- Type: {formation.get('type', 'Non spécifié')}\n"
            formatted += f"- Source: {formation.get('source', 'Non spécifiée')}\n"
            formatted += f"- Contenu: {formation.get('content', 'Non spécifié')}\n"
            formatted += f"- Score de pertinence: {formation.get('score', 0):.2f}\n"
        return formatted

    def generate_recommendation(self, employee):
        """Generate a recommendation for an employee."""
        # Extract employee information
        name = employee["employe"]
        evaluation = employee["evaluation"]
        score = employee["score"]
        priority = employee.get("priority", "Medium")

        # Format recommended training programs
        formations = employee.get("formations_recommandees", [])
        formations_text = self.format_formations(formations)

        # Prepare variables for the prompt
        variables = {
            "nom": name,
            "evaluation": evaluation,
            "score": score,
            "priorite": priority,
            "formations": formations_text,
        }

        # Generate the recommendation
        prompt = self.prompt_template.format_messages(**variables)
        response = self.llm.invoke(prompt)
        recommendation = response.content

        # Add the recommendation to the employee data
        result = employee.copy()
        result["recommendation"] = recommendation

        print(f"Recommendation generated for {name}")

        return result

    def generate_combined_recommendation(self, employee):
        """Combine recommendations from different agents."""
        # Extract partial recommendations
        training_rec = employee.get("recommendation_programme de formation", "")
        practices_rec = employee.get("recommendation_meilleures pratiques", "")
        case_study_rec = employee.get("recommendation_étude de cas", "")

        # Create a prompt to merge recommendations
        prompt = ChatPromptTemplate.from_template(
            """
            Vous êtes un expert RH chargé de créer une recommandation de formation cohérente
            en combinant les analyses spécialisées suivantes :
            
            PROGRAMMES DE FORMATION :
            {training_rec}
            
            MEILLEURES PRATIQUES :
            {practices_rec}
            
            ÉTUDES DE CAS :
            {case_study_rec}
            
            EMPLOYÉ(E) :
            - Nom : {name}
            - Évaluation : {evaluation}
            - Score : {score}/100
            
            Rédigez une recommandation complète qui intègre harmonieusement ces trois perspectives.
            La recommandation doit :
            1. Être rédigée en français
            2. Être professionnelle et concise (maximum 300 mots)
            3. Proposer un plan d'action clair et réalisable
            4. Être directement applicable
            """
        )

        # Prepare variables
        variables = {
            "training_rec": training_rec,
            "practices_rec": practices_rec,
            "case_study_rec": case_study_rec,
            "name": employee["employe"],
            "evaluation": employee["evaluation"],
            "score": employee["score"],
        }

        # Generate the merged recommendation
        formatted_prompt = prompt.format_messages(**variables)
        response = self.llm.invoke(formatted_prompt)

        # Add the final recommendation
        result = employee.copy()
        result["recommendation"] = response.content

        print(f"Combined recommendation generated for {employee['employe']}")
        return result

    def process_employees(self, input_file, output_file):
        """Generate recommendations for all employees."""
        # Load the enriched data
        with open(input_file, "r", encoding="utf-8") as f:
            employees = json.load(f)

        # Generate recommendations for each employee
        results = []
        for employee in employees:
            recommendation = self.generate_recommendation(employee)
            results.append(recommendation)

        # Save the results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Recommendations saved in {output_file}")
        return results

    def process_multi_agent_employees(self, input_file, output_file):
        """Generate combined recommendations from multiple agents for all employees."""
        # Load the enriched data
        with open(input_file, "r", encoding="utf-8") as f:
            employees = json.load(f)

        # Generate combined recommendations for each employee
        results = []
        for employee in employees:
            recommendation = self.generate_combined_recommendation(employee)
            results.append(recommendation)

        # Save the results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Combined recommendations saved in {output_file}")
        return results


# Usage example
if __name__ == "__main__":
    generator = Generator()

    test_employee = {
        "employe": "Marie Dupont",
        "evaluation": "Besoins de renforcer la gestion de projet et la communication.",
        "score": 70,
        "keywords": ["gestion de projet", "communication"],
        "recommendation_programme de formation": "Marie would benefit from our Advanced Project Management course...",
        "recommendation_meilleures pratiques": "Based on best practices, Marie should focus on improving her meeting facilitation skills...",
        "recommendation_étude de cas": "Similar cases show that communication training paired with real project experience yields best results...",
    }

    test_recommendation = generator.generate_combined_recommendation(test_employee)
    print("\nCombined recommendation:")
    print(test_recommendation["recommendation"])
