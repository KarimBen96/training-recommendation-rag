"""
agent.py - Specialized retrieval agents module

Implements specialized agents for different document types
to enhance the recommendation quality.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


class Agent:
    """Agent specialized in a specific document type for training recommendations."""

    def __init__(self, document_type, retriever):
        """
        Initialize the agent with a specific document type.

        Args:
            document_type: Type of documents this agent specializes in
            retriever: An initialized Retriever instance
        """
        self.document_type = document_type
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Set the appropriate prompt based on document type
        if document_type == "programme de formation":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                En tant qu'expert en programmes de formation, vous devez recommander des formations spécifiques pour :
                - Employé(e) : {name}
                - Évaluation : {evaluation}
                - Score : {score}/100
                
                Programmes de formation disponibles :
                {formations}
                
                Rédigez une recommandation brève et précise du point de vue des programmes de formation uniquement.
                """
            )
        elif document_type == "meilleures pratiques":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                En tant qu'expert en meilleures pratiques RH, vous devez suggérer des méthodes d'amélioration pour :
                - Employé(e) : {name}
                - Évaluation : {evaluation}
                - Score : {score}/100
                
                Meilleures pratiques pertinentes :
                {formations}
                
                Rédigez une recommandation brève et précise du point de vue des meilleures pratiques uniquement.
                """
            )
        elif document_type == "étude de cas":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                En tant qu'analyste d'études de cas, vous devez fournir des conseils basés sur des expériences réelles pour :
                - Employé(e) : {name}
                - Évaluation : {evaluation}
                - Score : {score}/100
                
                Études de cas pertinentes :
                {formations}
                
                Rédigez une recommandation brève et précise du point de vue des études de cas uniquement.
                """
            )
        else:
            raise ValueError(f"Unknown document type: {document_type}")

        print(f"Agent initialized for document type: {document_type}")

    def process_employee(self, employee):
        """
        Process an employee evaluation and generate domain-specific recommendations.

        Args:
            employee: Dictionary containing employee evaluation data

        Returns:
            Enriched employee data with domain-specific recommendations
        """
        # Create query
        query = employee["evaluation"]
        if "keywords" in employee:
            query += " " + " ".join(employee["keywords"])

        # Filter results by document type
        formations = self.retriever.search_formations(query)
        filtered_formations = [f for f in formations if f["type"] == self.document_type]

        # Format formations
        formatted_formations = self._format_formations(filtered_formations)

        # Generate recommendation specific to this agent's domain
        variables = {
            "name": employee["employe"],
            "evaluation": employee["evaluation"],
            "score": employee["score"],
            "formations": formatted_formations,
        }

        prompt = self.prompt_template.format_messages(**variables)
        response = self.llm.invoke(prompt)
        recommendation = response.content

        # Return the result
        result = employee.copy()
        result[f"recommendation_{self.document_type}"] = recommendation
        result[f"formations_{self.document_type}"] = filtered_formations

        print(
            f"Generated {self.document_type} recommendation for {employee['employe']}"
        )
        return result

    def _format_formations(self, formations):
        """
        Format the training programs for the prompt.

        Args:
            formations: List of formation dictionaries

        Returns:
            Formatted string for prompt
        """
        if not formations:
            return "Aucun document pertinent trouvé."

        formatted = ""
        for i, formation in enumerate(formations, 1):
            formatted += f"\nFormation {i}:\n"
            formatted += f"- ID: {formation.get('id', 'Non spécifié')}\n"
            formatted += f"- Source: {formation.get('source', 'Non spécifiée')}\n"
            formatted += f"- Contenu: {formation.get('content', 'Non spécifié')}\n"
            formatted += f"- Score de pertinence: {formation.get('score', 0):.2f}\n"
        return formatted


# Test function
if __name__ == "__main__":
    from retriever import Retriever

    # Initialize retriever
    retriever = Retriever()
    retriever.load_index("data/qdrant_export.json")

    # Create an agent
    agent = Agent(document_type="programme de formation", retriever=retriever)

    # Test with sample data
    test_employee = {
        "employe": "Marie Dupont",
        "evaluation": "Besoins de renforcer la gestion de projet et la communication.",
        "score": 70,
        "keywords": ["gestion de projet", "communication"],
    }

    result = agent.process_employee(test_employee)
    print("\nGenerated recommendation:")
    print(result["recommendation_programme de formation"])
