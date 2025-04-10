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
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Set the appropriate prompt based on document type
        if document_type == "programme de formation":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                You do all your work in French, and you must answer in French.
                As a training program expert, recommend specific trainings for:
                - Employee: {name}
                - Evaluation: {evaluation}
                - Score: {score}/100
                
                Available training programs:
                {formations}
                
                Recommendation (only from the training program perspective):
                """
            )
        elif document_type == "meilleures pratiques":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                As a best practices expert, suggest improvement methods for:
                - Employee: {name}
                - Evaluation: {evaluation}
                - Score: {score}/100
                
                Relevant best practices:
                {formations}
                
                Recommendation (only from the best practices perspective):
                """
            )
        elif document_type == "étude de cas":
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                As a case study analyst, provide insights based on real experiences for:
                - Employee: {name}
                - Evaluation: {evaluation}
                - Score: {score}/100
                
                Relevant case studies:
                {formations}
                
                Recommendation (only from the case study perspective):
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
            "formations": formatted_formations
        }
        
        prompt = self.prompt_template.format_messages(**variables)
        response = self.llm.invoke(prompt)
        recommendation = response.content
        
        # Return the result
        result = employee.copy()
        result[f"recommendation_{self.document_type}"] = recommendation
        result[f"formations_{self.document_type}"] = filtered_formations
        
        print(f"Generated {self.document_type} recommendation for {employee['employe']}")
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
            return "No relevant documents found."
            
        formatted = ""
        for i, formation in enumerate(formations, 1):
            formatted += f"\nItem {i}:\n"
            formatted += f"- ID: {formation.get('id', 'Not specified')}\n"
            formatted += f"- Source: {formation.get('source', 'Not specified')}\n"
            formatted += f"- Content: {formation.get('content', 'Not specified')}\n"
            formatted += f"- Relevance score: {formation.get('score', 0):.2f}\n"
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