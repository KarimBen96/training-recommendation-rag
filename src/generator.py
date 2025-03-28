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
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an HR expert specializing in professional development.
            
            Employee information:
            - Name: {nom}
            - Evaluation: {evaluation}
            - Score: {score}/100
            - Training priority: {priorite}
            
            Relevant available training programs:
            {formations}
            
            Write a recommendation report that:
            1. Summarizes the identified development needs
            2. Proposes a personalized training plan based on available programs
            3. Explains how these programs address the employee's specific needs
            4. Suggests an appropriate training schedule
            
            The report should be professional, concise, and directly applicable.
            """
        )
        
        print(f"Generator initialized with model {model_name}")
        
    def format_formations(self, formations):
        """Format the training programs for the prompt."""
        formatted = ""
        for i, formation in enumerate(formations, 1):
            formatted += f"\nTraining {i}:\n"
            formatted += f"- ID: {formation.get('id', 'Not specified')}\n"
            formatted += f"- Type: {formation.get('type', 'Not specified')}\n"
            formatted += f"- Source: {formation.get('source', 'Not specified')}\n"
            formatted += f"- Content: {formation.get('content', 'Not specified')}\n"
            formatted += f"- Relevance score: {formation.get('score', 0):.2f}\n"
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
            "formations": formations_text
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
        
    def process_employees(self, input_file, output_file):
        """Generate recommendations for all employees."""
        # Load the enriched data
        with open(input_file, 'r', encoding='utf-8') as f:
            employees = json.load(f)
            
        # Generate recommendations for each employee
        results = []
        for employee in employees:
            recommendation = self.generate_recommendation(employee)
            results.append(recommendation)
            
        # Save the results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Recommendations saved in {output_file}")
        return results


# Usage example
if __name__ == "__main__":
    generator = Generator()
    generator.process_employees(
        input_file="data/employe_enriched.json",
        output_file="data/recommendations.json"
    )