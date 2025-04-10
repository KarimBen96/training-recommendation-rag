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
    
    def __init__(self, model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo")):
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
    
    def generate_combined_recommendation(self, employee):
        """Combine recommendations from different agents."""
        # Extract partial recommendations
        training_rec = employee.get("recommendation_programme de formation", "")
        practices_rec = employee.get("recommendation_meilleures pratiques", "")
        case_study_rec = employee.get("recommendation_étude de cas", "")
        
        # Create a prompt to merge recommendations
        prompt = ChatPromptTemplate.from_template(
            """
            You need to create a coherent training recommendation by combining 
            the following specialized analyses:
            
            TRAINING PROGRAMS:
            {training_rec}
            
            BEST PRACTICES:
            {practices_rec}
            
            CASE STUDIES:
            {case_study_rec}
            
            EMPLOYEE:
            - Name: {name}
            - Evaluation: {evaluation}
            - Score: {score}/100
            
            Write a complete and coherent recommendation that smoothly integrates
            these three perspectives into a single, actionable training plan.
            The report should be professional, concise, and directly applicable.
            """
        )
        
        # Prepare variables
        variables = {
            "training_rec": training_rec,
            "practices_rec": practices_rec,
            "case_study_rec": case_study_rec,
            "name": employee["employe"],
            "evaluation": employee["evaluation"],
            "score": employee["score"]
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
    
    def process_multi_agent_employees(self, input_file, output_file):
        """Generate combined recommendations from multiple agents for all employees."""
        # Load the enriched data
        with open(input_file, 'r', encoding='utf-8') as f:
            employees = json.load(f)
            
        # Generate combined recommendations for each employee
        results = []
        for employee in employees:
            recommendation = self.generate_combined_recommendation(employee)
            results.append(recommendation)
            
        # Save the results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Combined recommendations saved in {output_file}")
        return results


# Usage example
if __name__ == "__main__":
    generator = Generator()

    # generator.process_employees(
    #     input_file="data/employe_enriched.json",
    #     output_file="data/recommendations.json"
    # )

    test_employee = {
        "employe": "Marie Dupont",
        "evaluation": "Besoins de renforcer la gestion de projet et la communication.",
        "score": 70,
        "keywords": ["gestion de projet", "communication"],
        "recommendation_programme de formation": "Marie would benefit from our Advanced Project Management course...",
        "recommendation_meilleures pratiques": "Based on best practices, Marie should focus on improving her meeting facilitation skills...",
        "recommendation_étude de cas": "Similar cases show that communication training paired with real project experience yields best results..."
    }

    test_recommendation = generator.generate_combined_recommendation(test_employee)
    print("\nCombined recommendation:")
    print(test_recommendation["recommendation"])