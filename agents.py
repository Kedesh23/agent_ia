import json
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool, SerperDevTool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from get_api_key import get_api_key
import pandas as pd
import os

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'NA'

# Initialize LLM with the API key and model
llm_groq = ChatGroq(
    temperature=0, 
    groq_api_key=get_api_key()['groq_api_key'], 
    model_name="llama3-8b-8192"
)

class SaveCSVTools(BaseTool):

    name: str = "SaveCSVTool"
    description: str = "Tool to save provided data into a CSV file."
    def _run(self, data:dict, filename: str = "tourist_sites.csv") -> str:
        df = pd.DataFrame(data["tourist_sites"])
        df.to_csv(filename, index=False)
        return "CSV saved as {filename}"



# Define the agent
agent = Agent(
    role="Search Specialist",
    goal="Trouves et liste tous les sites touristiques du Gabon",
    backstory="Tu es un expert en recherche de sites touristiques au Gabon. Ta tâche est d'identifier et de lister les sites touristiques du Gabon.",
    verbose=True,
    llm=llm_groq,
    tools = [SaveCSVTools(result_as_answer=True)]
)


# Define the task
task = Task(
    description=(
        "Rechercher et lister tous les sites touristiques du Gabon. "
        "Inclure l'adresse du site et les coordonnées GPS sous forme de lien cliquable."
    ),
    expected_output="Un objet JSON comme réponse avec pour clé 'tourist_sites'.",
    agent=agent
)

# Initialize the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# Start the task
result = crew.kickoff(inputs={"search_querry": "Tous les sites touristiques du Gabon"})

# Print the result
print(result)
