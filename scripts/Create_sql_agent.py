
"""
LangChain SQL Agent Pipeline
============================
This script:
1. Creates a SQL Agent with LangChain
2. Handles multi-line SQL queries safely
3. Runs natural language questions directly
"""
import os
import re
from pyprojroot import here
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

# -------------------------------
# Database & LLM Setup
# -------------------------------
# db_path = "F://assignment17//src//databases//PatientsDB.db"
# db_path = str(here("databases")) + "/PatientsDB.db"
db_path = os.path.join(here(), "databases", "PatientsDB.db")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# -------------------------------
# 2. Setup LLM (GitHub Models)
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY", "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")
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"

if not groq_api_key:
    raise ValueError("❌ GITHUB_TOKEN environment variable not set.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.5,
)


# -------------------------------
# SQL Agent Creation
# -------------------------------
agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
)

# -------------------------------
# Utility: SQL Extractor
# -------------------------------
def extract_sql_query(response: str):
    """Extract a clean SQL query from agent/LLM responses."""
    match = re.search(r"SQLQuery:\s*([\s\S]+?)(?:Answer:|Observation:|Thought:|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# -------------------------------
# Example Queries
# -------------------------------
print("\n🔹 Diabetes Patients")
response = agent_executor.invoke(
    {"input": "List the total diabetes_patients patients. Which types of patients suffer diabetes the most?"}
)
print("Agent Response:", response)

print("\n🔹 Cancer Patients Table Description")
response = agent_executor.invoke({"input": "Describe the cancer_patients table"})
print("Agent Response:", response)

print("\n🔹 Heart disease patients Table Primary Key")
response = agent_executor.invoke({"input": "Tell me about the heart_disease_patients table. What is the primary key?"})
print("Agent Response:", response)

