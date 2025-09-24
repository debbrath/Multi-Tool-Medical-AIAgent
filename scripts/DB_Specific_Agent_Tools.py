"""
DB-Specific SQL Agent Tools
===========================
We create separate agents for:
1. Heart Disease Patients
2. Cancer Patients
3. Diabetes Patients
"""

import os
from pyprojroot import here
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# -------------------------------
# Database & LLM Setup
# -------------------------------
db_path = os.path.join(here(), "databases", "PatientsDB.db")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

groq_api_key = os.getenv("GROQ_API_KEY", "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")
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.5,
)


# -------------------------------
# Utility: Create DB Agent
# -------------------------------
def build_db_agent(table_name: str, verbose: bool = False):
    """
    Creates a SQL agent that only works on a specific table.
    """
    # Restrict to just the selected table
    db_subset = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=[table_name])
    return create_sql_agent(llm, db=db_subset, agent_type="openai-tools", verbose=verbose)


from sqlalchemy import create_engine, inspect

engine = create_engine(f"sqlite:///{db_path}")
inspector = inspect(engine)
print("Tables List in DB:", inspector.get_table_names())



# -------------------------------
# Build Tools for Each Table
# -------------------------------
HeartDiseaseDBTool = build_db_agent("heart", verbose=True)
CancerDBTool = build_db_agent("cancer", verbose=True)
DiabetesDBTool = build_db_agent("diabetes", verbose=True)

# -------------------------------
# Example Usage
# -------------------------------
print("\n🔹 Query: Heart Disease Patients")
resp = HeartDiseaseDBTool.invoke({"input": "What are the top 5 ages with the highest heart disease occurrence?"})
print("Agent Response:", resp)

print("\n🔹 Query: Cancer Patients")
resp = CancerDBTool.invoke({"input": "How many cancer patients are there in total?"})
print("Agent Response:", resp)

print("\n🔹 Query: Diabetes Patients")
resp = DiabetesDBTool.invoke({"input": "Which patient type suffers diabetes the most?"})
print("Agent Response:", resp)
