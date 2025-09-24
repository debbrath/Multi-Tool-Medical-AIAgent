"""
Hybrid Medical Agent
====================

- DB Tools for patient stats (Heart, Cancer, Diabetes).
- Web Search Tool for general medical queries.
- Math utility tools.
- Main Agent routes intelligently between DB & Web Search.
"""

import os
from pyprojroot import here
import pandas as pd

from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

# --------------------------------
# 1. Database Setup
# --------------------------------
db_path = os.path.join(str(here("/assignment17/src/databases")), "PatientsDB.db")

# Restrict DB agent to a specific table
def build_db_agent(table_name: str, verbose: bool = False):
    db_subset = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=[table_name])
    return create_sql_agent(
        llm,
        db=db_subset,
        agent_type="openai-tools",
        verbose=verbose,
    )

# --------------------------------
# 2. LLM Setup (GitHub Models)
# --------------------------------
os.environ["TAVILY_API_KEY"] = "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"
token = "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"
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"

llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=token,
    openai_api_base=endpoint,
    temperature=0.2,
)

# --------------------------------
# 3. DB-Specific Agents
# --------------------------------
HeartDiseaseDBToolAgent = build_db_agent("heart_disease_patients")
CancerDBToolAgent = build_db_agent("cancer_patients")
DiabetesDBToolAgent = build_db_agent("diabetes_patients")

# --------------------------------
# 4. Web Search Tool (Medical)
# --------------------------------
MedicalWebSearchTool = TavilySearch(
    max_results=5,
    topic="general",  # Tavily only supports 'general', 'news', 'finance'
)
MedicalWebSearchTool.name = "MedicalWebSearchTool"
MedicalWebSearchTool.description = "Use this tool for general medical knowledge (definitions, symptoms, cures)."

# --------------------------------
# 5. Utility Tools
# --------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def get_maximum_age(file_path: str) -> int:
    """Get maximum age from a CSV dataset."""
    df = pd.read_csv(file_path)
    return int(df["Age"].max())

# --------------------------------
# 6. Wrap DB Agents as Tools
# --------------------------------
@tool
def heart_disease_query(query: str) -> str:
    """Query the Heart Disease database."""
    return HeartDiseaseDBToolAgent.invoke({"input": query})

@tool
def cancer_query(query: str) -> str:
    """Query the Cancer database."""
    return CancerDBToolAgent.invoke({"input": query})

@tool
def diabetes_query(query: str) -> str:
    """Query the Diabetes database."""
    return DiabetesDBToolAgent.invoke({"input": query})

# --------------------------------
# 7. Create Main Agent
# --------------------------------
memory = MemorySaver()

tools = [
    multiply,
    add,
    get_maximum_age,
    MedicalWebSearchTool,
    heart_disease_query,
    cancer_query,
    diabetes_query,
]

agent_executor = create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,
)

# --------------------------------
# 8. Run Example Queries
# --------------------------------
config = {"configurable": {"thread_id": "med123"}}

# Example 1: DB query
input_message = {"role": "user", "content": "Show me top 5 ages with highest heart disease cases"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# Example 2: Web search query
input_message = {"role": "user", "content": "What are common symptoms of diabetes?"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# Example 3: Mixed math utility
input_message = {"role": "user", "content": "Add 40 and 60, then multiply by 2"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# Example 4: CSV dataset query
csv_path = os.path.join("data", "diabetes.csv")  # ensure this file exists
input_message = {"role": "user", "content": f"Get the maximum age from this file path {csv_path}"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()
