"""
LangChain + SQLite + GROQ (GitHub Models) Pipeline
===================================================
This script:
1. Connects to a SQLite database
2. Validates tables and runs test queries
3. Loads environment variables securely
4. Connects to an LLM (GitHub-hosted Groq model)
5. Runs a sample query to validate setup
"""

# -------------------------------
# Imports
# -------------------------------
import os
import warnings
import pandas as pd
from pyprojroot import here
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

warnings.filterwarnings("ignore")

# -------------------------------
# 1. SQLite Database Connection
# -------------------------------
db_path = os.path.join(here(), "databases", "PatientsDB.db")
print(f"\n✅ Using database: {db_path}")

# LangChain SQLDatabase
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Quick validation
print("\n🔹 Database Dialect:", db.dialect)
print("🔹 Usable Tables:", db.get_usable_table_names())

try:
    result = db._execute("SELECT * FROM cancer_patients LIMIT 10;")
    print("\n📊 Sample Data from cancer_patients table:")
    for row in result:
        print(row)
except Exception as e:
    print("\n⚠️ Could not fetch from cancer_patients table:", e)

# -------------------------------
# 2. SQLAlchemy Inspector (Detailed Inspection)
# -------------------------------
engine = create_engine(f"sqlite:///{db_path}")
connection = engine.connect()
inspector = inspect(engine)

print("\n📋 Tables in database:", inspector.get_table_names())

for table_name in inspector.get_table_names():
    print(f"\n🔎 Inspecting table: {table_name}")

    columns = inspector.get_columns(table_name)
    for col in columns:
        print(f"   Column: {col['name']} | Type: {col['type']}")

    pk = inspector.get_pk_constraint(table_name)
    print("   Primary Key:", pk)

    fks = inspector.get_foreign_keys(table_name)
    print("   Foreign Keys:", fks)

connection.close()
print("\n✅ Database inspection complete.")

# -------------------------------
# 3. Environment Variables Setup
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY", "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")
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not set. Please provide a valid key.")

print("\n✅ GROQ API Key loaded successfully.")

# -------------------------------
# 4. LLM Setup (Groq via GitHub Models)
# -------------------------------
# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.5,
)

print("\n✅ Connected to LLM:", model_name)

# -------------------------------
# 5. Test LLM Query
# -------------------------------
response = llm.invoke([
    HumanMessage(content="What's the capital of Bangladesh?")
])

print("\n🤖 LLM Response:", response.content)
print("\n✅ Pipeline test complete.")
