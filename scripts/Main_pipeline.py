
"""
LangChain + SQLite + OpenAI (GitHub Models) Pipeline
===================================================
This script:
1. Connects to a SQLite database
2. Validates tables and runs test queries
3. Loads environment variables securely
4. Connects to an LLM (GitHub-hosted OpenAI model)
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
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

warnings.filterwarnings("ignore")

# -------------------------------
# 1. SQLite Database Connection
# -------------------------------
# db_path = "F://assignment17//src//databases//PatientsDB.db"
# db_path = str(here("databases")) + "/PatientsDB.db"
db_path = os.path.join(here(), "databases", "PatientsDB.db")
print(f"\n✅ Using database: {db_path}")

# LangChain SQLDatabase
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Quick validation
print("\n🔹 Database Dialect:", db.dialect)
print("🔹 Usable Tables:", db.get_usable_table_names())

# Example query (update table name if needed)
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

    # Columns
    columns = inspector.get_columns(table_name)
    for col in columns:
        print(f"   Column: {col['name']} | Type: {col['type']}")

    # Primary Key
    pk = inspector.get_pk_constraint(table_name)
    print("   Primary Key:", pk)

    # Foreign Keys
    fks = inspector.get_foreign_keys(table_name)
    print("   Foreign Keys:", fks)

connection.close()
print("\n✅ Database inspection complete.")

# -------------------------------
# 3. Environment Variables Setup
# -------------------------------
#token = "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"
token = "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"
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini" 

if not token:
    raise ValueError("❌ GITHUB_TOKEN environment variable not set. Please provide a valid token.")

print("\n✅ GitHub Token loaded successfully.")

# -------------------------------
# 4. LLM Setup (GitHub Models via LangChain)
# -------------------------------
#llm = ChatOpenAI(
#    model_name=model_name,
##    openai_api_key=token,
#    openai_api_base=endpoint,
#    temperature=0.5,
#)

print("Token:", token)
print("Endpoint:", endpoint)
print("Model name:", model_name)

try:
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=token,
        base_url=endpoint,
        temperature=0.5,
    )
    print("LLM constructed:", llm)
    response = llm.invoke([HumanMessage(content="Hello")])
    print("Response:", response)
except Exception as e:
    import traceback
    traceback.print_exc()  


print("\n✅ Connected to LLM:", model_name)

# -------------------------------
# 5. Test LLM Query
# -------------------------------
response = llm.invoke([
    HumanMessage(content="What's the capital of Bangladesh?")
])

print("\n🤖 LLM Response:", response.content)
print("\n✅ Pipeline test complete.")
