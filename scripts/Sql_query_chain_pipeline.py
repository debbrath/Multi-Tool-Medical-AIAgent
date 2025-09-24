
"""
LangChain SQL Query Chain Pipeline
==================================
This script:
1. Creates a SQL query chain using an LLM
2. Extracts SQL queries from LLM responses
3. Executes SQL queries against SQLite DB
4. Uses a final answer prompt to return natural language answers
"""

# -------------------------------
# Imports
# -------------------------------
import os
import re
import warnings
from pyprojroot import here
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Setup Database Connection
# -------------------------------
# db_path = "F://assignment17//src//databases//PatientsDB.db"
# db_path = str(here("databases")) + "/PatientsDB.db"
db_path = os.path.join(here(), "databases", "PatientsDB.db")
print(f"\n✅ Using database: {db_path}")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
print("🔹 Usable Tables:", db.get_usable_table_names())

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

print("\n✅ LLM connected:", model_name)

# -------------------------------
# 3. SQL Query Chain Setup
# -------------------------------
chain = create_sql_query_chain(llm, db)

response = chain.invoke({"question": "How many patients are there?"})
print("\n🤖 LLM Raw Response:\n", response)

# -------------------------------
# 4. SQL Query Extractor
# -------------------------------
def extract_sql_query_simple(response: str):
    """Extracts full SQL query (multi-line) from LLM response."""
    match = re.search(r"SQLQuery:\s*([\s\S]+?)(?:Answer:|$)", response, re.IGNORECASE)
    if match:
        sql_query = match.group(1).strip()
        # Clean up trailing formatting if needed
        return sql_query
    return None


query = extract_sql_query_simple(response)
print("\n📌 Extracted SQL Query:\n", query)

# Test direct execution
if query:
    try:
        print("\n📊 Query Result (db.run):")
        print(db.run(query))

        print("\n📊 Query Result (db._execute):")
        print(db._execute(query))
    except Exception as e:
        print("⚠️ SQL Execution Error:", e)

# -------------------------------
# 5. Tool-based Query Execution
# -------------------------------
write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

# Chain = write → extract → execute
chain_pipeline = write_query | extract_sql_query_simple | execute_query

print("\n🔹 Pipeline: Question → SQL → Execution\n")

print(chain_pipeline.invoke({"question": "How many patients are there"}))
print(chain_pipeline.invoke({"question": "Give me the data of all patient in the diabetis (first 15 data)"}))
print(chain_pipeline.invoke({"question": "Give me all patient titles (dont use any limit)"}))

# -------------------------------
# 6. Answer Prompt Chain
# -------------------------------
# Build query-only chain
query_chain = write_query | extract_sql_query_simple

# Answer formatting prompt
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, 
    answer the user question in plain English.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:"""
)

answer_chain = answer_prompt | llm | StrOutputParser()

# Example 1
query = query_chain.invoke({"question": "How many patients are there"})
sql_result = db._execute(query)

final_answer = answer_chain.invoke({
    "question": "How many patients are there",
    "query": query,
    "result": sql_result
})
print("\n✅ Final Answer:", final_answer)

# Example 2
query = query_chain.invoke({"question": "Give me the data of all patients in the diabetes (first 15 data)"})
sql_result = db._execute(query)

final_answer = answer_chain.invoke({
    "question": "Give me the data of all patients in the diabetes (first 15 data)",
    "query": query,
    "result": sql_result
})
print("\n✅ Final Answer:", final_answer)

