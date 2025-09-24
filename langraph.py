# main.py
import getpass
import os
import pandas as pd
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
# from langgraph.prebuilt import create_openai_functions_agent

import pandas as pd

df = pd.read_csv("data/diabetes.csv")
print(df.columns)


# ------------------------------
# 1. Define Tools
# ------------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers. Takes 2 integers and returns their sum."""
    return a + b


@tool
def get_maximum_age(file_path: str) -> int:
    """Returns the maximum age from a DataFrame loaded from a CSV file.
    Args:
        file_path: The path to the CSV file.
    Returns:
        The maximum age in the DataFrame.
    """
    df = pd.read_csv(file_path)
    return df["Age"].max()


# ------------------------------
# 2. Setup LangChain LLM
# ------------------------------

# Load API key from environment (recommended for security)
# Run in terminal before executing:
#   setx GITHUB_TOKEN "your_token_here"   (Windows)
#   export GITHUB_TOKEN="your_token_here" (Linux/Mac)

# -------------------------------
# 2. Setup LLM (GitHub Models)
# -------------------------------
os.environ["TAVILY_API_KEY"] = "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"
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

# ------------------------------
# 3. Create the Agent
# ------------------------------

memory = MemorySaver()
tools = [multiply, add, get_maximum_age]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# ------------------------------
# 4. Run Example Queries
# ------------------------------

config = {"configurable": {"thread_id": "abc123"}}

# 1) Multiply
input_message = {"role": "user", "content": "Multiply 3 and 4"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# 2) Add
input_message = {"role": "user", "content": "Add 3 and 4"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# 3) Memory test: last message
input_message = {"role": "user", "content": "What was my last message?"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# 4) Memory test: first message
input_message = {"role": "user", "content": "What was my first message?"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# 5) Get maximum age from diabetes dataset
csv_path = os.path.join("data", "diabetes.csv")  # Put file in ./data/diabetes.csv
input_message = {"role": "user", "content": f"Get the maximum age from this file path {csv_path}"}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()



# 3. Setup Tavily tool
tavily_tool = TavilySearch(max_results=5, topic="general")

# 4. Create agent that can use Tavily
agent_executor = create_react_agent(
    llm,
    tools=[tavily_tool]
)

# 5. Run agent
config = {"configurable": {"thread_id": "abc123"}}
input_message = {
    "role": "user",
    "content": "give me 4 add 5, multiply by 2, and then search Tavily for 'Python LangChain tutorial'",
}

for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()