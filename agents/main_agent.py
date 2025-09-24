import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

from db_tools import HeartDiseaseDBTool, CancerDBTool, DiabetesDBTool
from web_search_tool import medical_web_search

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_groq import ChatGroq


from langchain_groq import ChatGroq
import os


def build_agent():
    #llm = ChatOpenAI(temperature=0)
    #llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    # Load key from env
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Updated model
        temperature=0,
        groq_api_key=api_key
    )
    # Initialize DB tools
    heart_tool = HeartDiseaseDBTool(llm)
    cancer_tool = CancerDBTool(llm)
    diabetes_tool = DiabetesDBTool(llm)

    # Define available tools
    tools = [
        Tool(
            name="HeartDiseaseDBTool",
            func=heart_tool.query,
            description="Answer questions about the Heart Disease dataset",
        ),
        Tool(
            name="CancerDBTool",
            func=cancer_tool.query,
            description="Answer questions about the Cancer dataset",
        ),
        Tool(
            name="DiabetesDBTool",
            func=diabetes_tool.query,
            description="Answer questions about the Diabetes dataset",
        ),
        Tool(
            name="MedicalWebSearchTool",
            func=medical_web_search,
            description="Answer general medical knowledge questions (definitions, symptoms, cures)",
        ),
    ]

    # Build the main agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
    )
    return agent


def run_interactive():
    agent = build_agent()
    print("🧠 Multi-tool medical agent (type 'exit' to quit)")
    while True:        
        q = input("User: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        try:
            ans = agent.run(q)
            print("AI:", ans)
        except Exception as e:
            print("⚠️ Error:", e)


if __name__ == "__main__":
    run_interactive()