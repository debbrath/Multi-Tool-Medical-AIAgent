import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI  # if you use LLM in here, optional
from langchain_community.utilities import SerpAPIWrapper

def medical_web_search(query: str) -> str:
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        return "❌ SERPAPI_API_KEY missing. Please add it to your .env file."

    # Create the wrapper with the key
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    result = search.run(query)
    return result