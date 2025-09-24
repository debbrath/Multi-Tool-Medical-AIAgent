import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase


class BaseDBTool:
    def __init__(self, db_path, llm=None):
        uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.db = SQLDatabase.from_uri(uri)
        self.llm = llm or ChatOpenAI(temperature=0)
        self.chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=False)

    def query(self, question: str) -> str:
        """Run a natural language query against the DB."""
        return self.chain.run(question)


class HeartDiseaseDBTool(BaseDBTool):
    def __init__(self, llm=None):
        super().__init__("../data/heart_disease.db", llm)


class CancerDBTool(BaseDBTool):
    def __init__(self, llm=None):
        super().__init__("../data/cancer.db", llm)


class DiabetesDBTool(BaseDBTool):
    def __init__(self, llm=None):
        super().__init__("../data/diabetes.db", llm)