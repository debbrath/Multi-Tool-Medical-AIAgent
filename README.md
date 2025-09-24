Project layout 
multi-tool-med-agent/
├─ data/                      # downloaded CSVs (kaggle)
├─ dbs/              # resulting sqlite db files: heart_disease.db, cancer.db, diabetes.db
├─ scripts/
│  ├─ 
│  └─ csv_to_sqlite.py        # converts CSV -> sqlite with type mapping
├─ agents/
│  ├─ db_tools.py             # HeartDiseaseDBTool, CancerDBTool, DiabetesDBTool (LangChain SQL chains)
│  ├─ web_search_tool.py      # MedicalWebSearchTool (SerpAPI wrapper)
│  └─ main_agent.py           # router + example loop / API
├─ .env                       # API keys (OPENAI_API_KEY, SERPAPI_API_KEY)
├─ requirements.txt
└─ README.md


python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/csv_to_sqlite.py
cd agents
python main_agent.py






 
