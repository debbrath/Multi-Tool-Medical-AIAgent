# Multi-Tool-Medical-AIAgent Project

This project is a **multi-tool AI agent** that can answer medical questions using:
- Three medical datasets (Heart Disease, Cancer, Diabetes)
- Web search for general medical knowledge

<br/>

## Datasets
- Heart Disease: [Kaggle Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Cancer Prediction: [Kaggle Link](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
- Diabetes: [Kaggle Link](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

<br/>

## 📂 Project Structure

```
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



```
<br/>

## 🛠 Installation & Local Development
### 1. Prerequisites
```bash
- Python 3.12.10
- pip (Python package manager)
```
### 2. Clone the repository
```bash
git clone https://github.com/debbrath/Multi-Tool-Medical-AIAgent.git
cd Multi-Tool-Medical-AIAgent
```
### Step 3: Open VSCode
- Launch VSCode.
- Open your project folder 
### Step 4: Select the Interpreter
Open a terminal (`Ctrl+`` in VS Code) and run:
```bash
python --version

If not found.

- Open System Properties:

  Press Win + R, type sysdm.cpl, press Enter.

  Go to Advanced → Environment Variables.

  Under System variables, select Path → Edit → New.

- Paste:

  C:\Users\<YourName>\AppData\Local\Programs\Python\Python312\

  C:\Users\<YourName>\AppData\Local\Programs\Python\Python312\Scripts\

  Save, then restart VS Code.

- Press Ctrl+Shift+P → type Python: Select Interpreter → Enter.
  python -m venv venv

```

### 5. Create and activate a virtual environment
```bash
# On Windows PowerShell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate

On Linux/Mac
python -m venv env
source env/bin/activate

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
cd F:\Python\Multi-Tool-Medical-AIAgent
.\.venv\Scripts\Activate.ps1

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/csv_to_sqlite.py
cd agents
python main_agent.py

python -m pip install --upgrade pip setuptools wheel --use-pep517
```
### 6. Install dependencies
```bash
pip install -r requirements.txt
```
### 6. Install SQLite3
- Step 1: Download SQLite3
    Go to the official SQLite download page: https://www.sqlite.org/download.html
    Under Precompiled Binaries for Windows:
    Download sqlite-tools-win32-x86-xxxx.zip (contains sqlite3.exe).
    Extract the zip file to a folder, e.g., C:\sqlite.
- Step 2: Add SQLite to PATH
    Open Start Menu → Environment Variables → Edit the system environment variables.
    Click Environment Variables → System variables → Path → Edit → New.
    Add the path to the folder where sqlite3.exe is located (e.g., C:\sqlite).
    Click OK on all windows.

```
```
![Screenshot](https://github.com/debbrath/Multi-Tool-Medical-AIAgent/blob/main/image/1.png)

```
```  
- Step 3: Test SQLite Installation
    Open Command Prompt.
    Type:
    sqlite3 --version
```
```
![Screenshot](https://github.com/debbrath/Multi-Tool-Medical-AIAgent/blob/main/image/2.png)

```
```
    
```
```
### 7. Train the model (if not already trained)
```bash
(venv) PS F:\Python\Sentiment-Analysis> python -m app.model_train

```
### 8. Run locally
```bash
(venv) PS F:\Python\Sentiment-Analysis> uvicorn app.main:app –reload
```
![Screenshot](https://github.com/debbrath/Sentiment-Analysis/blob/main/images/image5.png)
```
```
<br/>

## 🛠 Technologies Used

Python 3.12+

SQLite

PLangChain

SerpAPI

SQLDatabaseChain (LangChain)

<br/>

---

✍️ Author

Debbrath Debnath

📫 [Connect on LinkedIn](https://www.linkedin.com/in/debbrathdebnath/)

🌐 [GitHub Profile](https://github.com/debbrath) 






 
