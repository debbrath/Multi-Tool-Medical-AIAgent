# scripts/convert_to_sqlite.py
import os
import glob
import pandas as pd
from sqlalchemy import create_engine, Integer, Float, Text, DateTime, Boolean
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "dbs")
os.makedirs(DB_DIR, exist_ok=True)

def pandas_to_sql_dtype(series):
    if pd.api.types.is_integer_dtype(series):
        return Integer()
    if pd.api.types.is_float_dtype(series):
        return Float()
    if pd.api.types.is_bool_dtype(series):
        return Boolean()
    if pd.api.types.is_datetime64_any_dtype(series):
        return DateTime()
    return Text()

def csv_to_sqlite(csv_path, db_path, table_name):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    engine = create_engine(f"sqlite:///{db_path}")
    dtype = {col: pandas_to_sql_dtype(df[col]) for col in df.columns}
    df.to_sql(table_name, con=engine, if_exists="replace", index=False, dtype=dtype)
    print(f"Wrote {len(df)} rows -> {db_path} (table: {table_name})")

def main():
    # Map common filenames to DB names (adjust as required)
    file_map = {
        "heart-disease.csv": ("heart_disease.db", "heart"),
        "heart.csv": ("heart_disease.db", "heart"),
        "cancer.csv": ("cancer.db", "cancer"),
        "diabetes.csv": ("diabetes.db", "diabetes"),
        "diabetes_data.csv": ("diabetes.db", "diabetes"),
    }
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in ./data — download datasets first.")
        return

    for csv in csv_files:
        fname = os.path.basename(csv)
        db_name, table = file_map.get(fname, (f"{os.path.splitext(fname)[0]}.db", os.path.splitext(fname)[0]))
        db_path = os.path.join(DB_DIR, db_name)
        csv_to_sqlite(csv, db_path, table)

if __name__ == "__main__":
    main()