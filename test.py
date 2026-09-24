import os
import subprocess
import mlflow
import sqlite3
import pandas as pd
from ultralytics import YOLO
import warnings

db_path = "./mlflow_runs/mlflow.db"
connection = sqlite3.connect(db_path)

# Pandas führt das SQL aus und baut direkt eine Tabelle (DataFrame) daraus
df = pd.read_sql_query("SELECT run_uuid, name, status, end_time FROM runs WHERE end_time = (select max(end_time) from runs) LIMIT 5;", connection)

# Verbindung schliessen
connection.close()

# Schön formatiert ausgeben
newest_tag = str(df.name)[str(df.name).find("v"):]

# 2. Das aktuelle Git-Tag (z.B. v1.0) auslesen
try:
    git_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()
except Exception:
    git_tag = "unknown_version"


  
print(float(newest_tag[1:4]))
if newest_tag[1:]>=git_tag[1:]:
    print(newest_tag[1:]>=git_tag[1:])
    git_tag = 'v' + str(float(git_tag[1:])+0.1)

    
print(git_tag)