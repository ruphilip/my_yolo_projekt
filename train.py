import math
import os
import sqlite3
import subprocess
import warnings

import mlflow
import mlflow.pytorch
import pandas as pd
import yaml
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")

db_path = "./mlflow_runs/mlflow.db"
connection = sqlite3.connect(db_path)

# Pandas führt das SQL aus und baut direkt eine Tabelle (DataFrame) daraus
df = pd.read_sql_query("SELECT run_uuid, name, status, end_time FROM runs WHERE end_time = (select max(end_time) from runs) LIMIT 5;", connection)

# Verbindung schliessen
connection.close()

# Schön formatiert ausgeben
newest_tag = str(df.name)[str(df.name).find("v"):]

# Lokalen MLflow-Speicherort für Artefakte (Modelle, Plots) erstellen
os.makedirs("./mlflow_runs", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow_runs/mlflow.db")

# Experiment-Namen festlegen
mlflow.set_experiment("YOLO_Mac_Training")

# --- TRAININGS-PARAMETER DEFINIEREN ---
EPOCHS = 80
BATCH_SIZE = 16
IMG_SIZE = 640
YOLO_MODEL = "yolov8n.pt"
DROPOUT = 0.5
WEIGHT_D = 0.0005
OPTIMIZER = 'auto'
PATH_TO_DATA_YAML = "dataset/data.yaml"  # Pfad zu deiner data.yaml-Datei


# Das aktuelle Git-Tag auslesen
try:
    git_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()
except Exception:
    git_tag = "unknown_version"
    
if float(newest_tag[1:4])>=float(git_tag[1:]):
    print(float(newest_tag[1:4])>=float(git_tag[1:]))
    git_tag = 'v' + str(math.ceil((float(newest_tag[1:4])+float(0.1))*10)/10)
    
# MD5-Hash aus der DVC-Datei extrahieren
dvc_md5 = "unknown_dvc_hash"
try:
    # Ersetze 'data.yaml.dvc' mit dem tatsächlichen Pfad zu deiner .dvc-Datei
    with open("dataset.dvc", "r") as f:
        dvc_data = yaml.safe_load(f)
        # DVC speichert Hashes typischerweise unter 'outs' -> Liste von Files
        if "outs" in dvc_data and len(dvc_data["outs"]) > 0:
            dvc_md5 = dvc_data["outs"][0].get("md5", "no_md5_found")
except Exception as e:
    print(f"Fehler beim Lesen der DVC-Datei: {e}")
    
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

# MLflow Run starten
with mlflow.start_run(run_name=f"YOLO_Train_{git_tag}") as run:
    
    # Versionierung als Tag in MLflow speichern
    mlflow.set_tag("dataset_version", git_tag)
    mlflow.set_tag("dvc_data_md5", dvc_md5)
    mlflow.set_tag("dataset.git_commit", git_commit)
    mlflow.log_artifact("dataset/data.yaml")   # Klassen + Pfade
    mlflow.log_artifact("dataset.dvc")         # Pointer auf die exakten Daten
    
    # Logge die Hyperparameter explizit in MLflow
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("image_size", IMG_SIZE)
    mlflow.log_param("yolo_model", YOLO_MODEL)
    mlflow.log_param("device", "mps")
    mlflow.log_param("dropout", DROPOUT)
    mlflow.log_param("weight_decay", WEIGHT_D)
    mlflow.log_param("optimizer", OPTIMIZER)
    
    # 4. YOLO Modell laden
    model = YOLO(YOLO_MODEL) 
    
    # 5. Training starten mit Mac-Optimierungen (Apple Silicon)
    results = model.train(
        data=PATH_TO_DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device="mps",       # Aktiviert Apple Silicon GPU Beschleunigung
        workers=4,          # Optimal für M-Champs Core-Architektur
        batch=BATCH_SIZE,           # Ausgewogen für den Mac Unified Memory
        cache=True,          # Nutzt den schnellen RAM des Mac aus
        deterministic=False,
        dropout=DROPOUT, 
        weight_decay=WEIGHT_D,
        optimizer = OPTIMIZER
    )
    
    # Pfad zu den besten Gewichten abgreifen
    best_model_path = results.save_dir / "weights" / "best.pt"
    
    # Das trainierte YOLO-Modell laden
    trained_model = YOLO(str(best_model_path))
    
    # Wir loggen das zugrundeliegende PyTorch-Modell von YOLO (.model)
    mlflow.pytorch.log_model(
        pytorch_model=trained_model.model, 
        artifact_path=f"YOLO_Train_{git_tag}".replace(".","_"),
        serialization_format="pickle"
    )
    
    print(f"Erfolgreich geloggt! Das Modell befindet sich in: {best_model_path}")

