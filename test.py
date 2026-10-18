import os
import sqlite3
import pandas as pd
from ultralytics import YOLO
import mlflow
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")

# 1. MLflow Tracking konfigurieren
mlflow.set_tracking_uri("sqlite:///mlflow_runs/mlflow.db")
mlflow.set_experiment("YOLO_Mac_Training")

# --- DYNAMISCH DEN NEUESTEN ERFOLGREICHEN RUN ERMITTELN ---
def get_latest_run_info():
    db_path = "./mlflow_runs/mlflow.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"MLflow-Datenbank nicht gefunden unter: {db_path}")
        
    connection = sqlite3.connect(db_path)
    # Holt die Run-ID und das Git-Tag des neuesten erfolgreichen Trainings-Runs
    query = """
        SELECT r.run_uuid, r.artifact_uri, t.value as git_tag
        FROM runs r 
        JOIN experiments e ON r.experiment_id = e.experiment_id
        LEFT JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'dataset_version'
        WHERE e.name = 'YOLO_Mac_Training' AND r.status = 'FINISHED'
        ORDER BY r.end_time DESC 
        LIMIT 1;
    """
    df_run = pd.read_sql_query(query, connection)
    connection.close()
    
    if df_run.empty:
        raise ValueError("Keine erfolgreichen Trainings-Runs in der Datenbank gefunden.")
        
    run_id = df_run.iloc[0]['run_uuid']
    artifact_uri = df_run.iloc[0]['artifact_uri'].replace("file://", "")
    git_tag = df_run.iloc[0]['git_tag'] or "unknown"
    
    # Pfad zur trainierten PyTorch-Modell-Datei (.pth)
    model_path = os.path.join(artifact_uri, "yolo_best_model", "model.pth")
    return run_id, model_path, git_tag

try:
    train_run_id, model_path, dataset_version = get_latest_run_info()
    print(f"📦 Verwende das neueste Modell aus Run: {train_run_id} (Version: {dataset_version})")
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    exit()

# 2. YOLO Modell initialisieren
# Ultralytics lädt die weights direkt aus dem geloggten PyTorch-Status
model = YOLO(model_path)

# 3. Test-Run in MLflow starten (als Child-Run oder separater Test-Run)
with mlflow.start_run(run_name=f"YOLO_Test_{dataset_version}") as run:
    
    # Metadaten setzen, um den Test-Run mit dem Trainings-Run zu verknüpfen
    mlflow.set_tag("test_for_train_run", train_run_id)
    mlflow.set_tag("dataset_version", dataset_version)
    mlflow.log_param("task", "test_evaluation")
    mlflow.log_param("device", "mps")

    print("\n🚀 Starte Modell-Prüfung auf den Testdaten...")
    
    # 4. Validierung/Test ausführen
    # WICHTIG: In deiner 'data.yaml' sollte der Pfad 'test:' definiert sein.
    # Falls nicht, nutzt YOLO standardmässig die 'val:'-Daten.
    metrics = model.val(
        data="data.yaml",
        split="test",       # Erzwingt die Nutzung des Test-Splits aus der data.yaml
        device="mps",       # Nutzt die Mac GPU
        imgsz=640,
        plots=True          # Generiert PR-Kurven, Confusion Matrix Plots etc.
    )

    # 5. Wichtigste Metriken extrahieren
    map50 = metrics.box.map50          # mAP @ IoU=0.5
    map95 = metrics.box.map            # mAP @ IoU=0.5:0.95
    precision = metrics.box.mp         # Mean Precision
    recall = metrics.box.mr            # Mean Recall
    
    # 6. Metriken in MLflow loggen
    mlflow.log_metric("test_mAP50", map50)
    mlflow.log_metric("test_mAP50_95", map95)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)

    # 7. Ergebnisse schön formatiert im Terminal ausgeben
    print("\n" + "="*40)
    print("📊 PRÜFBERICHT (TEST DATEN)")
    print("="*40)
    print(f"🎯 Precision (Präzision):  {precision * 100:.2f}%")
    print(f"🔄 Recall (Trefferquote):  {recall * 100:.2f}%")
    print(f"📈 mAP50:                 {map50 * 100:.2f}%")
    print(f"📉 mAP50-95:              {map95 * 100:.2f}%")
    print("="*40)

    # 8. Die von YOLO erzeugten Ergebnis-Bilder (Confusion Matrix, etc.) in MLflow sichern
    results_dir = metrics.save_dir
    if os.path.exists(results_dir):
        print(f"\n📂 Hochladen der Test-Visualisierungen aus: {results_dir}")
        for file in os.listdir(results_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                mlflow.log_artifact(os.path.join(results_dir, file), artifact_path="test_plots")
                
    print("\n✅ Test erfolgreich abgeschlossen und in MLflow registriert!")
