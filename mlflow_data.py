import mlflow

# Innerhalb von "with mlflow.start_run():"
# 1. Erstelle ein Metadaten-Objekt für deinen YOLO-Ordner
dataset_source = mlflow.data.local_dataset_source.LocalDatasetSource(path="./dataset")

# 2. Definiere den Datensatz (Name, Quelle und optionale Version/Targets)
mlflow_dataset = mlflow.data.dataset.Dataset(
    source=dataset_source,
    name="Swiss_Coins_YOLO",
    digest=git_tag # Hier kannst du dein Git-Tag oder den DVC-Hash als Version nutzen
)

# 3. Den Datensatz im aktuellen Run registrieren
mlflow.log_input(mlflow_dataset, context="training")

