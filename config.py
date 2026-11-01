from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

MLFLOW_PORT = 5050  # 5000 ist auf macOS vom AirPlay-Empfänger belegt
MLFLOW_BACKEND = "sqlite:///mlflow_runs/mlflow.db"

LS_PORT = 8080
LS_DIR = Path("/Users/philipprutimann/Documents/ZHAW/CAS_ADML/MLOps/LabelStudio")
LS_VENV = LS_DIR / ".venv"

# Great Expectations: Ordner, in dem nach den Data Docs (index.html) gesucht wird.
# Bei Bedarf anpassen. Wird der Report nicht gefunden, GX_REPORT direkt auf die
# index.html setzen, z. B. Path(".../gx/uncommitted/data_docs/local_site/index.html")
GX_SEARCH_DIRS = [
    BASE_DIR / "ge_expectations",
    LS_DIR.parent / "ge_expectations",
]
GX_REPORT = None  # oder Path("...")

POLL_SECONDS = 2  # Intervall der automatischen Statusprüfung während des Startens
