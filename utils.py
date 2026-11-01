import socket
import subprocess
import time
from pathlib import Path
import sys
import os

from label_studio_sdk.client import LabelStudio

from config import *


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_port(port: int) -> str:
    """Beendet alle Prozesse, die auf dem Port LAUSCHEN (nicht bloss verbunden sind,
    z. B. der Browser), und prüft danach, dass der Port wirklich frei ist."""
    total = 0
    for _ in range(5):  # mehrere Runden, falls ein Elternprozess Worker neu startet
        pids = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True,
        ).stdout.split()
        if not pids:
            break
        for pid in pids:
            subprocess.run(["kill", "-9", pid])
            total += 1
        time.sleep(1)

    if is_port_open(port):
        raise RuntimeError(f"Port {port} ist nach dem Beenden weiterhin belegt.")
    return f"{total} Prozess(e) auf Port {port} beendet."


def read_log(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return "(noch kein Log vorhanden)"
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:]) or "(Log ist leer)"


# ---------------------------------------------------------------------------
# Start-/Stop-Funktionen
# Start-Funktionen starten den Prozess nur und kehren sofort zurück (Popen).
# ---------------------------------------------------------------------------
def start_mlflow_server() -> subprocess.Popen:
    log = open(BASE_DIR / "mlflow_server.log", "w")
    return subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", MLFLOW_BACKEND,
            "--port", str(MLFLOW_PORT),
        ],
        cwd=BASE_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def kill_mlflow_server() -> str:
    return kill_port(MLFLOW_PORT)


def start_label_studio_server() -> subprocess.Popen:
    if not (LS_VENV / "bin" / "activate").exists():
        raise FileNotFoundError(f"venv nicht gefunden: {LS_VENV}")

    log = open(BASE_DIR / "label_studio.log", "w")
    return subprocess.Popen(
        f"source '{LS_VENV}/bin/activate' && label-studio start --port {LS_PORT}",
        shell=True,
        executable="/bin/bash",
        cwd=LS_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def kill_label_studio_server() -> str:
    return kill_port(LS_PORT)


# ---------------------------------------------------------------------------
# Connect to Label Studio
# ---------------------------------------------------------------------------


def connect_label_studio_sdk(url: str, api_token: str, project_id: int) -> LabelStudio:
    """
    Prüft die Verbindung via SDK und gibt den initialisierten Client zurück.
    Bricht bei Fehlern das Skript ab.
    """
    print("🔄 Prüfe Verbindung via Label Studio SDK...")
    try:
        # Client initialisieren (Validiert das Token noch nicht aktiv)
        ls = LabelStudio(base_url=url, api_key=api_token)
        
        # Projekt abrufen – das erzwingt den API-Call und prüft Token + ID
        project = ls.projects.get(id=project_id)
        
        print("✅ SDK-Verbindung erfolgreich!")
        print(f"   Projekt: '{project.title}' (ID: {project.id})")
        print(f"   Gelabelte Tasks: {project.task_number}\n")
        
        return ls
        
    except Exception as e:
        # Das SDK wirft je nach Fehler spezifische HTTP-Fehler aus
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("❌ Fehler: Der API_TOKEN ist ungültig.")
        elif "404" in error_msg:
            print(f"❌ Fehler: Projekt-ID {project_id} nicht gefunden.")
        else:
            print(f"❌ Verbindungsfehler: {error_msg}")
        
        print("💥 Skript abgebrochen.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Versionierung der Daten
# ---------------------------------------------------------------------------       
        
def run_version1():
    """
    Führt das Bash-Skript 'version_data.sh' aus, das die Versionierungsinformationen sammelt.
    """
    SCRIPT = "version_data.sh"   # Name deines Bash-Skripts (im selben Ordner wie diese Datei)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["bash", SCRIPT], cwd=script_dir, check=True)
    
    

def run_version() -> str:
    """
    Führt 'version_data.sh' aus und gibt die Ausgabe (echo) des Skripts zurück.
    """
    SCRIPT = "version_data.sh"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    result = subprocess.run(
        ["bash", SCRIPT],   # Beschreibung als Argument übergeben
        cwd=script_dir,
        check=True,
        capture_output=True,   # stdout + stderr abfangen
        text=True,             # als str statt bytes
    )
    return result.stdout       # ggf. + result.stderr