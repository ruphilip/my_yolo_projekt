"""
Streamlit-Steuerung für lokale MLOps-Dienste (MLflow, Label Studio, ...).

Start:
    streamlit run app.py      (benötigt Streamlit >= 1.37)

- Beim Öffnen der App wird NICHTS gestartet. Dienste laufen erst nach Klick auf "Start".
- "Start" blockiert nicht: der Dienst bekommt Zeit (startup_timeout), die App zeigt
  "startet…" und aktualisiert sich automatisch, bis der Port erreichbar ist.

Neue Dienste:   -> Abschnitt "SERVICES" ergänzen.
Neue Aktionen:  -> Abschnitt "ACTIONS" ergänzen.
"""

import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import streamlit as st

from check_data import qc_data
from config import *
from label_studio_export import ls_export
from utils import *

from eval_view import render_evaluation      # zu den Imports

                
from models_view import render_models      # zu den Imports



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class Service:
    name: str
    port: int
    url: str
    start: Callable[[], subprocess.Popen]
    stop: Callable[[], str]
    log_file: Path
    startup_timeout: int = 60  # Sekunden, die der Dienst zum Hochfahren bekommt


@dataclass
class Action:
    label: str
    func: Callable[[], str]
    description: str = ""


# ---- SERVICES: hier neue Dienste ergänzen ---------------------------------
SERVICES = [
    Service(
        name="MLflow",
        port=MLFLOW_PORT,
        url=f"http://127.0.0.1:{MLFLOW_PORT}",
        start=start_mlflow_server,
        stop=kill_mlflow_server,
        log_file=BASE_DIR / "mlflow_server.log",
        startup_timeout=90,
    ),
    Service(
        name="Label Studio",
        port=LS_PORT,
        url=f"http://localhost:{LS_PORT}",
        start=start_label_studio_server,
        stop=kill_label_studio_server,
        log_file=BASE_DIR / "label_studio.log",
        startup_timeout=180,  # Label Studio braucht oft lange (Migrationen etc.)
    ),
]


# ---- ACTIONS: hier weitere Einzelfunktionen ergänzen ----------------------
def example_action() -> str:
    return "Beispiel-Aktion ausgeführt."


def find_gx_report() -> Path:
    """Sucht die neueste Great-Expectations-Data-Docs-Startseite (index.html)."""
    if GX_REPORT is not None:
        if GX_REPORT.exists():
            return GX_REPORT
        raise FileNotFoundError(f"GX_REPORT existiert nicht: {GX_REPORT}")

    candidates = []
    for d in GX_SEARCH_DIRS:
        if d.exists():
            candidates += [p for p in d.rglob("index.html") if "data_docs" in p.parts]

    if not candidates:
        suchorte = ", ".join(str(d) for d in GX_SEARCH_DIRS)
        raise FileNotFoundError(
            f"Kein Data-Docs-Report gefunden (gesucht in: {suchorte}). "
            "Pfad in GX_SEARCH_DIRS bzw. GX_REPORT anpassen oder Data Docs bauen."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def open_gx_report() -> str:
    report = find_gx_report()
    webbrowser.open(report.as_uri())
    return f"Report im Browser geöffnet: {report}"


def cleanup_all_ports() -> str:
    parts = []
    for svc in SERVICES:
        st.session_state.starting.pop(svc.name, None)
        if is_port_open(svc.port):
            parts.append(f"{svc.name}: {svc.stop()}")
    return " | ".join(parts) or "Auf keinem Port lief etwas."


ACTIONS = [
    Action("Daten von Label Studio laden", ls_export, 
           "Die Daten sollten aus Label Studio exportiert werden."),
    Action("GX Report erstellen", qc_data,
               "Öffnet die Data Docs (index.html) im Browser."),
    Action("Great Expectations Report", open_gx_report,
           "Öffnet die Data Docs (index.html) im Browser."),
    Action("Versioniere die Daten", run_version, 
           "Die Daten wurden Versioniert und auch in Git getrackt"),
    Action("Alle Ports aufräumen", cleanup_all_ports,
           "Beendet alles, was auf den Ports der Dienste läuft."),

    Action("Beispiel-Aktion", example_action, ""),
]


# ---------------------------------------------------------------------------
# Session-State
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("starting", {})  # name -> (startzeit, Popen)


def notify(kind: str, text: str):
    st.session_state.messages.append((kind, text))


def begin_start(svc: Service):
    try:
        proc = svc.start()
        st.session_state.starting[svc.name] = (time.time(), proc)
    except Exception as e:  # noqa: BLE001
        notify("err", f"{svc.name} konnte nicht gestartet werden: {e}")


def do_stop(svc: Service):
    st.session_state.starting.pop(svc.name, None)
    try:
        notify("ok", f"{svc.name} gestoppt: {svc.stop()}")
    except Exception as e:  # noqa: BLE001
        notify("err", f"{svc.name} stoppen fehlgeschlagen: {e}")


def service_state(svc: Service) -> tuple[str, float]:
    """Gibt (status, vergangene Sekunden) zurück: running | starting | stopped."""
    running = is_port_open(svc.port)
    entry = st.session_state.starting.get(svc.name)

    if running:
        if entry:
            st.session_state.starting.pop(svc.name, None)
            notify("ok", f"{svc.name} ist bereit: {svc.url}")
        return "running", 0.0

    if entry:
        started, proc = entry
        elapsed = time.time() - started
        if proc.poll() is not None and elapsed > 3:
            st.session_state.starting.pop(svc.name, None)
            notify("err", f"{svc.name} wurde unerwartet beendet – siehe Log.")
            return "stopped", 0.0
        if elapsed > svc.startup_timeout:
            st.session_state.starting.pop(svc.name, None)
            notify(
                "err",
                f"{svc.name} ist nach {svc.startup_timeout}s nicht erreichbar – siehe Log.",
            )
            return "stopped", 0.0
        return "starting", elapsed

    return "stopped", 0.0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MLOps Control Panel", page_icon="🛠️", layout="wide")
st.title("🛠️ MLOps Control Panel")


@st.cache_resource(show_spinner=False)
def cleanup_ports_on_startup() -> tuple[str, list[str]]:
    """Wird genau einmal pro Streamlit-Serverstart ausgeführt (nicht bei Reruns
    oder Browser-Reload): beendet alles, was schon auf den Dienst-Ports läuft."""
    zeit = time.strftime("%H:%M:%S")
    results = []
    for svc in SERVICES:
        if is_port_open(svc.port):
            try:
                results.append(f"{svc.name}: {svc.stop()}")
            except Exception as e:  # noqa: BLE001
                results.append(f"{svc.name}: FEHLER – {e}")
    return zeit, results


zeit, cleanup_results = cleanup_ports_on_startup()
if cleanup_results:
    st.caption(f"Port-Cleanup beim Serverstart ({zeit}): " + " | ".join(cleanup_results))
else:
    st.caption(f"Port-Cleanup beim Serverstart ({zeit}): es lief nichts auf den Ports.")


def panel():
    # Meldungen
    for kind, msg in st.session_state.messages[-5:]:
        (st.success if kind == "ok" else st.error)(msg)

    top_l, top_r = st.columns([6, 1])
    if st.session_state.messages and top_l.button("Meldungen löschen"):
        st.session_state.messages = []
        st.rerun(scope="fragment")
    if top_r.button("🔄 Aktualisieren", use_container_width=True):
        st.rerun(scope="fragment")

    # --- Dienste -----------------------------------------------------------
    st.header("Dienste")
    cols = st.columns(len(SERVICES))

    for col, svc in zip(cols, SERVICES):
        with col, st.container(border=True):
            status, elapsed = service_state(svc)
            icon = {"running": "🟢", "starting": "🟡", "stopped": "🔴"}[status]
            label = {
                "running": "läuft",
                "starting": f"startet… ({int(elapsed)}s / max. {svc.startup_timeout}s)",
                "stopped": "gestoppt",
            }[status]

            st.subheader(f"{icon} {svc.name}")
            st.caption(f"Port {svc.port} · {label}")
            if status == "starting":
                st.progress(min(elapsed / svc.startup_timeout, 1.0))

            b1, b2, b3 = st.columns(3)
            if b1.button("Start", key=f"start_{svc.name}",
                         disabled=status != "stopped", use_container_width=True):
                begin_start(svc)
                st.rerun(scope="fragment")
            if b2.button("Stop", key=f"stop_{svc.name}",
                         disabled=status == "stopped", use_container_width=True):
                do_stop(svc)
                st.rerun(scope="fragment")
            if b3.button("Restart", key=f"restart_{svc.name}",
                         disabled=status == "starting", use_container_width=True):
                do_stop(svc)
                time.sleep(1)  # kurz warten, bis der Port frei ist
                begin_start(svc)
                st.rerun(scope="fragment")

            if status == "running":
                st.link_button(f"{svc.name} öffnen", svc.url, use_container_width=True)

            with st.expander("Log"):
                st.code(read_log(svc.log_file), language="text")

    # --- Weitere Aktionen --------------------------------------------------
    st.header("Weitere Funktionen")
    for action in ACTIONS:
        c1, c2 = st.columns([1, 3])
        if c1.button(action.label, key=f"action_{action.label}", use_container_width=True):
            try:
                with st.spinner(f"{action.label} …"):
                    notify("ok", f"{action.label}: {action.func()}")
            except Exception as e:  # noqa: BLE001
                notify("err", f"{action.label} fehlgeschlagen: {e}")
            st.rerun(scope="fragment")
        c2.caption(action.description)


# Automatische Aktualisierung nur, solange ein Dienst gerade startet
auto_refresh = POLL_SECONDS if st.session_state.starting else None
st.fragment(run_every=auto_refresh)(panel)()

render_evaluation(BASE_DIR)                 # ganz ans Ende der Datei, nach st.fragment(...)(panel)()

render_models(BASE_DIR)                    # z. B. vor render_evaluation(BASE_DIR)

