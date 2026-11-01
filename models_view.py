"""
Streamlit-Abschnitt "Modelle (MLflow)": Übersicht aller Runs/Modelle aus der MLflow-Datenbank.

Einbindung in app.py (2 Zeilen):
    from models_view import render_models          # bei den Imports
    render_models(BASE_DIR)                        # z. B. vor render_evaluation(BASE_DIR)
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def fmt_ts(ms) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if ms else ""


def fmt_dur(start, end) -> str:
    if not start or not end:
        return ""
    s = int((end - start) / 1000)
    return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s" if s >= 3600 else f"{s // 60}m {s % 60}s"


def find_weights(base_dir: Path, run_id: str) -> Path | None:
    hits = sorted(base_dir.glob(f"mlruns/*/{run_id}/artifacts/weights/best.pt"))
    return hits[0] if hits else None


def has_evaluation(base_dir: Path, run_id: str) -> bool:
    return (base_dir / "evaluation_results" / run_id / "results.json").exists()


def get_client(base_dir: Path) -> MlflowClient:
    db = base_dir / "mlflow_runs" / "mlflow.db"
    return MlflowClient(tracking_uri=f"sqlite:///{db}")


@st.cache_data(ttl=10, show_spinner="Lade Runs aus MLflow …")
def load_runs(base_dir_str: str) -> tuple[pd.DataFrame, dict]:
    """Gibt (Tabelle, Details je run_id) zurück."""
    base_dir = Path(base_dir_str)
    client = get_client(base_dir)
    exps = {e.experiment_id: e.name for e in client.search_experiments()}
    runs = client.search_runs(list(exps), max_results=1000, order_by=["attributes.start_time DESC"])

    rows, details = [], {}
    for r in runs:
        i, d = r.info, r.data
        weights = find_weights(base_dir, i.run_id)
        map_key = next((k for k in d.metrics if "map50-95" in k.lower().replace("_", "")), None)
        rows.append({
            "Run-Name": i.run_name,
            "Run-ID": i.run_id,
            "Experiment": exps.get(i.experiment_id, i.experiment_id),
            "Status": i.status,
            "Start": fmt_ts(i.start_time),
            "Dauer": fmt_dur(i.start_time, i.end_time),
            "Dataset-Version": d.tags.get("dataset_version", ""),
            "DVC-MD5": d.tags.get("dvc_data_md5", "")[:10],
            "Git-Commit": d.tags.get("dataset.git_commit", "")[:8],
            "YOLO-Modell": d.params.get("yolo_model", ""),
            "Epochs": d.params.get("epochs", ""),
            "Batch": d.params.get("batch_size", ""),
            "mAP50-95 (Training)": round(d.metrics[map_key], 4) if map_key else None,
            "Gewichte (best.pt)": "✅" if weights else "–",
            "Evaluiert": "✅" if has_evaluation(base_dir, i.run_id) else "–",
        })
        details[i.run_id] = {
            "weights": str(weights) if weights else None,
            "artifact_uri": i.artifact_uri,
            "params": dict(d.params),
            "metrics": dict(d.metrics),
            "tags": {k: v for k, v in d.tags.items() if not k.startswith("mlflow.log-model")},
        }
    return pd.DataFrame(rows), details


@st.cache_data(ttl=10, show_spinner=False)
def load_logged_models(base_dir_str: str) -> pd.DataFrame:
    """MLflow-3-'Logged Models' (haben eine eigene model_id)."""
    client = get_client(Path(base_dir_str))
    if not hasattr(client, "search_logged_models"):
        return pd.DataFrame()
    try:
        exp_ids = [e.experiment_id for e in client.search_experiments()]
        models = client.search_logged_models(experiment_ids=exp_ids, max_results=1000)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return pd.DataFrame([{
        "Modell-ID": getattr(m, "model_id", ""),
        "Modell-Name": getattr(m, "name", ""),
        "Run-ID": getattr(m, "source_run_id", ""),
        "Experiment-ID": getattr(m, "experiment_id", ""),
        "Status": str(getattr(m, "status", "")),
        "Erstellt": fmt_ts(getattr(m, "creation_timestamp", None)),
        "Artefakt-Ort": getattr(m, "artifact_location", ""),
    } for m in models])


@st.cache_data(ttl=10, show_spinner=False)
def load_registered(base_dir_str: str) -> pd.DataFrame:
    client = get_client(Path(base_dir_str))
    rows = []
    try:
        for rm in client.search_registered_models():
            for v in rm.latest_versions or []:
                rows.append({
                    "Modell-Name": rm.name,
                    "Version": v.version,
                    "Stage": v.current_stage,
                    "Run-ID": v.run_id,
                    "Status": v.status,
                    "Erstellt": fmt_ts(v.creation_timestamp),
                    "Quelle": v.source,
                })
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deployment (Gewichte in die Coin-Counter-App kopieren)
# ---------------------------------------------------------------------------
DEPLOY_PATH = "/Users/philipprutimann/Documents/ZHAW/CAS_ADML/MLOps/Projekt/app_coin_counter/best.pt"


@st.cache_data(show_spinner=False)
def file_hash(path_str: str, mtime: float) -> str:
    """SHA-256 einer Datei (mtime nur als Cache-Schlüssel)."""
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_of(path: Path) -> str:
    return file_hash(str(path), path.stat().st_mtime)


def deploy_panel(base_dir: Path, df: pd.DataFrame, details: dict):
    target = Path(st.text_input("Zielort", value=DEPLOY_PATH, key="mv_target").strip())

    # Was liegt aktuell am Zielort?
    current_hash = None
    if target.exists():
        st.caption(f"Aktuell am Zielort: {target.stat().st_size / 1e6:.1f} MB · "
                   f"geändert {datetime.fromtimestamp(target.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
        current_hash = hash_of(target)
        match = [rid for rid, d in details.items()
                 if d["weights"] and hash_of(Path(d["weights"])) == current_hash]
        if match:
            names = df.set_index("Run-ID").loc[match, "Run-Name"]
            st.success("Aktuell deployed: " + ", ".join(f"{n} ({rid})" for rid, n in names.items()))
        else:
            st.warning("Die Datei am Zielort stimmt mit keinem bekannten Run überein.")
    else:
        st.info("Am Zielort liegt noch keine Datei.")

    candidates = df[df["Gewichte (best.pt)"] == "✅"]
    if candidates.empty:
        st.info("Keine Runs mit Gewichten (best.pt) gefunden.")
        return

    rid = st.selectbox(
        "Modell (Run) zum Deployen", candidates["Run-ID"],
        format_func=lambda r: (
            f"{candidates.loc[candidates['Run-ID'] == r, 'Run-Name'].iloc[0]}  ·  "
            f"{candidates.loc[candidates['Run-ID'] == r, 'Start'].iloc[0]}  ·  {r[:8]}"),
        key="mv_deploy_run",
    )
    src = Path(details[rid]["weights"])
    row = candidates[candidates["Run-ID"] == rid].iloc[0]
    st.caption(f"Quelle: `{src}`  ·  Dataset-Version: {row['Dataset-Version'] or '–'}  ·  "
               f"mAP50-95 (Training): {row['mAP50-95 (Training)']}")

    res_file = base_dir / "evaluation_results" / rid / "results.json"
    if res_file.exists():
        try:
            m = json.loads(res_file.read_text())["clean"]["metrics"]
            st.caption(f"Test-Set-Evaluation: mAP50 {m['mAP50']:.3f} · mAP50-95 {m['mAP50-95']:.3f} · "
                       f"Precision {m['precision']:.3f} · Recall {m['recall']:.3f}")
        except Exception:  # noqa: BLE001
            pass
    else:
        st.caption("Für diesen Run liegt noch keine Test-Set-Evaluation vor.")

    already = current_hash is not None and hash_of(src) == current_hash
    if already:
        st.info("Dieses Modell ist bereits am Zielort deployed.")

    confirm = st.checkbox("Ja, die Datei am Zielort ersetzen" if target.exists() else "Datei am Zielort anlegen",
                          key="mv_confirm")
    backup = st.checkbox("Vorherige Datei als best.pt.bak sichern", value=False, key="mv_backup",
                         disabled=not target.exists())

    if st.button("🚀 Gewichte deployen", key="mv_deploy_btn", disabled=not confirm or already):
        tmp = target.with_name(target.name + ".tmp")
        try:
            if not target.parent.is_dir():
                raise FileNotFoundError(f"Zielordner existiert nicht: {target.parent}")
            shutil.copy2(src, tmp)  # zuerst daneben kopieren …
            if backup and target.exists():
                shutil.copy2(target, target.with_name(target.name + ".bak"))
            os.replace(tmp, target)  # … dann atomar ersetzen
            if hash_of(target) != hash_of(src):
                raise RuntimeError("Prüfsumme nach dem Kopieren stimmt nicht überein.")
            st.success(f"Deployed: {row['Run-Name']} → {target}")
            st.rerun(scope="fragment")
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            st.error(f"Deployment fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@st.fragment
def render_models(base_dir: Path):
    base_dir = Path(base_dir)
    st.header("Modelle (MLflow)")

    if not (base_dir / "mlflow_runs" / "mlflow.db").exists():
        st.info(f"Keine MLflow-Datenbank gefunden: {base_dir / 'mlflow_runs' / 'mlflow.db'}")
        return

    top_l, top_r = st.columns([6, 1])
    if top_r.button("🔄 Neu laden", key="mv_reload", use_container_width=True):
        load_runs.clear()
        load_logged_models.clear()
        load_registered.clear()
        st.rerun(scope="fragment")

    try:
        df, details = load_runs(str(base_dir))
    except Exception as e:  # noqa: BLE001
        st.error(f"MLflow konnte nicht gelesen werden: {e}")
        return
    if df.empty:
        st.info("Noch keine Runs vorhanden.")
        return

    t1, t2, t3, t4 = st.tabs(["Runs / Modelle", "Logged Models (model_id)",
                              "Registrierte Modelle", "🚀 Deployment"])

    # --- Runs ---------------------------------------------------------------
    with t1:
        c1, c2, c3 = st.columns([2, 2, 3])
        exp = c1.selectbox("Experiment", ["Alle"] + sorted(df["Experiment"].unique()), key="mv_exp")
        status = c2.selectbox("Status", ["Alle"] + sorted(df["Status"].unique()), key="mv_status")
        query = c3.text_input("Suche (Name, Run-ID, Dataset-Version …)", key="mv_query")

        view = df
        if exp != "Alle":
            view = view[view["Experiment"] == exp]
        if status != "Alle":
            view = view[view["Status"] == status]
        if query.strip():
            q = query.strip().lower()
            view = view[view.apply(lambda r: q in " ".join(map(str, r.values)).lower(), axis=1)]

        st.caption(f"{len(view)} von {len(df)} Runs")
        st.dataframe(view, hide_index=True, use_container_width=True)

        st.markdown("**Details zu einem Run**")
        if view.empty:
            st.info("Kein Run entspricht dem Filter.")
        else:
            run_id = st.selectbox(
                "Run", view["Run-ID"],
                format_func=lambda rid: f"{view.loc[view['Run-ID'] == rid, 'Run-Name'].iloc[0]}  ·  {rid}",
                key="mv_run",
            )
            d = details[run_id]
            st.caption(f"Gewichte: `{d['weights'] or 'nicht gefunden'}`")
            st.caption(f"Artefakt-URI: `{d['artifact_uri']}`")
            if d["weights"]:
                st.code(f"python evaluate_yolo.py --run-id {run_id} --noise-std 10 25 50", language="bash")
            p, m, t = st.tabs(["Parameter", "Metriken", "Tags"])
            for tab, content, col in ((p, d["params"], "Parameter"), (m, d["metrics"], "Wert"), (t, d["tags"], "Wert")):
                with tab:
                    if content:
                        st.dataframe(pd.DataFrame({col: content}).astype(str), use_container_width=True)
                    else:
                        st.caption("Keine Einträge.")

    # --- Logged Models --------------------------------------------------------
    with t2:
        lm = load_logged_models(str(base_dir))
        if lm.empty:
            st.info("Keine 'Logged Models' vorhanden (z. B. weil nur die Gewichte als Artefakt "
                    "geloggt wurden und nicht über mlflow.pytorch.log_model).")
        else:
            st.dataframe(lm, hide_index=True, use_container_width=True)

    # --- Registry ---------------------------------------------------------------
    with t3:
        rm = load_registered(str(base_dir))
        if rm.empty:
            st.info("Keine registrierten Modelle in der Model Registry.")
        else:
            st.dataframe(rm, hide_index=True, use_container_width=True)

    # --- Deployment ---------------------------------------------------------------
    with t4:
        deploy_panel(base_dir, df, details)