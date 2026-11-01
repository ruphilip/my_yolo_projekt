"""
Streamlit-Abschnitt "Evaluation": zeigt die Ergebnisse von evaluate_yolo.py je Run.

Einbindung in app.py (2 Zeilen):
    from eval_view import render_evaluation          # bei den Imports
    render_evaluation(BASE_DIR)                      # ganz am Ende der Datei
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mAP50": "mAP@50",
    "mAP75": "mAP@75",
    "mAP50-95": "mAP@50-95",
}


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
def load_eval_runs(eval_dir: Path) -> dict[str, tuple[Path, dict]]:
    """name -> (Ordner, results.json). Neueste zuerst."""
    found = []
    files = list(eval_dir.glob("*/results.json"))
    if (eval_dir / "results.json").exists():  # ältere Ausgabe ohne Run-Unterordner
        files.append(eval_dir / "results.json")
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        name = f.parent.name if f.parent != eval_dir else "(ohne Run-ID)"
        found.append((f.stat().st_mtime, name, f.parent, data))
    found.sort(key=lambda x: x[0], reverse=True)
    return {name: (folder, data) for _, name, folder, data in found}


def noise_levels(data: dict) -> list[str]:
    return sorted(data.get("noisy", {}).keys(), key=float)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(data: dict):
    clean = data["clean"]["metrics"]
    cols = st.columns(len(METRIC_LABELS))
    for col, (key, label) in zip(cols, METRIC_LABELS.items()):
        col.metric(label, f"{clean[key]:.3f}")

    levels = noise_levels(data)
    if levels:
        st.markdown("**Mit Rauschen** (Änderung gegenüber sauberem Test-Set)")
        sigma = st.selectbox("Rauschstärke σ", levels, key=f"ov_sigma_{data.get('run_id')}")
        noisy = data["noisy"][sigma]["metrics"]
        cols = st.columns(len(METRIC_LABELS))
        for col, (key, label) in zip(cols, METRIC_LABELS.items()):
            col.metric(label, f"{noisy[key]:.3f}", delta=f"{noisy[key] - clean[key]:+.3f}")

    info = []
    if data.get("n_test_images"):
        info.append(f"{data['n_test_images']} Test-Bilder")
    if data.get("imgsz"):
        info.append(f"imgsz {data['imgsz']}")
    if data.get("created"):
        info.append(f"erstellt {data['created']}")
    if info:
        st.caption(" · ".join(info))


def tab_per_class(data: dict):
    df = pd.DataFrame({"sauber": data["clean"]["per_class"]})
    for s in noise_levels(data):
        df[f"σ={float(s):g}"] = pd.Series(data["noisy"][s]["per_class"])
    st.markdown("**AP@50-95 pro Klasse**")
    st.bar_chart(df)
    with st.expander("Tabelle"):
        st.dataframe(df.round(4), use_container_width=True)


def tab_robustness(data: dict, folder: Path):
    levels = noise_levels(data)
    if not levels:
        st.info("Für diesen Run wurde kein Robustheitstest durchgeführt.")
        return

    rows = {0.0: data["clean"]["metrics"]}
    for s in levels:
        rows[float(s)] = data["noisy"][s]["metrics"]
    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = "σ"
    df = df[list(METRIC_LABELS)].rename(columns=METRIC_LABELS)

    st.markdown("**Metriken in Abhängigkeit der Rauschstärke** (σ = 0: sauber)")
    st.line_chart(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Absolutwerte")
        st.dataframe(df.round(4), use_container_width=True)
    with c2:
        st.markdown("Relative Änderung gegenüber sauber [%]")
        rel = (df.loc[df.index > 0] / df.loc[0.0].replace(0, np.nan) - 1) * 100
        st.dataframe(rel.round(1), use_container_width=True)

    examples = sorted(folder.glob("noise_example_sigma*.png"))
    if examples:
        st.markdown("**Beispielbilder mit Rauschen**")
        cols = st.columns(min(len(examples), 4))
        for i, img in enumerate(examples):
            cols[i % len(cols)].image(
                str(img), caption="σ = " + img.stem.replace("noise_example_sigma", ""),
                use_container_width=True,
            )


def tab_hardest(data: dict, folder: Path):
    hard = data.get("hardest_cases", [])
    if not hard:
        st.info("Keine Daten zu den schwierigsten Fällen vorhanden.")
        return

    df = pd.DataFrame(hard)
    if "rank" not in df:
        df.insert(0, "rank", range(1, len(df) + 1))
    if "visualization" not in df:
        df["visualization"] = [f"{r:02d}_{Path(p).stem}.jpg" for r, p in zip(df["rank"], df["image"])]
    df["bild"] = df["image"].map(lambda p: Path(p).name)

    st.markdown("**Bilder mit dem höchsten Loss** (Loss = box + cls + dfl)")
    st.dataframe(
        df[["rank", "bild", "n_objects", "loss_total", "loss_box", "loss_cls", "loss_dfl"]].round(3),
        hide_index=True, use_container_width=True,
    )

    st.caption("Grün = Ground Truth · Rot = Vorhersage (conf ≥ 0.25)")
    cols = st.columns(2)
    for i, row in df.iterrows():
        img = folder / "hardest_cases" / row["visualization"]
        with cols[i % 2], st.container(border=True):
            if img.exists():
                st.image(str(img), use_container_width=True)
            st.caption(f"#{int(row['rank'])} · {row['bild']} · Loss {row['loss_total']:.3f} "
                       f"(box {row['loss_box']:.2f} / cls {row['loss_cls']:.2f} / dfl {row['loss_dfl']:.2f})")

    csv = folder / "loss_per_image.csv"
    if csv.exists():
        with st.expander("Loss-Verteilung über alle Test-Bilder"):
            losses = pd.read_csv(csv)["loss_total"]
            counts, edges = np.histogram(losses, bins=30)
            hist = pd.DataFrame({"Anzahl Bilder": counts}, index=[f"{e:.2f}" for e in edges[:-1]])
            st.bar_chart(hist)


# ---------------------------------------------------------------------------
# Neue Evaluation starten
# ---------------------------------------------------------------------------
def start_panel(base_dir: Path):
    with st.expander("▶️ Neue Evaluation starten"):
        run_id = st.text_input("MLflow Run-ID (leer = Standard-Gewichte aus evaluate_yolo.py)", key="ev_runid")
        noise = st.text_input("Rauschstärken σ (0–255, durch Leerzeichen getrennt)", value="10 25 50", key="ev_noise")
        log_mlflow = st.checkbox("Ergebnisse zusätzlich in MLflow loggen", key="ev_log")
        if st.button("Evaluation starten", key="ev_start"):
            cmd = [sys.executable, str(base_dir / "evaluate_yolo.py"), "--noise-std", *noise.split()]
            if run_id.strip():
                cmd += ["--run-id", run_id.strip()]
            if log_mlflow:
                cmd.append("--log-mlflow")
            with st.spinner("Evaluation läuft … (kann einige Minuten dauern)"):
                proc = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
            st.session_state["ev_output"] = (proc.returncode, proc.stdout + "\n" + proc.stderr)
            st.rerun(scope="fragment")

        if "ev_output" in st.session_state:
            code, out = st.session_state["ev_output"]
            (st.success if code == 0 else st.error)(
                "Evaluation abgeschlossen." if code == 0 else f"Evaluation fehlgeschlagen (Exit-Code {code}).")
            st.code(out[-6000:], language="text")


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------
@st.fragment
def render_evaluation(base_dir: Path, eval_dirname: str = "evaluation_results"):
    base_dir = Path(base_dir)
    eval_dir = base_dir / eval_dirname

    st.header("Evaluation")
    start_panel(base_dir)

    runs = load_eval_runs(eval_dir)
    if not runs:
        st.info(f"Noch keine Ergebnisse in '{eval_dir}'. Starte eine Evaluation (oben) "
                "oder führe `python evaluate_yolo.py` aus.")
        return

    names = list(runs)
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Evaluierter Run", names, key="ev_choice",
        format_func=lambda n: f"{n}  ·  {runs[n][1].get('created', '')}",
    )
    if c2.button("🔄 Neu laden", key="ev_reload", use_container_width=True):
        st.rerun(scope="fragment")

    folder, data = runs[choice]
    st.caption(f"Modell: `{data.get('model_uri', '?')}`")

    t1, t2, t3, t4 = st.tabs(["Übersicht", "Pro Klasse", "Robustheit", "Schwierigste Fälle"])
    with t1:
        tab_overview(data)
    with t2:
        tab_per_class(data)
    with t3:
        tab_robustness(data, folder)
    with t4:
        tab_hardest(data, folder)