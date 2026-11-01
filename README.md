# my_yolo_projekt

End-to-End-MLOps-Projekt rund um **Objekterkennung mit YOLOv8** (Ultralytics). Das Projekt entstand im Rahmen des MLOps-Moduls (ZHAW, CAS ADML) und deckt den gesamten Lebenszyklus ab: Daten annotieren, Datenqualität prüfen, Daten versionieren, Modelle trainieren und tracken, evaluieren und das beste Modell deployen. Gesteuert wird alles über ein zentrales **Streamlit Control Panel**.

Das Zielmodell wird in eine separate *Coin-Counter-App* deployed (Münzen erkennen und zählen).

---

## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Was wurde in diesem Projekt gemacht?](#was-wurde-in-diesem-projekt-gemacht)
3. [Projektstruktur](#projektstruktur)
4. [Tech-Stack](#tech-stack)
5. [Installation](#installation)
6. [Konfiguration](#konfiguration)
7. [Benutzung](#benutzung)
8. [Evaluation im Detail](#evaluation-im-detail)
9. [Datenversionierung im Detail](#datenversionierung-im-detail)
10. [Bekannte Einschränkungen](#bekannte-einschränkungen)

---

## Überblick

```
Label Studio ──► Export & Split ──► Great Expectations ──► DVC + Git-Tag
 (Annotation)    (train/val/test)    (Datenqualität)       (Versionierung)
                                                                │
                                                                ▼
 Coin-Counter-App ◄── Deployment ◄── Evaluation ◄── Training (YOLOv8 + MLflow)
                      (best.pt)      (Test-Set,      (Hyperparameter, Metriken,
                                      Robustheit)     Daten-Version als Tag)
```

Alle Schritte lassen sich per Knopfdruck im Streamlit-Panel (`app.py`) auslösen oder einzeln als Skript ausführen.

---

## Was wurde in diesem Projekt gemacht?

### 1. Datenannotation mit Label Studio
- Bilder werden in **Label Studio** mit Bounding Boxes annotiert.
- `label_studio_export.py` exportiert das Projekt über das Label-Studio-SDK im Format `YOLO_WITH_IMAGES`, wartet auf den Export-Job, lädt das ZIP herunter und entpackt es.
- Fehlen Bilder im ZIP, werden sie automatisch direkt über die Label-Studio-API nachgeladen (inkl. Token-Handling für JWT-Access-Tokens).
- Der Datensatz wird reproduzierbar (`random_state=42`) in **70 % Train / 15 % Val / 15 % Test** aufgeteilt und in die YOLO-Struktur `dataset/images/<split>` und `dataset/labels/<split>` geschrieben.
- Die `data.yaml` für Ultralytics wird aus der `classes.txt` automatisch erzeugt.

### 2. Datenqualitätsprüfung mit Great Expectations
`check_data.py` liest alle YOLO-Label-Dateien in einen Pandas-DataFrame (ein Eintrag pro Bounding Box, inkl. Split, Klasse, Position, Grösse, Fläche und Objektanzahl pro Bild) und validiert ihn mit einer Expectation-Suite (`yolo_annotations_suite`):

| Prüfung | Zweck |
|---|---|
| Spalten stimmen mit erwarteter Reihenfolge überein | Schema-Check |
| `image_name` nicht null | Jedes Label hat ein zugehöriges Bild |
| `class_id` in der Menge der Klassen aus `data.yaml` | Keine ungültigen Klassen |
| `class_name` matcht nicht `^UNKNOWN_` | Keine unbekannten Klassen |
| `x_center`, `y_center`, `width`, `height` zwischen 0 und 1 | Gültige normalisierte Koordinaten |
| `num_objects` zwischen 0 und 50 | Plausible Objektanzahl pro Bild |
| `area` zwischen 0.0001 und 0.95 | Keine winzigen/riesigen Boxen |

Ergebnis ist ein **Data-Docs-Report** (HTML), der direkt aus dem Panel im Browser geöffnet werden kann.

### 3. Datenversionierung mit DVC und Git
- Der Ordner `dataset/` wird mit **DVC** versioniert (`dataset.dvc`, `.dvc/`); die Daten selbst liegen nicht in Git, sondern in einem DVC-Remote/Cache.
- `version_data.sh` automatisiert den Ablauf: nächste Major-Version aus dem letzten Git-Tag berechnen (`v1.0`, `v2.0`, …), `dvc add`, `dvc push`, Commit der Pointer-Datei und Git-Tag setzen.
- Dadurch ist jede Datenversion eindeutig einem Git-Tag, einem Commit und einem DVC-MD5-Hash zugeordnet.

### 4. Modelltraining mit YOLOv8 und MLflow-Tracking
`train.py` trainiert ein YOLOv8-Modell (Standard: `yolov8n.pt`; `yolov8s.pt` und `yolov8m.pt` liegen ebenfalls im Repo) und trackt alles in **MLflow** (SQLite-Backend unter `mlflow_runs/mlflow.db`, Experiment `YOLO_Mac_Training`):

- **Hyperparameter**: Epochs (80), Batch Size (16), Bildgrösse (640), Dropout (0.5), Weight Decay (0.0005), Optimizer (`auto`)
- **Apple-Silicon-Optimierung**: Training auf der GPU via `device="mps"`, Caching im RAM
- **Daten-Lineage**: Als MLflow-Tags werden die Datensatz-Version (Git-Tag), der DVC-MD5-Hash und der Git-Commit gespeichert; `data.yaml` und `dataset.dvc` werden als Artefakte mitgeloggt. So ist jedes Modell auf die exakten Trainingsdaten zurückführbar.
- **Automatische Run-Namen** (`YOLO_Train_vX.Y`) anhand des neuesten Tags bzw. des letzten MLflow-Runs
- Das beste Modell (`best.pt`) wird zusätzlich als PyTorch-Modell in MLflow geloggt.

### 5. Evaluation auf dem Test-Set
`evaluate_yolo.py` evaluiert ein MLflow-Modell auf dem Test-Set (siehe [Evaluation im Detail](#evaluation-im-detail)): Metriken, die schwierigsten Bilder und ein Robustheitstest mit Gauss-Rauschen.

### 6. Streamlit Control Panel
`app.py` ist die zentrale Oberfläche ("MLOps Control Panel"):

- **Dienste** (MLflow-Server, Label Studio) starten, stoppen und neu starten, mit Statusanzeige (läuft / startet / gestoppt), Fortschrittsbalken, Log-Ansicht und Direktlink
- **Aktionen** per Knopfdruck: Daten aus Label Studio laden, GX-Report erstellen und öffnen, Daten versionieren, alle Ports aufräumen
- **Abschnitt "Evaluation"** (`eval_view.py`): Ergebnisse je Run mit Tabs *Übersicht*, *Pro Klasse*, *Robustheit* und *Schwierigste Fälle*; neue Evaluationen können direkt gestartet werden
- **Abschnitt "Modelle (MLflow)"** (`models_view.py`): Übersicht aller Runs (inkl. Dataset-Version, DVC-Hash, Git-Commit, Metriken), Logged Models, Model Registry und ein **Deployment-Panel**

### 7. Deployment
Im Deployment-Panel kann ein Run ausgewählt und dessen `best.pt` in die Coin-Counter-App kopiert werden. Dabei wird:
- per SHA-256 geprüft, welcher Run aktuell deployed ist,
- das Modell atomar ersetzt (erst Kopie, dann `os.replace`),
- optional ein Backup (`best.pt.bak`) angelegt,
- nach dem Kopieren die Prüfsumme verifiziert.

### 8. Zusätzliches
- **Airflow** ist als Abhängigkeit eingebunden; `airflow.txt` enthält Notizen zum lokalen Start (`airflow standalone`, Port 8585).
- **SAHI** (Slicing Aided Hyper Inference) ist als Abhängigkeit vorhanden, z. B. für Inferenz auf grossen Bildern.
- Abhängigkeiten werden mit **uv** verwaltet (`pyproject.toml`, `uv.lock`).
- `mlflow_data.py` bietet Hilfsfunktionen für den Zugriff auf die MLflow-Daten.

---

## Projektstruktur

```
my_yolo_projekt/
├── app.py                    # Streamlit Control Panel (Einstiegspunkt)
├── config.py                 # Ports, Pfade, Backend-URIs
├── utils.py                  # Start/Stop der Dienste, Port-Handling, LS-Verbindung, Versionierung
├── label_studio_export.py    # Export aus Label Studio, Train/Val/Test-Split, data.yaml
├── check_data.py             # Datenqualität mit Great Expectations
├── version_data.sh           # Daten-Versionierung (DVC + Git-Tag)
├── train.py                  # YOLO-Training mit MLflow-Tracking
├── evaluate_yolo.py          # Evaluation: Metriken, schwierigste Fälle, Robustheit
├── eval_view.py              # Streamlit-Ansicht für Evaluationsergebnisse
├── models_view.py            # Streamlit-Ansicht für MLflow-Modelle + Deployment
├── mlflow_data.py            # Hilfsfunktionen für MLflow-Daten
├── dataset.dvc               # DVC-Pointer auf den versionierten Datensatz
├── .dvc/, .dvcignore         # DVC-Konfiguration
├── mlflow_runs/              # MLflow-Backend (SQLite)
├── mlruns/                   # MLflow-Artefakte (Gewichte, Modelle)
├── yolov8{n,s,m}.pt          # Vortrainierte YOLOv8-Basismodelle
├── airflow.txt               # Notizen zum Airflow-Start
├── pyproject.toml, uv.lock   # Abhängigkeiten
└── .python-version
```

Nicht im Repo (werden erzeugt): `dataset/` (via DVC), `evaluation_results/`, `ge_expectations/`.

---

## Tech-Stack

| Bereich | Werkzeug |
|---|---|
| Objekterkennung | Ultralytics YOLOv8, PyTorch |
| Annotation | Label Studio (`label-studio-sdk`, `label-studio-converter`) |
| Datenqualität | Great Expectations |
| Datenversionierung | DVC, Git-Tags |
| Experiment-Tracking | MLflow (SQLite-Backend) |
| Orchestrierung | Apache Airflow |
| Inferenz grosser Bilder | SAHI |
| UI | Streamlit |
| Tests | pytest |
| Paketverwaltung | uv (Python ≥ 3.12) |

---

## Installation

Voraussetzungen: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), Git, DVC. Entwickelt und getestet auf macOS (Apple Silicon).

```bash
git clone https://github.com/ruphilip/my_yolo_projekt.git
cd my_yolo_projekt
uv sync
```

Zusätzlich wird eine separate **Label-Studio**-Installation (eigene virtuelle Umgebung) benötigt, die vom Panel gestartet wird.

---

## Konfiguration

Die zentralen Einstellungen stehen in `config.py`:

| Variable | Bedeutung | Standard |
|---|---|---|
| `MLFLOW_PORT` | Port des MLflow-Servers (5000 ist auf macOS durch AirPlay belegt) | `5050` |
| `MLFLOW_BACKEND` | MLflow-Backend-Store | `sqlite:///mlflow_runs/mlflow.db` |
| `LS_PORT` | Port von Label Studio | `8080` |
| `LS_DIR` / `LS_VENV` | Ordner und virtuelle Umgebung der Label-Studio-Installation | **an eigenes System anpassen** |
| `GX_SEARCH_DIRS` / `GX_REPORT` | Suchorte bzw. fester Pfad der Great-Expectations-Data-Docs | `ge_expectations/` |

Weitere Anpassungen vor der Nutzung:
- `PROJECT_ID` und `LABEL_STUDIO_URL` in `label_studio_export.py`
- `DEPLOY_PATH` in `models_view.py` (Zielort der Coin-Counter-App)
- Standard-Gewichtspfad (`--weights`) in `evaluate_yolo.py`

> **Wichtig:** API-Tokens und Passwörter gehören **nicht** ins Repository. Das Label-Studio-Token sollte z. B. über eine Umgebungsvariable (`export LS_API_TOKEN="..."`) bereitgestellt werden.

---

## Benutzung

### Control Panel starten

```bash
uv run streamlit run app.py
```

Typischer Ablauf im Panel:

1. **Label Studio** und **MLflow** starten
2. *Daten von Label Studio laden* (Export, Split, `data.yaml`)
3. *GX Report erstellen* und *Great Expectations Report* prüfen
4. *Versioniere die Daten* (DVC + Git-Tag)
5. Training starten (siehe unten)
6. Im Abschnitt *Evaluation* das Modell evaluieren
7. Im Abschnitt *Modelle (MLflow)* → *Deployment* das beste Modell ausrollen

### Training

```bash
uv run python train.py
```

Hyperparameter werden am Anfang von `train.py` angepasst (`EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`, `YOLO_MODEL`, `DROPOUT`, `WEIGHT_D`, `OPTIMIZER`).

### Evaluation

```bash
uv run python evaluate_yolo.py                              # neuester Run, Rauschen σ=25
uv run python evaluate_yolo.py --noise-std 10 25 50         # mehrere Rauschstärken
uv run python evaluate_yolo.py --run-id <RUN_ID> --log-mlflow
```

### Dienste einzeln starten

```bash
mlflow server --backend-store-uri sqlite:///mlflow_runs/mlflow.db --port 5050
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8585 && airflow standalone
```

---

## Evaluation im Detail

`evaluate_yolo.py` lädt `best.pt` eines MLflow-Runs und führt drei Analysen durch:

1. **Performance auf dem sauberen Test-Set**: Precision, Recall, F1, mAP@50, mAP@75, mAP@50-95 sowie AP pro Klasse.
2. **Schwierigste Bilder**: Pro Bild wird der Loss (box + cls + dfl) berechnet. Die K Bilder mit dem höchsten Loss werden visualisiert (grün = Ground Truth, rot = Vorhersage). Zusätzlich wird die Loss-Verteilung über alle Test-Bilder als CSV gespeichert.
3. **Robustheitstest**: Die Test-Bilder werden mit Gauss-Rauschen verschiedener Stärke (σ) versehen und neu bewertet. Verglichen werden absolute und relative Veränderungen gegenüber dem sauberen Test-Set.

Ergebnisse landen in `evaluation_results/<run_id>/` (`results.json`, `robustness_comparison.csv`, `loss_per_image.csv`, `hardest_cases/`, Beispielbilder). Mit `--log-mlflow` werden Metriken und Artefakte zusätzlich im MLflow-Run gespeichert. Das Device wird automatisch gewählt (MPS, CUDA oder CPU).

---

## Datenversionierung im Detail

Jedes trainierte Modell ist über drei Verknüpfungen auf seine Daten zurückführbar:

| Information | Quelle | Speicherort in MLflow |
|---|---|---|
| Datensatz-Version | letzter Git-Tag (`vX.Y`) | Tag `dataset_version` |
| Daten-Hash | MD5 aus `dataset.dvc` | Tag `dvc_data_md5` |
| Code-/Pointer-Stand | `git rev-parse HEAD` | Tag `dataset.git_commit` |

Zusätzlich liegen `data.yaml` und `dataset.dvc` als Artefakte beim Run. Mit `git checkout <tag>` und `dvc checkout` lässt sich der exakte Datenstand wiederherstellen.

---

## Bekannte Einschränkungen

- Einige Pfade in `config.py` und `models_view.py` sind auf das lokale macOS-System des Autors zugeschnitten und müssen angepasst werden.
- `kill_port` nutzt `lsof` und funktioniert damit nur unter macOS/Linux.
- Training ist auf Apple Silicon (`mps`) ausgelegt; für andere Systeme muss `device` in `train.py` angepasst werden.
- Es gibt aktuell keine automatisierten Tests, obwohl `pytest` als Abhängigkeit eingetragen ist.