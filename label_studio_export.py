import inspect
import os
import shutil
import sys
import time
import warnings
import zipfile
from urllib.parse import parse_qs, unquote, urlparse

import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import connect_label_studio_sdk

# --- PYDANTIC / SDK WARNUNGEN STUMMSCHALTEN ---
warnings.filterwarnings("ignore", message="Pydantic serialization warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def ls_export():

    # --- 1. KONFIGURATION ---
    LABEL_STUDIO_URL = "http://localhost:8080"
    PROJECT_ID = 1
    # Token NICHT im Code speichern: vorher z. B. `export LS_API_TOKEN="..."` setzen
    API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODEwMDE0MjI2NCwiaWF0IjoxNzkyOTQyMjY0LCJqdGkiOiIwNDlmNjA0YjBkNjc0YWE4YWFhYTUyZjJhYTg4MzQ5ZiIsInVzZXJfaWQiOiIxIn0.MhzpiUOCeO6z_OvSo4TpKpgEjFBq3CiyKGXDCO3OD-o"

    FINAL_DIR = "./dataset"        # Das gewünschte YOLO-Zielverzeichnis
    TEMP_DIR = "./temp_ls_export"  # Temporärer Ordner zum Entpacken
    ZIP_FILENAME = "export.zip"

    EXPORT_FORMAT = "YOLO_WITH_IMAGES"
    EXPORT_POLL_INTERVAL = 2       # Sekunden zwischen Status-Abfragen
    EXPORT_TIMEOUT = 600           # Max. Wartezeit auf den Export in Sekunden

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    # --- HILFSFUNKTIONEN: BILDER DIREKT ÜBER DIE API LADEN ---
    def get_auth_headers():
        """Persönliche Access-Tokens (JWT) werden gegen ein Access-Token getauscht,
        sonst wird der klassische 'Token ...'-Header verwendet."""
        try:
            r = requests.post(
                f"{LABEL_STUDIO_URL}/api/token/refresh",
                json={"refresh": API_TOKEN},
                timeout=30,
            )
            if r.ok and "access" in r.json():
                return {"Authorization": f"Bearer {r.json()['access']}"}
        except Exception:
            pass
        return {"Authorization": f"Token {API_TOKEN}"}


    def find_image_ref(task_data):
        """Sucht im Task den Wert, der auf ein Bild zeigt (z. B. data['image'])."""
        for value in (task_data or {}).values():
            if isinstance(value, str) and any(ext in value.lower() for ext in IMAGE_EXTENSIONS):
                return value
        return None


    def download_images_via_api(ls_client, label_paths, target_dir):
        """Lädt die Bilder aller Tasks, für die ein Label vorhanden ist, direkt von Label Studio."""
        os.makedirs(target_dir, exist_ok=True)
        wanted_stems = {os.path.splitext(n)[0] for n in label_paths}
        headers = get_auth_headers()
        downloaded = {}
        missing = 0

        tasks = list(ls_client.tasks.list(project=PROJECT_ID, fields="all"))
        print(f"   {len(tasks)} Tasks im Projekt gefunden.")

        for task in tqdm(tasks, desc="Lade Bilder"):
            ref = find_image_ref(getattr(task, "data", None))
            if not ref:
                missing += 1
                continue

            parsed = urlparse(ref)
            query = parse_qs(parsed.query)
            if "d" in query:   # Local-Files-Storage: /data/local-files/?d=ordner/bild.jpg
                file_name = os.path.basename(unquote(query["d"][0]))
            else:
                file_name = os.path.basename(unquote(parsed.path))

            if os.path.splitext(file_name)[0] not in wanted_stems:
                continue   # kein Label für dieses Bild -> überspringen

            url = ref if ref.startswith("http") else f"{LABEL_STUDIO_URL}{ref}"
            req_headers = headers if url.startswith(LABEL_STUDIO_URL) else {}
            try:
                resp = requests.get(url, headers=req_headers, stream=True, timeout=60)
                resp.raise_for_status()
            except Exception as err:
                print(f"   ⚠ {file_name}: {err}")
                missing += 1
                continue

            dst = os.path.join(target_dir, file_name)
            with open(dst, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    out.write(chunk)
            downloaded[file_name] = dst

        print(f"   {len(downloaded)} Bilder geladen, {missing} nicht verfügbar.")
        return downloaded


    # --- 2. BEREINIGUNG ALTER ORDNER ---
    for folder in [FINAL_DIR, TEMP_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    try:
        # --- 3. CONNECT & EXPORT VIA SDK (v2.x Syntax) ---
        print("Verbinde mit Label Studio SDK...")
        ls = connect_label_studio_sdk(url=LABEL_STUDIO_URL, api_token=API_TOKEN, project_id=PROJECT_ID)

        print(f"Starte Full-Export von Projekt {PROJECT_ID} im Format '{EXPORT_FORMAT}'...")

        # 3a. Export-Job anlegen (liefert die Export-ID = export_pk)
        export_job = ls.projects.exports.create(id=PROJECT_ID)
        export_pk = export_job.id
        print(f"Export-Job erstellt (ID {export_pk}), warte auf Fertigstellung...")

        # 3b. Warten, bis der Export fertig ist
        start = time.time()
        while True:
            job = ls.projects.exports.get(id=PROJECT_ID, export_pk=export_pk)
            status = str(job.status).lower()
            if "completed" in status:
                break
            if "failed" in status:
                raise RuntimeError(f"Export fehlgeschlagen (Status: {job.status})")
            if time.time() - start > EXPORT_TIMEOUT:
                raise TimeoutError("Export dauert zu lange (Timeout überschritten).")
            time.sleep(EXPORT_POLL_INTERVAL)

        # 3c. Export herunterladen (Stream -> ZIP-Datei)
        # Je nach SDK-Version unterstützt download() unterschiedliche Parameter.
        # Wir übergeben nur die, die die installierte Version wirklich kennt.
        wanted_kwargs = {
            "id": PROJECT_ID,
            "export_pk": export_pk,
            "export_type": EXPORT_FORMAT,
            "download_all_tasks": True,   # False = nur beschriftete Tasks
            "download_resources": True,   # Bilddateien mit in die ZIP aufnehmen
        }
        supported = inspect.signature(ls.projects.exports.download).parameters
        download_kwargs = {k: v for k, v in wanted_kwargs.items() if k in supported}
        skipped = sorted(set(wanted_kwargs) - set(download_kwargs))
        if skipped:
            print(f"ℹ Von dieser SDK-Version nicht unterstützt, übersprungen: {', '.join(skipped)}")

        export_stream = ls.projects.exports.download(**download_kwargs)

        zip_filepath = os.path.join(TEMP_DIR, ZIP_FILENAME)
        with open(zip_filepath, "wb") as f:
            if hasattr(export_stream, "__iter__") and not isinstance(export_stream, (bytes, bytearray)):
                for chunk in export_stream:
                    if chunk:
                        f.write(chunk)
            else:
                f.write(export_stream)

        print("✅ Export-ZIP erfolgreich heruntergeladen.")

        # --- 4. ZIP-ARCHIV ENTPACKEN ---
        print("Entpacke ZIP-Archiv...")
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(TEMP_DIR)

        # Inhalt des ZIPs zur Diagnose anzeigen
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            names = [n for n in zip_ref.namelist() if not n.endswith("/")]
        print(f"🔎 Das ZIP enthält {len(names)} Dateien. Erste Einträge:")
        for n in names[:15]:
            print(f"   - {n}")

        # Bilder und Labels rekursiv suchen (die Ordnerstruktur variiert je nach Version)
        image_paths = {}   # Dateiname -> voller Pfad
        label_paths = {}   # Dateiname (.txt) -> voller Pfad
        meta_paths = {}    # classes.txt / notes.json -> voller Pfad
        for root, _, files in os.walk(TEMP_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                if full == zip_filepath:
                    continue
                low = fn.lower()
                if low.endswith(IMAGE_EXTENSIONS):
                    image_paths[fn] = full
                elif low in ("classes.txt", "notes.json", "notes.txt"):
                    meta_paths[fn] = full
                elif low.endswith(".txt"):
                    label_paths[fn] = full

        all_images = sorted(image_paths)
        print(f"📦 {len(all_images)} Bilder und {len(label_paths)} Label-Dateien im ZIP-Archiv gefunden.")

        if not all_images:
            print("⚠ Keine Bilder im ZIP – lade sie direkt über die Label-Studio-API nach...")
            image_paths = download_images_via_api(
                ls, label_paths, os.path.join(TEMP_DIR, "downloaded_images")
            )
            all_images = sorted(image_paths)
            if not all_images:
                print("⚠ Auch über die API konnten keine Bilder geladen werden. Siehe Meldungen oben.")
                sys.exit(1)
            print(f"📦 Jetzt {len(all_images)} Bilder verfügbar.")

        # --- 5. DATA SPLIT BERECHNEN ---
        train_imgs, rest_imgs = train_test_split(
            all_images, train_size=TRAIN_RATIO, random_state=42
        )
        val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_imgs, test_imgs = train_test_split(
            rest_imgs, train_size=val_size_adjusted, random_state=42
        )

        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

        # --- 6. DATEIEN AUFSPLITTEN & IN ZIELORDNER KOPIEREN ---
        for split_name, img_list in splits.items():
            print(f"📂 Erstelle '{split_name}'-Split ({len(img_list)} Bilder)...")

            dst_img_dir = os.path.join(FINAL_DIR, "images", split_name)
            dst_lbl_dir = os.path.join(FINAL_DIR, "labels", split_name)
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)

            for img_name in tqdm(img_list, desc=f"Kopiere {split_name}"):
                base_name, _ = os.path.splitext(img_name)
                lbl_name = f"{base_name}.txt"

                src_img_path = image_paths.get(img_name, "")
                src_lbl_path = label_paths.get(lbl_name, "")
                dst_img_path = os.path.join(dst_img_dir, img_name)
                dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)

                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)

                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_lbl_path, dst_lbl_path)

        # Metadaten (Klassenübersichten) kopieren, falls vorhanden
        for meta_file, meta_src in meta_paths.items():
            shutil.copy2(meta_src, os.path.join(FINAL_DIR, meta_file))

        # --- 7. OPTIONAL: data.yaml für Ultralytics YOLO erzeugen ---
        classes_path = os.path.join(FINAL_DIR, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]
            yaml_path = os.path.join(FINAL_DIR, "data.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(f"path: {os.path.abspath(FINAL_DIR)}\n")
                f.write("train: images/train\n")
                f.write("val: images/val\n")
                f.write("test: images/test\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write("names:\n")
                for i, name in enumerate(class_names):
                    f.write(f"  {i}: {name}\n")
            print(f"📝 data.yaml mit {len(class_names)} Klassen erstellt.")

        print(f"\n🎉 Datensatz erfolgreich heruntergeladen, gesplittet und unter '{FINAL_DIR}' abgelegt!")

    except Exception as e:
        print(f"💥 Fehler beim SDK-Export oder Splitting: {e}")
        raise
    finally:
        # Temporären Export-Ordner aufräumen
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            
    return f"\n🎉 Datensatz erfolgreich heruntergeladen, gesplittet und unter '{FINAL_DIR}' abgelegt!"