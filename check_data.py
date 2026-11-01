from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
import yaml

ANNOTATION_DIR = Path("./dataset/labels")
IMAGE_DIR = Path("./dataset/images")
IMAGE_EXTENSIONS = (".jpg", ".jpeg")
YAMEL_PATH = "dataset/data.yaml"


def get_class_names_from_yaml(yaml_path="dataset/data.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


CLASS_NAMES = get_class_names_from_yaml(YAMEL_PATH)

def find_image(txt_file):
    """Sucht das Bild im gleichen Split-Ordner (train/val/test) wie das Label."""
    split_dir = txt_file.parent.relative_to(ANNOTATION_DIR)   # z. B. 'train'
    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / split_dir / (txt_file.stem + ext)
        if candidate.exists():
            return candidate.name
    return None 


def extract_yolo_annotations(annotation_dir=ANNOTATION_DIR, class_names=CLASS_NAMES):
    rows = []
    txt_files = sorted(Path(annotation_dir).rglob("*.txt"))
    print(f"{len(txt_files)} Label-Dateien gefunden.")

    for txt_file in txt_files:
        split = txt_file.parent.relative_to(annotation_dir).as_posix() or None
        image_name = find_image(txt_file)

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]

        if not lines:   # leere Datei -> kein Objekt annotiert
            rows.append({
                "split": split, 
                "image_name": image_name,
                "class_id": None, 
                "class_name": None,
                "x_center": None, 
                "y_center": None,
                "width": None, 
                "height": None,
            })
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            rows.append({
                "split": split, 
                "image_name": image_name,
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"UNKNOWN_{class_id}"),
                "x_center": x_c, 
                "y_center": y_c, 
                "width": w, 
                "height": h,
            })

    df = pd.DataFrame(rows)

    # Objektanzahl pro Bild (leere Labels zählen als 0)
    df["num_objects"] = df.groupby(["split", "image_name"], dropna=False)["class_id"].transform("count")
    df["area"] = df["width"] * df["height"]
    return df
 
# ------------------------------------
# Kompletter Workflow auf einen Blick
# ------------------------------------

def qc_data():
    # 1. Extrahieren
    df = extract_yolo_annotations(ANNOTATION_DIR, CLASS_NAMES)

    # 2. Context, Data Source, Asset, Batch
    context = gx.get_context(mode="file", project_root_dir="./ge_expectations")
    data_source = context.data_sources.add_or_update_pandas("yolo_annotations")
    asset = data_source.add_dataframe_asset(name="annotations_df")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_df")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # 3. Suite zusammenstellen
    suite = gx.ExpectationSuite(name="yolo_annotations_suite")

    suite.add_expectation(gxe.ExpectTableColumnsToMatchOrderedList(column_list=[
        "split", 
        "image_name", 
        "class_id", 
        "class_name",
        "x_center", 
        "y_center", 
        "width", 
        "height",
        "num_objects", 
        "area",
    ]))


    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="image_name"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="class_id", value_set=list(CLASS_NAMES.keys())))
    suite.add_expectation(gxe.ExpectColumnValuesToNotMatchRegex(
        column="class_name", regex=r"^UNKNOWN_"))
    for col in ["x_center", "y_center", "width", "height"]:
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
            column=col, min_value=0.0, max_value=1.0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="num_objects", min_value=0, max_value=50))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="area", min_value=0.0001, max_value=0.95))

    suite = context.suites.add_or_update(suite)   # in ./ge_expectations gespeichert


    # 4. Validieren & Report
    result = batch.validate(suite)
    print("Gesamtergebnis:", "OK" if result.success else "FEHLER")
    for r in result.results:
        status = "✅" if r.success else "❌"
        print(status, r.expectation_config.type, r.expectation_config.kwargs.get("column", ""))


    # 5. Validation Definition + Checkpoint (speichert die Ergebnisse und aktualisiert die Data Docs)
    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(
            name="yolo_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="yolo_checkpoint",
            validation_definitions=[validation_definition],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
        )
    )

    result = checkpoint.run(batch_parameters={"dataframe": df})
    print("Gesamtergebnis:", "OK" if result.success else "FEHLER")

    # 6. Report bauen und im Browser öffnen
    context.build_data_docs()
    #context.open_data_docs()
    return "Gesamtergebnis:", "OK" if result.success else "FEHLER"
