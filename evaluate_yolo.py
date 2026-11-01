"""
Evaluation eines mit MLflow geloggten YOLO-Modells auf dem Test-Set.

1. Performance auf dem sauberen Test-Set (Precision, Recall, F1, mAP50, mAP75, mAP50-95, AP pro Klasse)
2. Die K schwierigsten Bilder (höchster Loss pro Bild: box + cls + dfl) inkl. Visualisierung
3. Robustheitstest: Test-Bilder mit Gauss-Rauschen versehen, Performance erneut messen und vergleichen

Beispiele:
    python evaluate_yolo.py                              # neuester Run, Rauschen sigma=25
    python evaluate_yolo.py --noise-std 10 25 50         # mehrere Rauschstärken
    python evaluate_yolo.py --run-id <RUN_ID> --log-mlflow
"""
import argparse
import copy
import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG

warnings.filterwarnings("ignore", category=UserWarning)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def load_yolo(args) -> tuple[YOLO, str, str | None]:
    """Lädt best.pt aus dem MLflow-Artefaktordner. Gibt (yolo, weights_path, run_id) zurück."""
    weights = Path(args.weights)
    if args.run_id:  # Run-ID angegeben -> Pfad im Artefaktordner suchen
        hits = sorted(Path(args.artifact_root).glob(f"*/{args.run_id}/artifacts/weights/best.pt"))
        if not hits:
            raise SystemExit(f"Keine best.pt für Run {args.run_id} unter '{args.artifact_root}' gefunden.")
        weights = hits[0]
    if not weights.exists():
        raise SystemExit(f"Gewichte nicht gefunden: {weights}")

    # Run-ID aus dem Pfad ableiten: .../<run_id>/artifacts/weights/best.pt
    run_id = weights.resolve().parts[-4] if weights.resolve().parts[-3] == "artifacts" else None
    return YOLO(str(weights)), str(weights), run_id


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def locate_split(data_dir: Path, split: str) -> tuple[Path, Path]:
    """Findet (images_dir, labels_dir) für z. B. dataset/test."""
    # Deine Struktur: dataset/images/test + dataset/labels/test
    if (data_dir / "images" / split).is_dir():
        return data_dir / "images" / split, data_dir / "labels" / split
    # Alternative: dataset/test/images + dataset/test/labels
    root = data_dir / split
    if (root / "images").is_dir():
        return root / "images", root / "labels"
    return root, root  # flache Struktur (Bilder + .txt im gleichen Ordner)


def make_yaml(images_dir: Path, names, out_path: Path) -> str:
    """Temporäre data.yaml, die den Testordner als 'test' (und Dummy val/train) definiert."""
    cfg = {
        "path": str(images_dir.parent.resolve()),
        "train": images_dir.name,
        "val": images_dir.name,
        "test": images_dir.name,
        "names": names,
    }
    out_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return str(out_path)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_noisy_dataset(images_dir: Path, labels_dir: Path, sigma: float, seed: int, root: Path) -> Path:
    """Kopiert Test-Set nach root/images + root/labels, Bilder mit Gauss-Rauschen (sigma in 0..255)."""
    rng = np.random.default_rng(seed)
    out_img, out_lbl = root / "images", root / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for p in image_files(images_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(out_img / (p.stem + ".png")), add_gaussian_noise(img, sigma, rng))  # PNG = verlustfrei
        lbl = labels_dir / (p.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, out_lbl / lbl.name)
    return out_img


def run_val(yolo: YOLO, data_yaml: str, args, device: str, name: str) -> dict:
    res = yolo.val(
        data=data_yaml,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
        project=str(Path(args.out).resolve()),
        name=name,
        exist_ok=True,
    )
    b = res.box
    p, r = float(b.mp), float(b.mr)
    metrics = {
        "precision": p,
        "recall": r,
        "f1": float(np.mean(b.f1)) if len(b.f1) else (2 * p * r / (p + r + 1e-12)),
        "mAP50": float(b.map50),
        "mAP75": float(b.map75),
        "mAP50-95": float(b.map),
    }
    per_class = {}
    for i, c in enumerate(b.ap_class_index):
        per_class[res.names[int(c)]] = float(b.maps[int(c)]) if len(b.maps) > int(c) else float(b.all_ap[i].mean())
    return {"metrics": metrics, "per_class": per_class}


# --------------------------------------------------------------------------- #
# Schwierigste Fälle (Loss pro Bild)
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def read_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.loadtxt(path, ndmin=2, dtype=np.float32)
    return arr[:, :5]  # cls cx cy w h (Detection-Format)


@torch.no_grad()
def per_image_losses(model: torch.nn.Module, images_dir: Path, labels_dir: Path, imgsz: int, device: str) -> pd.DataFrame:
    # Loss-Berechnung (TaskAlignedAssigner) läuft auf MPS teils instabil -> dann auf CPU
    loss_device = "cpu" if device == "mps" else ("cuda:0" if device == "0" else device)
    model = model.float().to(loss_device).eval()
    model.args = DEFAULT_CFG        # liefert die Loss-Gewichte (box=7.5, cls=0.5, dfl=1.5)
    model.criterion = None          # Criterion neu initialisieren

    rows = []
    for p in image_files(images_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        canvas, r, left, top = letterbox(img, imgsz)
        lab = read_labels(labels_dir / (p.stem + ".txt"))

        if len(lab):
            cx = (lab[:, 1] * w * r + left) / imgsz
            cy = (lab[:, 2] * h * r + top) / imgsz
            bw = lab[:, 3] * w * r / imgsz
            bh = lab[:, 4] * h * r / imgsz
            bboxes = torch.from_numpy(np.stack([cx, cy, bw, bh], 1)).float()
            cls = torch.from_numpy(lab[:, 0:1]).float()
        else:
            bboxes, cls = torch.zeros((0, 4)), torch.zeros((0, 1))

        tensor = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
        batch = {
            "img": tensor.to(loss_device),
            "batch_idx": torch.zeros(len(lab)).to(loss_device),
            "cls": cls.to(loss_device),
            "bboxes": bboxes.to(loss_device),
        }
        _, items = model.loss(batch)
        # Neuere ultralytics-Versionen liefern ein dict, ältere einen Tensor [box, cls, dfl]
        if isinstance(items, dict):
            comp = {k: float(torch.as_tensor(v).detach().float().sum().cpu()) for k, v in items.items()}
        else:
            vals = [float(x) for x in torch.as_tensor(items).detach().cpu().flatten()]
            comp = dict(zip(["box_loss", "cls_loss", "dfl_loss"], vals))
        pick = lambda key: sum(v for k, v in comp.items() if key in k.lower())
        rows.append({
            "image": str(p), "n_objects": len(lab),
            "loss_total": sum(comp.values()),
            "loss_box": pick("box"), "loss_cls": pick("cls"), "loss_dfl": pick("dfl"),
        })
    return pd.DataFrame(rows).sort_values("loss_total", ascending=False).reset_index(drop=True)


def visualize_case(yolo: YOLO, image_path: Path, labels_dir: Path, out_file: Path, imgsz: int, device: str):
    """Grün = Ground Truth, Rot = Vorhersage (conf >= 0.25)."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    names = yolo.names
    for c, cx, cy, bw, bh in read_labels(labels_dir / (image_path.stem + ".txt")):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"GT {names.get(int(c), int(c))}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    pred = yolo.predict(str(image_path), imgsz=imgsz, conf=0.25, device=device, verbose=False)[0]
    for box in pred.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(img, f"{names[int(box.cls)]} {float(box.conf):.2f}", (x1, min(y2 + 14, h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    cv2.imwrite(str(out_file), img)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="YOLO-Modell aus MLflow auf Test-Set evaluieren")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow_runs/mlflow.db")
    ap.add_argument("--weights",
                    default="mlruns/1/a0389a22e3df445e83d5f01ca1223081/artifacts/weights/best.pt",
                    help="Pfad zur best.pt im MLflow-Artefaktordner")
    ap.add_argument("--run-id", default=None,
                    help="Alternativ: Run-ID, best.pt wird dann unter --artifact-root gesucht")
    ap.add_argument("--artifact-root", default="mlruns")
    ap.add_argument("--data-dir", default="dataset", help="Ordner mit data.yaml und test/")
    ap.add_argument("--split", default="test", help="Name des Test-Ordners in --data-dir")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | 0")
    ap.add_argument("--noise-std", type=float, nargs="+", default=[25.0],
                    help="Rauschstärke(n): Std. des Gauss-Rauschens auf der 0-255-Skala (z. B. 10 25 50)")
    ap.add_argument("--top-k", type=int, default=10, help="Anzahl schwierigster Bilder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="evaluation_results")
    ap.add_argument("--log-mlflow", action="store_true", help="Ergebnisse zusätzlich im MLflow-Run loggen")
    args = ap.parse_args()

    device = pick_device(args.device)
    data_dir = Path(args.data_dir)

    # --- Modell aus MLflow laden --------------------------------------------
    mlflow.set_tracking_uri(args.tracking_uri)
    yolo, model_uri, run_id = load_yolo(args)
    print(f"Lade Gewichte: {model_uri} (Run: {run_id})")
    torch_model = copy.deepcopy(yolo.model)  # eigene Kopie für die Loss-Berechnung

    # Ergebnisse pro Run getrennt ablegen: <out>/<run_id>/
    args.out = str(Path(args.out) / (run_id or "unknown_run"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Klassen: {yolo.names}")

    # --- Test-Set vorbereiten -----------------------------------------------
    with open(data_dir / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    images_dir, labels_dir = locate_split(data_dir, args.split)
    if not images_dir.is_dir():
        raise SystemExit(f"Test-Ordner nicht gefunden: {images_dir}")
    print(f"Test-Bilder: {len(image_files(images_dir))} in {images_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix="yolo_eval_"))

    # --- 1) Performance (sauber) ----------------------------------------------
    print("\n=== 1) Performance auf sauberem Test-Set ===")
    clean_yaml = make_yaml(images_dir, names, tmp_root / "clean.yaml")
    clean = run_val(yolo, clean_yaml, args, device, "clean")
    print(pd.Series(clean["metrics"]).round(4).to_string())
    print("\nAP50-95 pro Klasse:")
    print(pd.Series(clean["per_class"]).round(4).to_string())

    # --- 2) Schwierigste Fälle ------------------------------------------------
    print(f"\n=== 2) Die {args.top_k} schwierigsten Bilder (höchster Loss) ===")
    df_loss = per_image_losses(torch_model, images_dir, labels_dir, args.imgsz, device)
    df_loss.to_csv(out_dir / "loss_per_image.csv", index=False)
    hardest = df_loss.head(args.top_k).copy()
    hardest.insert(0, "rank", range(1, len(hardest) + 1))
    hardest["visualization"] = [f"{r:02d}_{Path(p).stem}.jpg" for r, p in zip(hardest["rank"], hardest["image"])]
    print(hardest.drop(columns="visualization")
          .assign(image=hardest["image"].map(lambda s: Path(s).name)).round(4).to_string(index=False))
    hard_dir = out_dir / "hardest_cases"
    hard_dir.mkdir(exist_ok=True)
    for rank, row in enumerate(hardest.itertuples(), 1):
        visualize_case(yolo, Path(row.image), labels_dir, hard_dir / f"{rank:02d}_{Path(row.image).stem}.jpg",
                       args.imgsz, device)
    print(f"Visualisierungen (grün = GT, rot = Vorhersage): {hard_dir}")

    # --- 3) Robustheit --------------------------------------------------------
    print("\n=== 3) Robustheit gegen Gauss-Rauschen ===")
    noisy_results = {}
    for sigma in args.noise_std:
        root = tmp_root / f"noisy_{sigma:g}" / "test"
        noisy_img_dir = build_noisy_dataset(images_dir, labels_dir, sigma, args.seed, root)
        noisy_yaml = make_yaml(noisy_img_dir, names, tmp_root / f"noisy_{sigma:g}.yaml")
        noisy_results[sigma] = run_val(yolo, noisy_yaml, args, device, f"noisy_{sigma:g}")
        # Beispielbild speichern
        first = next(iter(sorted(noisy_img_dir.iterdir())))
        shutil.copy(first, out_dir / f"noise_example_sigma{sigma:g}.png")

    table = pd.DataFrame({"clean": clean["metrics"]})
    for sigma, res in noisy_results.items():
        table[f"noise_{sigma:g}"] = pd.Series(res["metrics"])
        table[f"Δabs_{sigma:g}"] = table[f"noise_{sigma:g}"] - table["clean"]
        table[f"Δrel_{sigma:g} [%]"] = 100 * table[f"Δabs_{sigma:g}"] / table["clean"].replace(0, np.nan)
    print(table.round(4).to_string())

    # --- Speichern --------------------------------------------------------------
    table.to_csv(out_dir / "robustness_comparison.csv")
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "model_uri": model_uri,
            "run_id": run_id,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_test_images": len(image_files(images_dir)),
            "imgsz": args.imgsz,
            "clean": clean,
            "noisy": {str(k): v for k, v in noisy_results.items()},
            "hardest_cases": hardest.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nAlle Ergebnisse gespeichert in: {out_dir.resolve()}")

    if args.log_mlflow and run_id:
        with mlflow.start_run(run_id=run_id):
            for k, v in clean["metrics"].items():
                mlflow.log_metric(f"test_{k}", v)
            for sigma, res in noisy_results.items():
                for k, v in res["metrics"].items():
                    mlflow.log_metric(f"test_noise{sigma:g}_{k}", v)
            mlflow.log_artifacts(str(out_dir), artifact_path="evaluation")
        print("Ergebnisse in MLflow geloggt.")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()