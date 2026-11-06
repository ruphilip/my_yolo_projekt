"""Erzeugt aus einem YOLO-Datensatz (jpg/jpeg) einen RGB-D-Datensatz mit 4 Kanälen.

Für jedes Bild wird mit yolo26-depth eine Tiefenkarte berechnet und als 4. Kanal
(B, G, R, D) in eine .tif-Datei geschrieben. Labels werden unverändert kopiert,
und eine neue data.yaml mit `channels: 4` wird erzeugt.

Beispiel:
    python make_rgbd_dataset.py --data dataset/data.yaml --out dataset_rgbd
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml

from rgbd_utils import DEPTH_MODEL, D_MAX, D_MIN, load_depth_model, predict_depth_u8, stack_rgbd

IMG_EXTS = {".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")


def _entries(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _strip_parent_refs(rel: Path) -> Path:
    return Path(*[p for p in rel.parts if p not in ("..", ".")])


def _labels_dir_for(images_dir: Path) -> Path | None:
    """Ultralytics ersetzt in Bildpfaden das letzte '/images/' durch '/labels/'."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return None


def build_rgbd_dataset(data_yaml, out_dir, depth_fn, overwrite=False, batch=8):
    """depth_fn(list[Path], list[(H, W)]) -> list[np.ndarray uint8 (H, W)]"""
    data_yaml, out_dir = Path(data_yaml).resolve(), Path(out_dir).resolve()
    cfg = yaml.safe_load(data_yaml.read_text())

    base = data_yaml.parent
    if cfg.get("path"):
        p = Path(cfg["path"])
        base = p if p.is_absolute() else (data_yaml.parent / p).resolve()

    new_cfg = dict(cfg)
    new_cfg["path"] = str(out_dir)
    new_cfg["channels"] = 4

    n_images = 0
    for split in SPLITS:
        new_entries = []
        for entry in _entries(cfg.get(split)):
            src_dir = (base / entry).resolve()
            if not src_dir.is_dir():
                raise SystemExit(
                    f"'{split}: {entry}' ist kein Ordner ({src_dir}). "
                    "Dateilisten (.txt) werden von diesem Skript nicht unterstützt."
                )
            rel = _strip_parent_refs(Path(entry))
            dst_dir = out_dir / rel
            dst_dir.mkdir(parents=True, exist_ok=True)
            new_entries.append(rel.as_posix())

            # Labels kopieren
            src_labels = _labels_dir_for(src_dir)
            dst_labels = _labels_dir_for(dst_dir)
            if src_labels is None or dst_labels is None:
                print(f"WARNUNG: Pfad '{entry}' enthält keinen 'images'-Ordner, Labels werden nicht gefunden.")
            elif src_labels.is_dir():
                shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)
            else:
                print(f"WARNUNG: Label-Ordner fehlt: {src_labels}")

            # Bilder -> RGB-D
            paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            seen, todo = set(), []
            for p in paths:
                if p.stem in seen:
                    print(f"WARNUNG: doppelter Dateiname '{p.stem}' (jpg/jpeg), übersprungen: {p.name}")
                    continue
                seen.add(p.stem)
                if overwrite or not (dst_dir / f"{p.stem}.tif").exists():
                    todo.append(p)

            print(f"[{split}] {src_dir}: {len(paths)} Bilder, {len(todo)} zu berechnen")
            for i in range(0, len(todo), batch):
                chunk = todo[i : i + batch]
                imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in chunk]
                ok = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
                for p, im in zip(chunk, imgs):
                    if im is None:
                        print(f"WARNUNG: nicht lesbar, übersprungen: {p}")
                if not ok:
                    continue
                depths = depth_fn([p for p, _ in ok], [im.shape[:2] for _, im in ok])
                for (p, im), d in zip(ok, depths):
                    cv2.imwrite(str(dst_dir / f"{p.stem}.tif"), stack_rgbd(im, d))
                if (i // batch) % 20 == 0:
                    print(f"   {min(i + batch, len(todo))}/{len(todo)}")
            n_images += len(paths)

        if new_entries:
            new_cfg[split] = new_entries[0] if len(new_entries) == 1 else new_entries

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(yaml.safe_dump(new_cfg, sort_keys=False, allow_unicode=True))
    return n_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--out", default="dataset_rgbd")
    ap.add_argument("--depth-model", default=DEPTH_MODEL, help="z.B. yolo26n-depth.pt ... yolo26x-depth.pt")
    ap.add_argument("--device", default="mps", help="mps, cpu oder cuda:0")
    ap.add_argument("--imgsz", type=int, default=None, help="Eingangsgrösse des Depth-Modells (Standard: Modell-Default)")
    ap.add_argument("--norm", choices=["log", "minmax"], default="log")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="vorhandene .tif neu berechnen")
    args = ap.parse_args()

    model = load_depth_model(args.depth_model)

    def depth_fn(paths, shapes):
        return predict_depth_u8(model, paths, device=args.device, imgsz=args.imgsz, norm=args.norm, shapes=shapes)

    n = build_rgbd_dataset(args.data, args.out, depth_fn, overwrite=args.overwrite, batch=args.batch)

    import ultralytics

    meta = {
        "depth_model": args.depth_model,
        "depth_norm": args.norm,
        "depth_range_m": [D_MIN, D_MAX],
        "depth_imgsz": args.imgsz,
        "ultralytics_version": ultralytics.__version__,
        "source_data_yaml": str(Path(args.data)),
        "n_images": n,
    }
    (Path(args.out) / "depth_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Fertig: {n} Bilder -> {args.out}/ (data.yaml, depth_meta.json)")


if __name__ == "__main__":
    main()
