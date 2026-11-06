"""Gemeinsame Hilfsfunktionen für RGB-D (Bild + Tiefenkarte als 4. Kanal).

Dieselbe Normalisierung MUSS beim Training und bei der Inferenz verwendet werden,
deshalb liegt sie hier an genau einer Stelle.
"""

from __future__ import annotations

import cv2
import numpy as np

# yolo26-depth liefert metrische Tiefe in Metern (Arbeitsbereich grob 0.02 - 150 m).
# Log-Skalierung mit festen Grenzen: gleiche Distanz -> gleicher Grauwert in jedem Bild.
D_MIN = 0.1    # Meter  -> 0
D_MAX = 100.0  # Meter  -> 255

DEPTH_MODEL = "yolo26s-depth.pt"  # n / s / m / l / x verfügbar (benötigt ultralytics >= 8.4.104)


def depth_to_u8(depth_m: np.ndarray, norm: str = "log") -> np.ndarray:
    """Metrische Tiefenkarte (H, W, float, Meter) -> uint8 (H, W).

    norm="log"    : feste Log-Skala D_MIN..D_MAX (empfohlen, bildübergreifend konsistent)
    norm="minmax" : pro Bild auf 0..255 gestreckt (verliert die absolute Skala)
    """
    d = np.nan_to_num(depth_m.astype(np.float32), nan=D_MAX, posinf=D_MAX, neginf=D_MIN)
    if norm == "log":
        d = np.clip(d, D_MIN, D_MAX)
        d = (np.log(d) - np.log(D_MIN)) / (np.log(D_MAX) - np.log(D_MIN))
    elif norm == "minmax":
        lo, hi = float(d.min()), float(d.max())
        d = (d - lo) / (hi - lo + 1e-8)
    else:
        raise ValueError(f"Unbekannte Normalisierung: {norm}")
    return np.clip(d * 255.0 + 0.5, 0, 255).astype(np.uint8)


def load_depth_model(weights: str = DEPTH_MODEL):
    from ultralytics import YOLO  # lazy import, damit dieses Modul auch ohne torch importierbar ist

    return YOLO(weights)


def predict_depth_m(model, images, device="mps", imgsz=None, shapes=None):
    """Metrische Tiefenkarten (float32, Meter) berechnen.

    images: Liste von Bildpfaden ODER BGR-Arrays (H, W, 3).
    shapes: optionale Liste (H, W) der Originalbilder. Weicht die Tiefenkarte davon ab,
            wird sie per bilinearer Interpolation angepasst.
    """
    kwargs = dict(device=device, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz
    sources = [im if isinstance(im, np.ndarray) else str(im) for im in images]
    results = model.predict(sources, **kwargs)
    out = []
    for i, r in enumerate(results):
        depth = r.depth.data.cpu().numpy().astype(np.float32)
        if depth.ndim == 3:  # (1, H, W) -> (H, W)
            depth = depth[0]
        if shapes is not None and depth.shape != tuple(shapes[i]):
            h, w = shapes[i]
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(depth)
    return out


def predict_depth_u8(model, image_paths, device="mps", imgsz=None, norm="log", shapes=None):
    """Tiefenkarten (uint8) für eine Liste von Bildpfaden berechnen (siehe predict_depth_m)."""
    maps = predict_depth_m(model, image_paths, device=device, imgsz=imgsz, shapes=shapes)
    return [depth_to_u8(d, norm=norm) for d in maps]


def stack_rgbd(img_bgr: np.ndarray, depth_u8: np.ndarray) -> np.ndarray:
    """BGR (H, W, 3) + Tiefe (H, W) -> (H, W, 4) mit Kanalreihenfolge B, G, R, D."""
    assert img_bgr.shape[:2] == depth_u8.shape, (img_bgr.shape, depth_u8.shape)
    return np.dstack([img_bgr, depth_u8])
