from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .device import resolve_device


CLASSIFIER_TYPE_MAP = {
    "door_scratch": "rayure_probable",
    "bumper_scratch": "rayure_probable",
    "door_dent": "bosse_probable",
    "bumper_dent": "bosse_probable",
    "glass_shatter": "verre_brise_probable",
    "head_lamp": "feu_casse_probable",
    "tail_lamp": "feu_casse_probable",
}


def normalize_classifier_type(class_name: str) -> str | None:
    return CLASSIFIER_TYPE_MAP.get(class_name.strip().lower())


def should_apply_classifier(prediction: dict, has_segmented_type: bool, threshold: float = 0.56) -> bool:
    """Autorise uniquement une typologie de complément, jamais un remplacement."""
    return bool(
        prediction.get("damage_type")
        and not has_segmented_type
        and float(prediction.get("confidence", 0.0)) >= threshold
    )


class OptionalYOLOClassifier:
    """Classificateur facultatif appliqué uniquement aux patchs candidats."""

    def __init__(self, weights: str | Path | None, device: str = "cpu", imgsz: int = 320):
        self.enabled = False
        self.model: Any = None
        self.error: str | None = None
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)
        self.weights = str(weights) if weights else None
        if not weights:
            return
        path = Path(weights)
        if not path.exists():
            self.error = f"Poids du classificateur introuvables : {path}"
            return
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2]))
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(path))
            self.enabled = True
        except Exception as exc:  # pragma: no cover - dépendance optionnelle
            self.error = f"Échec du chargement du classificateur : {exc}"

    def predict(self, image: np.ndarray) -> dict | None:
        if not self.enabled or image.size == 0:
            return None
        results = self.model.predict(
            source=image, device=self.device, imgsz=self.imgsz, verbose=False
        )
        if not results or results[0].probs is None:
            return None
        result = results[0]
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf.detach().cpu())
        names = result.names
        class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        return {
            "class_id": class_id,
            "class_name": class_name,
            "damage_type": normalize_classifier_type(class_name),
            "confidence": round(confidence, 5),
        }
