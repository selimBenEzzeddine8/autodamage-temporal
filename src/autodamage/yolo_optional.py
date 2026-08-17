from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .device import resolve_device


class OptionalYOLOSegmenter:
    """Charge un modèle Ultralytics de segmentation uniquement si des poids sont fournis."""

    def __init__(
        self,
        weights: str | Path | None,
        confidence: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        self.enabled = False
        self.model: Any = None
        self.error: str | None = None
        self.confidence = confidence
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)
        self.weights = str(weights) if weights else None
        if not weights:
            return
        path = Path(weights)
        if not path.exists():
            self.error = f"Poids YOLO introuvables: {path}"
            return
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2]))
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optionnel
            self.error = f"Ultralytics indisponible: {exc}"
            return
        try:
            self.model = YOLO(str(path))
            self.enabled = True
        except Exception as exc:  # pragma: no cover
            self.error = f"Échec du chargement YOLO: {exc}"

    def predict_masks(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        h, w = image.shape[:2]
        union = np.zeros((h, w), np.uint8)
        instances: list[dict] = []
        if not self.enabled:
            return union, instances
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return union, instances
        result = results[0]
        if result.masks is None:
            return union, instances
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.zeros(len(masks), dtype=int)
        confs = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(masks))
        names = result.names
        for idx, mask in enumerate(masks):
            resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
            union[resized] = 255
            ys, xs = np.where(resized)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else [0, 0, 0, 0]
            class_id = int(classes[idx])
            instances.append({
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id]),
                "confidence": round(float(confs[idx]), 5),
                "area_px": int(resized.sum()),
                "bbox_xyxy": bbox,
            })
        return union, instances
