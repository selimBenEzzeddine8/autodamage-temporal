from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .io_utils import ensure_bgr_uint8


@dataclass(slots=True)
class QualityReport:
    width: int
    height: int
    blur_variance: float
    mean_brightness: float
    contrast_std: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    sharpness_score: float
    exposure_score: float
    contrast_score: float
    resolution_score: float
    overall_score: float
    status: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def assess_image_quality(image: np.ndarray) -> QualityReport:
    image = ensure_bgr_uint8(image)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    contrast_std = float(gray.std())
    dark_clip_ratio = float(np.mean(gray <= 8))
    bright_clip_ratio = float(np.mean(gray >= 247))

    sharpness_score = _clip01((blur_variance - 20.0) / 180.0)
    centered_exposure = 1.0 - abs(mean_brightness - 127.5) / 127.5
    clipping_penalty = min(1.0, 4.0 * (dark_clip_ratio + bright_clip_ratio))
    exposure_score = _clip01(centered_exposure * (1.0 - clipping_penalty))
    contrast_score = _clip01((contrast_std - 12.0) / 45.0)
    resolution_score = _clip01(min(h, w) / 720.0)

    overall = (
        0.35 * sharpness_score
        + 0.30 * exposure_score
        + 0.20 * contrast_score
        + 0.15 * resolution_score
    )
    warnings: list[str] = []
    if min(h, w) < 480:
        warnings.append("Résolution faible : utilisez idéalement au moins 720 px sur le petit côté.")
    if blur_variance < 45:
        warnings.append("Image probablement floue.")
    if mean_brightness < 45:
        warnings.append("Image très sombre.")
    elif mean_brightness > 215:
        warnings.append("Image très claire.")
    if dark_clip_ratio > 0.18:
        warnings.append("Zones noires bouchées importantes.")
    if bright_clip_ratio > 0.12:
        warnings.append("Hautes lumières surexposées importantes.")
    if contrast_std < 20:
        warnings.append("Contraste faible.")

    status = "bon" if overall >= 0.62 else "acceptable" if overall >= 0.42 else "insuffisant"
    return QualityReport(
        width=w,
        height=h,
        blur_variance=round(blur_variance, 3),
        mean_brightness=round(mean_brightness, 3),
        contrast_std=round(contrast_std, 3),
        dark_clip_ratio=round(dark_clip_ratio, 6),
        bright_clip_ratio=round(bright_clip_ratio, 6),
        sharpness_score=round(sharpness_score, 4),
        exposure_score=round(exposure_score, 4),
        contrast_score=round(contrast_score, 4),
        resolution_score=round(resolution_score, 4),
        overall_score=round(float(overall), 4),
        status=status,
        warnings=warnings,
    )


def compare_capture_compatibility(before: np.ndarray, after: np.ndarray) -> dict:
    before = ensure_bgr_uint8(before)
    after = ensure_bgr_uint8(after)
    hb, wb = before.shape[:2]
    ha, wa = after.shape[:2]
    aspect_before = wb / hb
    aspect_after = wa / ha
    aspect_delta = abs(aspect_before - aspect_after) / max(aspect_before, 1e-6)
    resolution_ratio = min(wb * hb, wa * ha) / max(wb * hb, wa * ha)
    warnings: list[str] = []
    if aspect_delta > 0.15:
        warnings.append("Les ratios d'image diffèrent fortement; le recalage peut être instable.")
    if resolution_ratio < 0.35:
        warnings.append("Les résolutions sont très différentes.")
    return {
        "aspect_ratio_delta": round(float(aspect_delta), 5),
        "resolution_ratio": round(float(resolution_ratio), 5),
        "compatible": aspect_delta <= 0.25 and resolution_ratio >= 0.2,
        "warnings": warnings,
    }
