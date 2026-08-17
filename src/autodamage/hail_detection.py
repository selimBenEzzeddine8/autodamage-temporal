from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .io_utils import ensure_bgr_uint8


@dataclass(slots=True)
class HailDetectionResult:
    probable: bool
    confidence: float
    micro_impact_count: int
    cluster_count: int
    mask: np.ndarray
    score_map: np.ndarray
    diagnostics: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("mask", None)
        data.pop("score_map", None)
        return data


def _norm_robust(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sample = values[valid]
    if sample.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    median = float(np.median(sample))
    positive = sample[sample > median + 1e-6]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.percentile(positive, 92.0)) - median, 1e-5)
    return np.clip((values - median) / scale, 0.0, 1.0).astype(np.float32)


def detect_hail_damage(
    before: np.ndarray,
    after_aligned: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    minimum_impacts: int = 4,
) -> HailDetectionResult:
    """Détecte des groupes de petites déformations lisses apparues entre deux prises de vue.

    Ce détecteur n'essaie pas de conclure à partir d'une image isolée. Il exige plusieurs
    micro-impacts nouveaux et spatialement cohérents, ce qui limite les faux positifs liés
    aux reflets, aux joints de carrosserie et au bruit de compression.
    """
    before = ensure_bgr_uint8(before)
    after_aligned = ensure_bgr_uint8(after_aligned)
    if before.shape != after_aligned.shape:
        raise ValueError("Les images de détection de grêle doivent avoir la même taille.")
    h, w = before.shape[:2]
    valid = np.ones((h, w), dtype=bool) if valid_mask is None else valid_mask > 0
    valid = cv2.erode(valid.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), 1) > 0

    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_a = cv2.cvtColor(after_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    scale_responses: list[np.ndarray] = []
    for sigma in (1.0, 1.8, 3.0, 4.5):
        high_b = gray_b - cv2.GaussianBlur(gray_b, (0, 0), sigma)
        high_a = gray_a - cv2.GaussianBlur(gray_a, (0, 0), sigma)
        scale_responses.append(np.abs(high_a - high_b))
    relief_delta = np.maximum.reduce(scale_responses)

    # Les arêtes déjà présentes (joints, bords de portes, lettrage) ne sont pas des impacts.
    before_u8 = np.clip(gray_b * 255, 0, 255).astype(np.uint8)
    structural_edges = cv2.Canny(before_u8, 55, 135)
    structural_edges = cv2.dilate(structural_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0

    score = _norm_robust(relief_delta, valid)
    score[~valid] = 0.0
    score[structural_edges] *= 0.18
    sample = score[valid]
    threshold = max(0.20, min(0.82, float(np.percentile(sample, 98.7)) if sample.size else 1.0))
    local_max = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    peaks = ((score >= threshold) & (score >= local_max - 1e-6)).astype(np.uint8) * 255
    peaks = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, 8)
    impact_mask = np.zeros((h, w), np.uint8)
    centers: list[tuple[int, int]] = []
    strengths: list[float] = []
    for label in range(1, num):
        x, y, rw, rh, area = [int(v) for v in stats[label]]
        aspect = max(rw / max(rh, 1), rh / max(rw, 1))
        # Une petite bosse crée souvent un disque ou un anneau un peu plus large que son
        # centre visuel, surtout sur une image haute résolution.
        if not 3 <= area <= max(900, int(0.005 * h * w)) or aspect > 3.2:
            continue
        component = labels == label
        strength = float(score[component].mean())
        if strength < threshold:
            continue
        cx, cy = centroids[label]
        centers.append((int(round(cx)), int(round(cy))))
        strengths.append(strength)
        radius = max(3, int(round(0.65 * max(rw, rh))))
        cv2.circle(impact_mask, centers[-1], radius, 255, -1)

    # Un impact isolé est trop ambigu. On ne conserve que les groupes cohérents.
    cluster_seed = cv2.dilate(impact_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    cluster_num, cluster_labels = cv2.connectedComponents((cluster_seed > 0).astype(np.uint8), 8)
    cluster_mask = np.zeros_like(impact_mask)
    kept_impacts = 0
    kept_clusters = 0
    kept_strengths: list[float] = []
    for cluster_id in range(1, cluster_num):
        member_indices = [i for i, (cx, cy) in enumerate(centers) if cluster_labels[cy, cx] == cluster_id]
        if len(member_indices) < minimum_impacts:
            continue
        kept_clusters += 1
        kept_impacts += len(member_indices)
        kept_strengths.extend(strengths[i] for i in member_indices)
        for index in member_indices:
            cx, cy = centers[index]
            cv2.circle(cluster_mask, (cx, cy), 8, 255, -1)
    if kept_clusters:
        cluster_mask = cv2.dilate(cluster_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        cluster_mask[~valid] = 0

    density_score = float(np.clip((kept_impacts - minimum_impacts + 1) / 12.0, 0.0, 1.0))
    signal_score = float(np.mean(kept_strengths)) if kept_strengths else 0.0
    confidence = float(np.clip(0.58 * density_score + 0.42 * signal_score, 0.0, 1.0))
    probable = kept_clusters > 0 and kept_impacts >= minimum_impacts and confidence >= 0.30
    return HailDetectionResult(
        probable=probable,
        confidence=round(confidence, 4),
        micro_impact_count=kept_impacts,
        cluster_count=kept_clusters,
        mask=cluster_mask,
        score_map=score,
        diagnostics={
            "candidate_micro_impacts": len(centers),
            "threshold": round(threshold, 5),
            "valid_pixel_ratio": round(float(valid.mean()), 5),
            "minimum_impacts_per_cluster": minimum_impacts,
        },
    )
