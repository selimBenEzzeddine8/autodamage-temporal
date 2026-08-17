from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .io_utils import ensure_bgr_uint8


@dataclass(slots=True)
class ChangeRegion:
    id: int
    bbox_xyxy: list[int]
    area_px: int
    area_ratio: float
    centroid_xy: list[float]
    mean_change_score: float
    max_change_score: float
    confidence: float
    severity: str
    aspect_ratio: float
    damage_type: str = "changement_surface_probable"
    evidence: dict | None = None
    siamese_probability: float | None = None
    part_labels: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ChangeDetectionResult:
    ssim_global: float
    threshold: float
    mask: np.ndarray
    heatmap: np.ndarray
    score_map: np.ndarray
    regions: list[ChangeRegion]
    diagnostics: dict


def _norm01(x: np.ndarray, percentile: float = 99.5, floor_scale: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32)
    scale = max(float(np.percentile(x, percentile)) if x.size else 1.0, floor_scale)
    if scale <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / scale, 0.0, 1.0)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _scratch_response(gray: np.ndarray) -> np.ndarray:
    """Réponse multi-échelle aux lignes fines claires ou sombres."""
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    responses: list[np.ndarray] = []
    for size in (7, 13, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32))
        responses.append(cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
    return np.maximum.reduce(responses)


def compare_temporal_masks(before_mask: np.ndarray, after_mask_aligned: np.ndarray, dilation_px: int = 9) -> np.ndarray:
    before_mask = (before_mask > 0).astype(np.uint8) * 255
    after_mask_aligned = (after_mask_aligned > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
    known = cv2.dilate(before_mask, kernel, iterations=1)
    return cv2.bitwise_and(after_mask_aligned, cv2.bitwise_not(known))


def detect_changes(
    before: np.ndarray,
    normalized_after: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    min_region_area: int = 70,
    max_region_ratio: float = 0.12,
    sensitivity: float = 0.55,
    external_new_damage_mask: np.ndarray | None = None,
    hail_mask: np.ndarray | None = None,
) -> ChangeDetectionResult:
    before = ensure_bgr_uint8(before)
    normalized_after = ensure_bgr_uint8(normalized_after)
    if before.shape != normalized_after.shape:
        raise ValueError("Les images comparées doivent avoir la même taille.")
    h, w = before.shape[:2]
    valid = np.full((h, w), 255, np.uint8) if valid_mask is None else valid_mask.copy()
    valid = cv2.erode((valid > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8), iterations=1)
    valid_bool = valid > 0

    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(normalized_after, cv2.COLOR_BGR2GRAY)
    ssim_value, ssim_map = structural_similarity(gray_b, gray_a, data_range=255, full=True, gaussian_weights=True, sigma=1.2)
    ssim_change = np.clip(1.0 - ssim_map.astype(np.float32), 0, 1)

    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(normalized_after, cv2.COLOR_BGR2LAB).astype(np.float32)
    raw_lab_diff = np.mean(np.abs(lab_b - lab_a), axis=2)
    low_freq = cv2.GaussianBlur(raw_lab_diff, (0, 0), 13)
    local_color = np.clip(raw_lab_diff - 0.68 * low_freq, 0, None)
    local_color = _norm01(local_color, floor_scale=20.0)

    grad_b = _gradient_magnitude(gray_b)
    grad_a = _gradient_magnitude(gray_a)
    grad_delta = _norm01(np.abs(grad_b - grad_a), floor_scale=100.0)

    scratch_b = _scratch_response(gray_b)
    scratch_a = _scratch_response(gray_a)
    scratch_delta = _norm01(np.clip(scratch_a - scratch_b, 0, None), percentile=99.2, floor_scale=18.0)

    # Une différence persistante à plusieurs échelles est moins sensible au bruit et aux reflets ponctuels.
    ssim_small = cv2.GaussianBlur(ssim_change, (0, 0), 1.1)
    ssim_large = cv2.GaussianBlur(ssim_change, (0, 0), 3.0)
    persistence = np.minimum(np.clip(ssim_small * 2.2, 0, 1), np.clip(ssim_large * 2.2, 0, 1))

    # Les bords déjà présents dans l'image avant sont une source majeure de faux positifs
    # lorsqu'un recalage est imparfait. On favorise donc les arêtes réellement nouvelles.
    before_edges = cv2.Canny(gray_b, 70, 150)
    after_edges = cv2.Canny(gray_a, 70, 150)
    known_edges = cv2.dilate(before_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    novel_edges = cv2.bitwise_and(after_edges, cv2.bitwise_not(known_edges))
    novel_edges = cv2.dilate(novel_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    novel_support = cv2.GaussianBlur(novel_edges.astype(np.float32) / 255.0, (0, 0), 3.0)

    base_score = 0.24 * persistence + 0.22 * local_color + 0.28 * grad_delta + 0.26 * scratch_delta
    high_frequency_gate = np.clip(2.5 * grad_delta + 0.7 * local_color, 0, 1)
    structural_support = np.maximum.reduce((novel_support, np.clip(local_color * 0.75, 0, 1), scratch_delta))
    score = base_score * (0.28 + 0.72 * structural_support) * (0.42 + 0.58 * np.maximum(high_frequency_gate, scratch_delta))

    if external_new_damage_mask is not None:
        ext = cv2.resize((external_new_damage_mask > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        score = np.clip(score + 0.22 * ext, 0, 1)
    hail_support = np.zeros((h, w), dtype=np.float32)
    if hail_mask is not None:
        hail_support = cv2.resize((hail_mask > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        score = np.clip(score + 0.30 * hail_support, 0, 1)

    score[~valid_bool] = 0.0
    values = score[valid_bool]
    if values.size:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) + 1e-6
        robust_t = median + (4.8 - 2.0 * sensitivity) * 1.4826 * mad
        percentile_t = float(np.percentile(values, 99.0 - 2.0 * sensitivity))
        threshold = max(0.13 - 0.035 * sensitivity, min(0.38, max(robust_t, 0.70 * percentile_t)))
    else:
        threshold = 1.0

    raw_mask = (score >= threshold).astype(np.uint8) * 255
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    raw_mask = cv2.bitwise_and(raw_mask, valid)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    regions: list[ChangeRegion] = []
    clean_mask = np.zeros_like(raw_mask)
    image_area = h * w
    next_id = 1
    for label in range(1, num):
        x, y, rw, rh, area = [int(v) for v in stats[label]]
        area_ratio = area / image_area
        aspect = max(rw / max(rh, 1), rh / max(rw, 1))
        if area < min_region_area or area_ratio > max_region_ratio:
            continue
        component = labels == label
        scratch_mean = float(scratch_delta[component].mean())
        hail_overlap = float(hail_support[component].mean())
        if min(rw, rh) < 10 and area < 300 and scratch_mean < 0.28:
            continue
        if aspect > 22.0 and area_ratio < 0.005 and scratch_mean < 0.30:
            continue
        if (x <= 2 or y <= 2 or x + rw >= w - 2 or y + rh >= h - 2) and area_ratio < 0.01:
            continue
        component_scores = score[component]
        mean_score = float(component_scores.mean())
        max_score = float(component_scores.max())
        fill_ratio = area / max(rw * rh, 1)
        size_score = min(1.0, np.log1p(area / min_region_area) / np.log(35.0))
        signal_score = np.clip((mean_score - threshold) / max(1.0 - threshold, 1e-6) * 2.6 + 0.25, 0, 1)
        shape_bonus = 0.10 if aspect > 2.4 else 0.0
        diffuse_penalty = 0.18 if fill_ratio > 0.82 and area_ratio > 0.025 else 0.0
        confidence = float(np.clip(0.58 * signal_score + 0.30 * size_score + shape_bonus - diffuse_penalty, 0, 1))
        if confidence < 0.36:
            continue
        clean_mask[component] = 255
        severity = "forte" if area_ratio > 0.018 or confidence > 0.82 else "moyenne" if area_ratio > 0.004 or confidence > 0.55 else "faible"
        if hail_overlap >= 0.10:
            damage_type = "grêle_probable"
        elif scratch_mean >= 0.24 or aspect >= 2.6:
            damage_type = "rayure_probable"
        else:
            damage_type = "changement_surface_probable"
        cx, cy = centroids[label]
        regions.append(ChangeRegion(
            id=next_id,
            bbox_xyxy=[x, y, x + rw, y + rh],
            area_px=area,
            area_ratio=round(float(area_ratio), 6),
            centroid_xy=[round(float(cx), 2), round(float(cy), 2)],
            mean_change_score=round(mean_score, 4),
            max_change_score=round(max_score, 4),
            confidence=round(confidence, 4),
            severity=severity,
            aspect_ratio=round(float(aspect), 4),
            damage_type=damage_type,
            evidence={
                "scratch_score": round(scratch_mean, 4),
                "hail_overlap": round(hail_overlap, 4),
            },
        ))
        next_id += 1

    heat = np.clip(score * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heatmap[~valid_bool] = 0
    return ChangeDetectionResult(
        ssim_global=round(float(ssim_value), 6),
        threshold=round(float(threshold), 6),
        mask=clean_mask,
        heatmap=heatmap,
        score_map=score,
        regions=regions,
        diagnostics={
            "valid_pixel_ratio": round(float(valid_bool.mean()), 5),
            "raw_changed_pixel_ratio": round(float(np.mean(raw_mask > 0)), 6),
            "kept_changed_pixel_ratio": round(float(np.mean(clean_mask > 0)), 6),
            "sensitivity": sensitivity,
            "scratch_peak": round(float(scratch_delta.max()), 5),
            "hail_supported_pixel_ratio": round(float(np.mean(hail_support > 0)), 6),
        },
    )


def render_overlay(image: np.ndarray, mask: np.ndarray, regions: list[ChangeRegion], alpha: float = 0.38) -> np.ndarray:
    image = ensure_bgr_uint8(image)
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 1 - alpha, red, alpha, 0)[selected]
    for region in regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        readable_type = region.damage_type.replace("_probable", "").replace("_", " ")
        label = f"#{region.id} {readable_type} {region.confidence:.0%}"
        cv2.putText(overlay, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay
