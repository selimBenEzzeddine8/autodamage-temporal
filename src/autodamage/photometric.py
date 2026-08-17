from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .io_utils import ensure_bgr_uint8


@dataclass(slots=True)
class PhotometricReport:
    method: str
    channels: list[dict]
    stable_pixel_ratio: float
    before_after_mae_before: float
    before_after_mae_after: float

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_channel_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    valid_src = src[mask]
    valid_ref = ref[mask]
    if valid_src.size < 100:
        return src.copy(), {"scale": 1.0, "offset": 0.0, "samples": int(valid_src.size)}
    qs = np.percentile(valid_src, [10, 50, 90])
    qr = np.percentile(valid_ref, [10, 50, 90])
    spread_s = max(float(qs[2] - qs[0]), 8.0)
    spread_r = max(float(qr[2] - qr[0]), 8.0)
    scale = float(np.clip(spread_r / spread_s, 0.55, 1.85))
    offset = float(qr[1] - scale * qs[1])
    corrected = np.clip(src.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
    return corrected, {"scale": round(scale, 5), "offset": round(offset, 5), "samples": int(valid_src.size)}


def normalize_photometry(before: np.ndarray, aligned_after: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, PhotometricReport]:
    before = ensure_bgr_uint8(before)
    aligned_after = ensure_bgr_uint8(aligned_after)
    if before.shape != aligned_after.shape:
        raise ValueError("Les images doivent avoir la même taille après recalage.")
    h, w = before.shape[:2]
    if valid_mask is None:
        valid_mask = np.full((h, w), 255, np.uint8)
    valid = valid_mask > 0
    raw_mae = float(np.mean(cv2.absdiff(before, aligned_after)[valid])) if np.any(valid) else 0.0
    if raw_mae < 0.5:
        report = PhotometricReport(
            method="identité (images déjà photométriquement compatibles)",
            channels=[
                {"channel": name, "scale": 1.0, "offset": 0.0, "samples": int(valid.sum())}
                for name in ["L", "a", "b"]
            ],
            stable_pixel_ratio=round(float(valid.mean()), 5),
            before_after_mae_before=round(raw_mae, 4),
            before_after_mae_after=round(raw_mae, 4),
        )
        return aligned_after.copy(), report

    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY)
    coarse_diff = cv2.absdiff(cv2.GaussianBlur(gray_b, (0, 0), 11), cv2.GaussianBlur(gray_a, (0, 0), 11))
    threshold = float(np.percentile(coarse_diff[valid], 82)) if np.any(valid) else 255.0
    stable = valid & (coarse_diff <= max(18.0, threshold))
    if stable.mean() < 0.15:
        stable = valid

    corrected = lab_a.copy()
    low_before = cv2.GaussianBlur(lab_b[:, :, 0], (0, 0), 11)
    low_after = cv2.GaussianBlur(lab_a[:, :, 0], (0, 0), 11)
    luminance_delta = np.clip(low_before - low_after, -55.0, 55.0)
    corrected[:, :, 0] = np.clip(lab_a[:, :, 0] + luminance_delta, 0, 255)
    stats: list[dict] = [{
        "channel": "L",
        "scale": 1.0,
        "offset": round(float(np.median(luminance_delta[stable])) if np.any(stable) else 0.0, 5),
        "local_correction_sigma": 11,
        "samples": int(stable.sum()),
    }]
    for idx, name in [(1, "a"), (2, "b")]:
        offset = float(np.median(lab_b[:, :, idx][stable] - lab_a[:, :, idx][stable])) if np.any(stable) else 0.0
        offset = float(np.clip(offset, -35.0, 35.0))
        corrected[:, :, idx] = np.clip(lab_a[:, :, idx] + offset, 0, 255)
        stats.append({"channel": name, "scale": 1.0, "offset": round(offset, 5), "samples": int(stable.sum())})

    candidate = cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)
    candidate_mae = float(np.mean(cv2.absdiff(before, candidate)[valid])) if np.any(valid) else raw_mae
    if candidate_mae > raw_mae * 1.03:
        normalized = aligned_after.copy()
        candidate_mae = raw_mae
        method = "identité (la correction photométrique n'améliorait pas l'écart robuste)"
        stats = [{"channel": name, "scale": 1.0, "offset": 0.0, "samples": int(stable.sum())} for name in ["L", "a", "b"]]
    else:
        normalized = candidate
        method = "correction locale d'éclairage L + recentrage robuste des chromas LAB"

    report = PhotometricReport(
        method=method,
        channels=stats,
        stable_pixel_ratio=round(float(stable.mean()), 5),
        before_after_mae_before=round(raw_mae, 4),
        before_after_mae_after=round(candidate_mae, 4),
    )
    return normalized, report
