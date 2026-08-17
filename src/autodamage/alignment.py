from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .io_utils import ensure_bgr_uint8


@dataclass(slots=True)
class AlignmentReport:
    method: str
    reliability_score: float
    reliability_label: str
    keypoints_before: int
    keypoints_after: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    coverage_ratio: float
    median_reprojection_error: float | None
    geometry_valid: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AlignmentResult:
    aligned_after: np.ndarray
    valid_mask: np.ndarray
    homography: np.ndarray
    report: AlignmentReport


def _preprocess_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if points is None or len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / max(width * height, 1))


def _geometry_valid(h: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[bool, float]:
    if h is None or not np.isfinite(h).all():
        return False, 0.0
    corners = np.float32([[[0, 0]], [[src_w - 1, 0]], [[src_w - 1, src_h - 1]], [[0, src_h - 1]]])
    warped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    ratio = area / max(dst_w * dst_h, 1)
    convex = cv2.isContourConvex(warped.astype(np.float32))
    finite = np.isfinite(warped).all()
    return bool(finite and convex and 0.18 <= ratio <= 3.5), float(ratio)


def _align_ecc_affine(
    before: np.ndarray,
    after: np.ndarray,
    *,
    reason: str,
    keypoints_before: int = 0,
    keypoints_after: int = 0,
    good_matches: int = 0,
    inliers: int = 0,
) -> AlignmentResult | None:
    """Repli photométrique pour les carrosseries lisses où ORB manque de texture."""
    hb, wb = before.shape[:2]
    ha, wa = after.shape[:2]
    resized = cv2.resize(after, (wb, hb), interpolation=cv2.INTER_AREA if wa > wb or ha > hb else cv2.INTER_CUBIC)
    template = _preprocess_gray(before).astype(np.float32) / 255.0
    moving = _preprocess_gray(resized).astype(np.float32) / 255.0
    template = cv2.GaussianBlur(template, (0, 0), 1.2)
    moving = cv2.GaussianBlur(moving, (0, 0), 1.2)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 180, 1e-6)
    try:
        correlation, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        return None

    if not np.isfinite(warp).all() or not np.isfinite(correlation):
        return None
    aligned = cv2.warpAffine(
        resized, warp, (wb, hb),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.warpAffine(
        np.full((hb, wb), 255, np.uint8), warp, (wb, hb),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8), iterations=1)

    inverse_affine = cv2.invertAffineTransform(warp).astype(np.float64)
    inverse_h = np.vstack([inverse_affine, [0.0, 0.0, 1.0]])
    resize_h = np.array([[wb / max(wa, 1), 0.0, 0.0], [0.0, hb / max(ha, 1), 0.0], [0.0, 0.0, 1.0]])
    homography = inverse_h @ resize_h
    geometry_ok, area_ratio = _geometry_valid(homography, wa, ha, wb, hb)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    after_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = valid > 0
    residual = float(np.median(np.abs(before_gray[mask] - after_gray[mask]))) if np.any(mask) else 255.0
    corr_score = float(np.clip((float(correlation) - 0.45) / 0.5, 0.0, 1.0))
    residual_score = float(np.exp(-residual / 28.0))
    overlap = float(np.mean(mask))
    reliability = float(np.clip(0.62 * corr_score + 0.23 * residual_score + 0.15 * overlap, 0.0, 1.0))
    warnings = [reason, "Recalage affine ECC utilisé sur une surface peu texturée."]
    if reliability < 0.35:
        warnings.append("Le recalage ECC reste peu fiable : une nouvelle prise de vue est recommandée.")
    if not geometry_ok:
        warnings.append(f"Géométrie ECC inhabituelle (ratio de surface {area_ratio:.2f}).")
        reliability *= 0.45
    label = "élevée" if reliability >= 0.72 else "moyenne" if reliability >= 0.42 else "faible"
    return AlignmentResult(
        aligned_after=aligned,
        valid_mask=valid,
        homography=homography,
        report=AlignmentReport(
            method="ECC_affine",
            reliability_score=round(reliability, 4),
            reliability_label=label,
            keypoints_before=keypoints_before,
            keypoints_after=keypoints_after,
            good_matches=good_matches,
            inliers=inliers,
            inlier_ratio=round(inliers / max(good_matches, 1), 4),
            coverage_ratio=round(overlap, 4),
            median_reprojection_error=None,
            geometry_valid=geometry_ok,
            warnings=warnings,
        ),
    )


def align_orb_ransac(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_features: int = 5000,
    ratio_test: float = 0.76,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 12,
) -> AlignmentResult:
    before = ensure_bgr_uint8(before)
    after = ensure_bgr_uint8(after)
    hb, wb = before.shape[:2]
    ha, wa = after.shape[:2]

    gray_b = _preprocess_gray(before)
    gray_a = _preprocess_gray(after)
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, edgeThreshold=19, fastThreshold=10)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    warnings: list[str] = []

    def fallback(reason: str, *, good_matches: int = 0, inliers: int = 0) -> AlignmentResult:
        ecc = _align_ecc_affine(
            before,
            after,
            reason=reason,
            keypoints_before=len(kp_b or []),
            keypoints_after=len(kp_a or []),
            good_matches=good_matches,
            inliers=inliers,
        )
        if ecc is not None and ecc.report.reliability_score >= 0.18:
            return ecc
        warnings.append(reason)
        resized = cv2.resize(after, (wb, hb), interpolation=cv2.INTER_AREA if wa > wb else cv2.INTER_CUBIC)
        valid = np.full((hb, wb), 255, np.uint8)
        sx, sy = wb / max(wa, 1), hb / max(ha, 1)
        hmat = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        report = AlignmentReport(
            method="resize_fallback",
            reliability_score=0.05,
            reliability_label="faible",
            keypoints_before=len(kp_b or []),
            keypoints_after=len(kp_a or []),
            good_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            coverage_ratio=0.0,
            median_reprojection_error=None,
            geometry_valid=False,
            warnings=warnings.copy(),
        )
        return AlignmentResult(resized, valid, hmat, report)

    if des_b is None or des_a is None or len(kp_b) < min_good_matches or len(kp_a) < min_good_matches:
        return fallback("Pas assez de points ORB pour estimer une homographie fiable.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_a, des_b, k=2)  # image retour -> image avant
    good = [m for pair in raw_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio_test * n.distance]
    if len(good) < min_good_matches:
        return fallback(f"Seulement {len(good)} correspondances ORB valides.", good_matches=len(good))

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    hmat, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold, maxIters=5000, confidence=0.995)
    if hmat is None or inlier_mask is None:
        return fallback("RANSAC n'a pas pu estimer l'homographie.", good_matches=len(good))

    inlier_mask_bool = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask_bool.sum())
    inlier_ratio = inliers / max(len(good), 1)
    coverage = _coverage(dst.reshape(-1, 2)[inlier_mask_bool], wb, hb)
    projected = cv2.perspectiveTransform(src[inlier_mask_bool], hmat)
    reproj = np.linalg.norm(projected.reshape(-1, 2) - dst[inlier_mask_bool].reshape(-1, 2), axis=1)
    median_error = float(np.median(reproj)) if reproj.size else 99.0
    geometry_ok, area_ratio = _geometry_valid(hmat, wa, ha, wb, hb)

    match_score = min(1.0, len(good) / 160.0)
    inlier_score = float(np.clip((inlier_ratio - 0.2) / 0.65, 0, 1))
    coverage_score = min(1.0, coverage / 0.28)
    reproj_score = float(np.exp(-median_error / 4.0))
    geometry_score = 1.0 if geometry_ok else 0.0
    reliability = 0.20 * match_score + 0.32 * inlier_score + 0.20 * coverage_score + 0.18 * reproj_score + 0.10 * geometry_score
    reliability = float(np.clip(reliability, 0, 1))

    if not geometry_ok or reliability < 0.25:
        return fallback(
            "Homographie ORB rejetée car elle est géométriquement invalide ou trop peu fiable.",
            good_matches=len(good),
            inliers=inliers,
        )

    aligned = cv2.warpPerspective(after, hmat, (wb, hb), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    source_valid = np.full((ha, wa), 255, dtype=np.uint8)
    valid = cv2.warpPerspective(source_valid, hmat, (wb, hb), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=1)

    if reliability < 0.35:
        warnings.append("Recalage peu fiable : interpréter les changements avec prudence.")
    if coverage < 0.06:
        warnings.append("Les correspondances couvrent une zone trop petite de l'image.")
    if not geometry_ok:
        warnings.append(f"Géométrie d'homographie inhabituelle (ratio de surface {area_ratio:.2f}).")

    label = "élevée" if reliability >= 0.72 else "moyenne" if reliability >= 0.42 else "faible"
    report = AlignmentReport(
        method="ORB+RANSAC",
        reliability_score=round(reliability, 4),
        reliability_label=label,
        keypoints_before=len(kp_b),
        keypoints_after=len(kp_a),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=round(float(inlier_ratio), 4),
        coverage_ratio=round(float(coverage), 4),
        median_reprojection_error=round(median_error, 4),
        geometry_valid=geometry_ok,
        warnings=warnings,
    )
    return AlignmentResult(aligned, valid, hmat, report)
