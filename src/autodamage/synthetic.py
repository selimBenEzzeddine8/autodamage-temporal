from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import save_image


def _car_scene(seed: int = 7, size: tuple[int, int] = (960, 540)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = size
    image = np.full((h, w, 3), (165, 175, 185), np.uint8)
    # Sol et fond texturés pour offrir des points ORB stables.
    image[:330] = (195, 205, 212)
    image[330:] = (78, 82, 84)
    for _ in range(450):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        value = int(rng.integers(-18, 19))
        image[y, x] = np.clip(image[y, x].astype(int) + value, 0, 255)
    for x in range(0, w, 80):
        cv2.line(image, (x, 0), (x + 30, 300), (180, 190, 198), 1)
    cv2.rectangle(image, (40, 35), (190, 105), (238, 238, 235), -1)
    cv2.putText(image, "RENTAL 24", (58, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)

    # Silhouette du véhicule.
    body = np.array([[150, 310], [220, 225], [385, 185], [650, 192], [775, 255], [835, 320], [800, 390], [180, 390]], np.int32)
    cv2.fillPoly(image, [body], (47, 118, 188))
    cv2.polylines(image, [body], True, (20, 55, 88), 4, cv2.LINE_AA)
    windows = np.array([[285, 230], [400, 202], [610, 205], [704, 252]], np.int32)
    cv2.fillPoly(image, [windows], (55, 75, 85))
    cv2.line(image, (500, 205), (500, 322), (20, 55, 88), 3)
    cv2.line(image, (250, 318), (775, 318), (26, 76, 120), 2)
    cv2.line(image, (405, 205), (385, 318), (20, 55, 88), 3)
    cv2.line(image, (610, 207), (665, 318), (20, 55, 88), 3)
    cv2.rectangle(image, (435, 278), (475, 288), (18, 44, 70), 2)
    cv2.rectangle(image, (625, 278), (665, 288), (18, 44, 70), 2)
    cv2.ellipse(image, (190, 326), (36, 18), 0, 0, 360, (235, 244, 250), -1)
    cv2.ellipse(image, (802, 326), (30, 15), 0, 0, 360, (220, 60, 45), -1)
    for center in [(285, 388), (710, 388)]:
        cv2.circle(image, center, 60, (30, 32, 36), -1)
        cv2.circle(image, center, 32, (145, 150, 155), -1)
        cv2.circle(image, center, 10, (45, 48, 50), -1)
        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            end = (int(center[0] + 27 * np.cos(rad)), int(center[1] + 27 * np.sin(rad)))
            cv2.line(image, center, end, (70, 75, 80), 3)
    # Texture discrète sur la carrosserie.
    for _ in range(100):
        x = int(rng.integers(190, 800)); y = int(rng.integers(230, 375))
        if image[y, x, 0] > 40 and image[y, x, 2] > 120:
            cv2.circle(image, (x, y), 1, (55, 125, 195), -1)
    return cv2.GaussianBlur(image, (3, 3), 0.35)


def generate_synthetic_pair(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    before = _car_scene(seed)
    damaged = before.copy()
    gt = np.zeros(before.shape[:2], np.uint8)

    # Rayure nouvelle et petit enfoncement sur la portière arrière.
    scratch_pts = np.array([[585, 274], [620, 283], [658, 277], [700, 291]], np.int32)
    cv2.polylines(damaged, [scratch_pts], False, (18, 28, 35), 6, cv2.LINE_AA)
    cv2.polylines(damaged, [scratch_pts], False, (220, 226, 230), 2, cv2.LINE_AA)
    cv2.polylines(gt, [scratch_pts], False, 255, 14, cv2.LINE_AA)
    cv2.ellipse(damaged, (650, 344), (30, 16), -8, 0, 360, (28, 75, 120), 4, cv2.LINE_AA)
    cv2.ellipse(gt, (650, 344), (36, 22), -8, 0, 360, 255, -1, cv2.LINE_AA)

    h, w = before.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[18, 13], [w - 28, 2], [w - 8, h - 18], [32, h - 2]])
    homography = cv2.getPerspectiveTransform(src, dst)
    after = cv2.warpPerspective(damaged, homography, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Changement photométrique global.
    after_f = after.astype(np.float32)
    after_f[:, :, 0] *= 1.07
    after_f[:, :, 1] *= 0.96
    after_f[:, :, 2] *= 0.90
    after_f = after_f * 0.91 + 13
    after = np.clip(after_f, 0, 255).astype(np.uint8)

    # Ombre douce et reflet spéculaire : ils ne doivent idéalement pas devenir des dégâts.
    shadow = np.zeros((h, w), np.float32)
    cv2.ellipse(shadow, (430, 295), (260, 75), 8, 0, 360, 0.28, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 35)
    after = np.clip(after.astype(np.float32) * (1.0 - shadow[..., None]), 0, 255).astype(np.uint8)
    reflection = np.zeros((h, w), np.uint8)
    cv2.line(reflection, (330, 235), (705, 254), 200, 12, cv2.LINE_AA)
    reflection = cv2.GaussianBlur(reflection, (0, 0), 7)
    after = np.clip(after.astype(np.int16) + reflection[..., None] // 3, 0, 255).astype(np.uint8)

    noise = rng.normal(0, 2.2, after.shape).astype(np.float32)
    after = np.clip(after.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    metadata = {
        "seed": seed,
        "description": "Perspective, lumière, ombre et reflet changés; rayure et enfoncement ajoutés.",
        "after_to_before_homography": np.linalg.inv(homography).round(8).tolist(),
    }
    return before, after, gt, metadata


def write_synthetic_examples(directory: str | Path, seed: int = 7) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before, after, gt, _ = generate_synthetic_pair(seed)
    return {
        "before": save_image(directory / "avant_synthetique.png", before),
        "after": save_image(directory / "apres_synthetique.png", after),
        "ground_truth": save_image(directory / "masque_verite_terrain.png", gt),
    }
