from __future__ import annotations

import base64
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ImageDecodeError("Image absente ou invalide.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ImageDecodeError(f"Format d'image non pris en charge: {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 1.0
            if max_value <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ImageDecodeError("Le fichier image est vide.")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError("Impossible de décoder l'image. Utilisez JPEG, PNG ou WEBP.")
    return ensure_bgr_uint8(image)


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = path.read_bytes()
    return decode_image_bytes(data)


def read_upload(upload: BinaryIO) -> np.ndarray:
    return decode_image_bytes(upload.read())


def encode_image(image: np.ndarray, ext: str = ".png", quality: int = 95) -> bytes:
    image = ensure_bgr_uint8(image)
    params: list[int] = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError("Échec de l'encodage de l'image.")
    return buf.tobytes()


def image_to_data_uri(image: np.ndarray, ext: str = ".png") -> str:
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(encode_image(image, ext)).decode("ascii")


def save_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(image, path.suffix or ".png"))
    return path
