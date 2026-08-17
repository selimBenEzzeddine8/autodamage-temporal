from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io_utils import encode_image, save_image


def write_result_directory(output_dir: str | Path, result: dict, images: dict[str, np.ndarray]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resultat.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, image in images.items():
        save_image(output_dir / f"{name}.png", image)
    return output_dir


def result_zip_bytes(result: dict, images: dict[str, np.ndarray]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("resultat.json", json.dumps(result, ensure_ascii=False, indent=2))
        for name, image in images.items():
            archive.writestr(f"images/{name}.png", encode_image(image, ".png"))
        archive.writestr(
            "README.txt",
            "Export généré par AutoDamage Temporal le "
            + datetime.now(timezone.utc).isoformat()
            + ".\nLe masque blanc et les boîtes rouges signalent les changements candidats, à faire valider par un humain.\n",
        )
    return payload.getvalue()
