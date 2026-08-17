from __future__ import annotations

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .device import resolve_device
from .exporting import result_zip_bytes
from .io_utils import ImageDecodeError, decode_image_bytes, encode_image
from .pipeline import DamageDetectionPipeline, PipelineConfig
from .synthetic import generate_synthetic_pair

app = FastAPI(
    title="AutoDamage Temporal API",
    version="4.0.0",
    description="Compare deux photographies d'un véhicule et détecte les nouveaux dégâts candidats.",
)


@lru_cache(maxsize=8)
def _pipeline(sensitivity: float = 0.55) -> DamageDetectionPipeline:
    default_damage = Path("models/car_damage_v2.pt")
    default_types = Path("models/car_damage_types_v2.pt")
    default_classifier = Path("models/car_damage_classifier_v3.pt")
    default_temporal = Path("models/temporal_damage_v4.pt")
    damage_weights = os.getenv("YOLO_DAMAGE_WEIGHTS") or (str(default_damage) if default_damage.exists() else None)
    type_weights = os.getenv("YOLO_DAMAGE_TYPE_WEIGHTS") or (str(default_types) if default_types.exists() else None)
    classifier_weights = os.getenv("YOLO_DAMAGE_CLASSIFIER_WEIGHTS") or (
        str(default_classifier) if default_classifier.exists() else None
    )
    temporal_weights = os.getenv("TEMPORAL_DAMAGE_WEIGHTS") or (
        str(default_temporal) if default_temporal.exists() else None
    )
    enable_yolo_setting = os.getenv("ENABLE_YOLO")
    enable_yolo = (
        enable_yolo_setting == "1"
        if enable_yolo_setting is not None
        else bool(damage_weights or type_weights or classifier_weights)
    )
    config = PipelineConfig(
        sensitivity=float(max(0.0, min(1.0, sensitivity))),
        enable_yolo=enable_yolo,
        enable_siamese=os.getenv("ENABLE_SIAMESE", "0") == "1",
        enable_temporal_segmentation=os.getenv("ENABLE_TEMPORAL", "0") == "1" and bool(temporal_weights),
        yolo_damage_weights=damage_weights,
        yolo_damage_type_weights=type_weights,
        yolo_damage_classifier_weights=classifier_weights,
        yolo_parts_weights=os.getenv("YOLO_PARTS_WEIGHTS"),
        siamese_weights=os.getenv("SIAMESE_WEIGHTS"),
        temporal_segmentation_weights=temporal_weights,
        device=resolve_device(os.getenv("MODEL_DEVICE")),
    )
    return DamageDetectionPipeline(config)


async def _read_pair(before: UploadFile, after: UploadFile):
    try:
        return decode_image_bytes(await before.read()), decode_image_bytes(await after.read())
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    pipeline = _pipeline()
    return {
        "status": "ok",
        "mode_par_defaut": "hybride" if pipeline.config.enable_yolo else "classique_sans_poids",
        "modele_degats_charge": pipeline.damage_segmenter.enabled,
        "modele_types_charge": pipeline.damage_type_segmenter.enabled,
        "classificateur_charge": pipeline.damage_classifier.enabled,
        "modele_temporel_charge": pipeline.temporal_segmenter.enabled,
        "version": "4.0.0",
    }


@app.post("/analyze")
async def analyze(
    before: Annotated[UploadFile, File(description="Photographie avant location")],
    after: Annotated[UploadFile, File(description="Photographie au retour")],
    sensitivity: Annotated[float, Form()] = 0.55,
    include_images_base64: Annotated[bool, Form()] = False,
) -> dict:
    image_before, image_after = await _read_pair(before, after)
    result, images = _pipeline(sensitivity).analyze(image_before, image_after)
    if include_images_base64:
        result["images_base64_png"] = {
            name: base64.b64encode(encode_image(image, ".png")).decode("ascii") for name, image in images.items()
        }
    return result


@app.post("/analyze/archive")
async def analyze_archive(
    before: Annotated[UploadFile, File(description="Photographie avant location")],
    after: Annotated[UploadFile, File(description="Photographie au retour")],
    sensitivity: Annotated[float, Form()] = 0.55,
):
    image_before, image_after = await _read_pair(before, after)
    result, images = _pipeline(sensitivity).analyze(image_before, image_after)
    payload = result_zip_bytes(result, images)
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="analyse_degats.zip"'},
    )


@app.get("/demo")
def demo(include_images_base64: bool = False) -> dict:
    before, after, _, metadata = generate_synthetic_pair()
    result, images = _pipeline().analyze(before, after)
    result["synthetic_example"] = metadata
    if include_images_base64:
        result["images_base64_png"] = {
            name: base64.b64encode(encode_image(image, ".png")).decode("ascii") for name, image in images.items()
        }
    return result


@app.get("/demo/archive")
def demo_archive():
    before, after, _, metadata = generate_synthetic_pair()
    result, images = _pipeline().analyze(before, after)
    result["synthetic_example"] = metadata
    return StreamingResponse(
        io.BytesIO(result_zip_bytes(result, images)),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="demo_analyse_degats.zip"'},
    )
