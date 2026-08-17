from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .alignment import align_orb_ransac
from .change_detection import compare_temporal_masks, detect_changes, render_overlay
from .classifier_optional import OptionalYOLOClassifier, should_apply_classifier
from .hail_detection import detect_hail_damage
from .io_utils import ensure_bgr_uint8
from .photometric import normalize_photometry
from .quality import assess_image_quality, compare_capture_compatibility
from .repair_estimation import estimate_repair_cost
from .siamese import SiameseVerifier
from .temporal_segmentation import OptionalTemporalSegmenter
from .yolo_optional import OptionalYOLOSegmenter


@dataclass(slots=True)
class PipelineConfig:
    min_region_area: int = 70
    max_region_ratio: float = 0.12
    sensitivity: float = 0.55
    minimum_alignment_for_decision: float = 0.28
    minimum_region_confidence: float = 0.65
    yolo_damage_weights: str | None = None
    yolo_damage_type_weights: str | None = None
    yolo_damage_classifier_weights: str | None = None
    yolo_parts_weights: str | None = None
    siamese_weights: str | None = None
    temporal_segmentation_weights: str | None = None
    device: str = "cpu"
    enable_yolo: bool = False
    enable_siamese: bool = False
    enable_temporal_segmentation: bool = False
    enable_hail_detector: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DamageDetectionPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.damage_segmenter = OptionalYOLOSegmenter(
            self.config.yolo_damage_weights if self.config.enable_yolo else None,
            device=self.config.device,
            imgsz=768,
        )
        self.parts_segmenter = OptionalYOLOSegmenter(
            self.config.yolo_parts_weights if self.config.enable_yolo else None,
            device=self.config.device,
            imgsz=640,
        )
        self.damage_type_segmenter = OptionalYOLOSegmenter(
            self.config.yolo_damage_type_weights if self.config.enable_yolo else None,
            device=self.config.device,
            imgsz=640,
        )
        self.damage_classifier = OptionalYOLOClassifier(
            self.config.yolo_damage_classifier_weights if self.config.enable_yolo else None,
            device=self.config.device,
            imgsz=320,
        )
        self.siamese = SiameseVerifier(
            self.config.siamese_weights if self.config.enable_siamese else None,
            device=self.config.device,
        )
        self.temporal_segmenter = OptionalTemporalSegmenter(
            self.config.temporal_segmentation_weights if self.config.enable_temporal_segmentation else None,
            device=self.config.device,
        )

    @staticmethod
    def _crop_pair(before: np.ndarray, after: np.ndarray, bbox: list[int], pad: int = 18) -> tuple[np.ndarray, np.ndarray]:
        h, w = before.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return before[y1:y2, x1:x2], after[y1:y2, x1:x2]

    def analyze(self, before: np.ndarray, after: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
        start = perf_counter()
        before = ensure_bgr_uint8(before)
        after = ensure_bgr_uint8(after)
        quality_before = assess_image_quality(before)
        quality_after = assess_image_quality(after)
        compatibility = compare_capture_compatibility(before, after)

        t0 = perf_counter()
        alignment = align_orb_ransac(before, after)
        t_alignment = perf_counter() - t0

        t0 = perf_counter()
        normalized_after, photometric = normalize_photometry(before, alignment.aligned_after, alignment.valid_mask)
        t_photo = perf_counter() - t0

        hail = detect_hail_damage(before, normalized_after, alignment.valid_mask) if self.config.enable_hail_detector else None

        yolo_info = {
            "enabled": self.config.enable_yolo,
            "damage_model_loaded": self.damage_segmenter.enabled,
            "parts_model_loaded": self.parts_segmenter.enabled,
            "type_model_loaded": self.damage_type_segmenter.enabled,
            "classifier_model_loaded": self.damage_classifier.enabled,
            "damage_error": self.damage_segmenter.error,
            "parts_error": self.parts_segmenter.error,
            "type_error": self.damage_type_segmenter.error,
            "classifier_error": self.damage_classifier.error,
            "before_damage_instances": [],
            "after_damage_instances": [],
            "part_instances": [],
            "type_instances": [],
            "classifier_predictions": [],
        }
        external_new_mask = None
        yolo_new_mask = None
        temporal_mask = None
        temporal_info = {
            "enabled": self.config.enable_temporal_segmentation,
            "loaded": self.temporal_segmenter.enabled,
            "error": self.temporal_segmenter.error,
        }
        part_mask = None
        if self.damage_segmenter.enabled:
            before_damage, before_instances = self.damage_segmenter.predict_masks(before)
            after_damage, after_instances = self.damage_segmenter.predict_masks(normalized_after)
            yolo_new_mask = compare_temporal_masks(before_damage, after_damage)
            external_new_mask = yolo_new_mask
            yolo_info["before_damage_instances"] = before_instances
            yolo_info["after_damage_instances"] = after_instances
        if self.temporal_segmenter.enabled:
            temporal_mask, temporal_diagnostics = self.temporal_segmenter.predict_mask(before, normalized_after)
            temporal_info.update(temporal_diagnostics)
            if temporal_mask is not None:
                temporal_mask = cv2.bitwise_and(temporal_mask, alignment.valid_mask)
                external_new_mask = temporal_mask if external_new_mask is None else cv2.bitwise_or(external_new_mask, temporal_mask)
        if self.parts_segmenter.enabled:
            part_mask, part_instances = self.parts_segmenter.predict_masks(before)
            yolo_info["part_instances"] = part_instances
        if self.damage_type_segmenter.enabled:
            _, type_instances = self.damage_type_segmenter.predict_masks(normalized_after)
            yolo_info["type_instances"] = type_instances

        t0 = perf_counter()
        changes = detect_changes(
            before,
            normalized_after,
            alignment.valid_mask,
            min_region_area=self.config.min_region_area,
            max_region_ratio=self.config.max_region_ratio,
            sensitivity=self.config.sensitivity,
            external_new_damage_mask=external_new_mask,
            hail_mask=hail.mask if hail is not None else None,
        )
        t_change = perf_counter() - t0

        for region in changes.regions:
            if region.damage_type == "grêle_probable" and hail is not None:
                region.confidence = round(float(np.clip(0.70 * region.confidence + 0.30 * hail.confidence, 0, 1)), 4)
            if self.siamese.enabled:
                patch_b, patch_a = self._crop_pair(before, normalized_after, region.bbox_xyxy)
                if patch_b.size and patch_a.size:
                    prob = self.siamese.predict(patch_b, patch_a)
                    region.siamese_probability = None if prob is None else round(prob, 5)
                    if prob is not None:
                        region.confidence = round(float(np.clip(0.72 * region.confidence + 0.28 * prob, 0, 1)), 4)
            if part_mask is not None:
                x1, y1, x2, y2 = region.bbox_xyxy
                labels: list[str] = []
                for instance in yolo_info["part_instances"]:
                    px1, py1, px2, py2 = instance.get("bbox_xyxy", [0, 0, 0, 0])
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)
                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    region_area = max(1, (x2 - x1) * (y2 - y1))
                    if intersection / region_area >= 0.08:
                        labels.append(str(instance["class_name"]))
                region.part_labels = sorted(set(labels))
            best_type: tuple[float, str] | None = None
            x1, y1, x2, y2 = region.bbox_xyxy
            region_area = max(1, (x2 - x1) * (y2 - y1))
            for instance in yolo_info["type_instances"]:
                px1, py1, px2, py2 = instance.get("bbox_xyxy", [0, 0, 0, 0])
                intersection = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
                score = intersection / region_area * float(instance.get("confidence", 0.0))
                if score >= 0.04 and (best_type is None or score > best_type[0]):
                    best_type = (score, str(instance.get("class_name", "damage")))
            if best_type is not None:
                type_map = {
                    "Scratch": "rayure_probable",
                    "Dent": "bosse_probable",
                    "Cracked": "fissure_probable",
                    "Paint chip": "éclat_peinture_probable",
                    "Broken part": "pièce_cassée_probable",
                    "Missing part": "pièce_manquante_probable",
                    "Flaking": "peinture_écaillée_probable",
                    "Corrosion": "corrosion_probable",
                }
                region.damage_type = type_map.get(best_type[1], region.damage_type)
                if region.evidence is None:
                    region.evidence = {}
                region.evidence["learned_type"] = best_type[1]
                region.evidence["learned_type_score"] = round(best_type[0], 4)
            if self.damage_classifier.enabled and region.damage_type != "grêle_probable":
                _, patch_after = self._crop_pair(before, normalized_after, region.bbox_xyxy, pad=28)
                prediction = self.damage_classifier.predict(patch_after)
                if prediction is not None:
                    prediction_with_region = {"region_id": region.id, **prediction}
                    yolo_info["classifier_predictions"].append(prediction_with_region)
                    if region.evidence is None:
                        region.evidence = {}
                    region.evidence["classifier_type"] = prediction["class_name"]
                    region.evidence["classifier_confidence"] = prediction["confidence"]
                    proposed_type = prediction.get("damage_type")
                    has_segmented_type = "learned_type" in region.evidence
                    if should_apply_classifier(prediction, has_segmented_type):
                        region.damage_type = str(proposed_type)

        decision_regions = [r for r in changes.regions if r.confidence >= self.config.minimum_region_confidence]
        reliable_alignment = alignment.report.reliability_score >= self.config.minimum_alignment_for_decision
        has_new_damage = bool(decision_regions) and reliable_alignment
        review_required = (
            not reliable_alignment
            or quality_before.status == "insuffisant"
            or quality_after.status == "insuffisant"
            or (bool(changes.regions) and not has_new_damage)
        )
        decision = "nouveau_degât_probable" if has_new_damage else "aucun_nouveau_degât_confirmé"
        if review_required and not has_new_damage:
            decision = "revue_humaine_requise"

        repair_estimate = estimate_repair_cost(
            [region.to_dict() for region in decision_regions],
            has_new_damage=has_new_damage,
            alignment_reliability=alignment.report.reliability_score,
            hail_impact_count=hail.micro_impact_count if hail is not None else 0,
        )

        all_candidates_overlay = render_overlay(before, changes.mask, changes.regions)
        accepted_mask = np.zeros_like(changes.mask)
        for region in decision_regions:
            x1, y1, x2, y2 = region.bbox_xyxy
            accepted_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                accepted_mask[y1:y2, x1:x2], changes.mask[y1:y2, x1:x2]
            )
        overlay = render_overlay(before, accepted_mask, decision_regions)
        binary_bgr = cv2.cvtColor(changes.mask, cv2.COLOR_GRAY2BGR)
        images = {
            "avant": before,
            "retour_original": after,
            "retour_recale": alignment.aligned_after,
            "retour_normalise": normalized_after,
            "carte_changement": changes.heatmap,
            "masque_changement": binary_bgr,
            "masque_degats_acceptes": cv2.cvtColor(accepted_mask, cv2.COLOR_GRAY2BGR),
            "resultat_superpose": overlay,
            "tous_les_candidats": all_candidates_overlay,
        }
        if yolo_new_mask is not None:
            images["masque_nouveau_degât_yolo"] = cv2.cvtColor(yolo_new_mask, cv2.COLOR_GRAY2BGR)
        if temporal_mask is not None:
            images["masque_temporel_appris"] = cv2.cvtColor(temporal_mask, cv2.COLOR_GRAY2BGR)
        if hail is not None:
            images["masque_grêle"] = cv2.cvtColor(hail.mask, cv2.COLOR_GRAY2BGR)
            hail_heat = cv2.applyColorMap(np.clip(hail.score_map * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            images["carte_grêle"] = hail_heat

        result = {
            "schema_version": "4.0",
            "mode": "classique+options" if (self.config.enable_yolo or self.config.enable_siamese or self.config.enable_temporal_segmentation) else "classique_sans_poids",
            "decision": {
                "label": decision,
                "has_new_damage": has_new_damage,
                "review_required": review_required,
                "candidate_count": len(changes.regions),
                "accepted_candidate_count": len(decision_regions),
                "accepted_region_ids": [region.id for region in decision_regions],
                "limitations": [
                    "Une validation humaine reste nécessaire avant toute décision contractuelle ou assurantielle.",
                    "La saleté, les reflets spéculaires et les occultations peuvent encore produire des candidats.",
                ],
            },
            "quality": {
                "before": quality_before.to_dict(),
                "after": quality_after.to_dict(),
                "compatibility": compatibility,
            },
            "alignment": alignment.report.to_dict(),
            "photometric_normalization": photometric.to_dict(),
            "change_detection": {
                "ssim_global": changes.ssim_global,
                "threshold": changes.threshold,
                "diagnostics": changes.diagnostics,
                "regions": [region.to_dict() for region in changes.regions],
            },
            "hail_detection": hail.to_dict() if hail is not None else {"enabled": False},
            "repair_estimate": repair_estimate,
            "optional_models": {
                "yolo": yolo_info,
                "siamese": {
                    "enabled": self.config.enable_siamese,
                    "loaded": self.siamese.enabled,
                    "error": self.siamese.error,
                },
                "temporal_segmentation": temporal_info,
            },
            "configuration": self.config.to_dict(),
            "timings_seconds": {
                "alignment": round(t_alignment, 4),
                "photometric": round(t_photo, 4),
                "change_detection": round(t_change, 4),
                "total": round(perf_counter() - start, 4),
            },
        }
        return result, images

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DamageDetectionPipeline":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(PipelineConfig(**data))
