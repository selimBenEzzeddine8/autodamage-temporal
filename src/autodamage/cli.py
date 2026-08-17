from __future__ import annotations

import argparse
import json
from pathlib import Path

from .device import resolve_device
from .exporting import write_result_directory
from .io_utils import read_image
from .pipeline import DamageDetectionPipeline, PipelineConfig
from .synthetic import generate_synthetic_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Détection temporelle de nouveaux dégâts automobiles")
    parser.add_argument("--before", type=Path, help="Image avant location")
    parser.add_argument("--after", type=Path, help="Image au retour")
    parser.add_argument("--output", type=Path, default=Path("outputs/analyse"))
    parser.add_argument("--demo", action="store_true", help="Utilise la paire synthétique incluse")
    parser.add_argument("--sensitivity", type=float, default=0.55)
    parser.add_argument("--yolo-damage-weights", type=str)
    parser.add_argument("--yolo-damage-type-weights", type=str)
    parser.add_argument("--yolo-damage-classifier-weights", type=str)
    parser.add_argument("--yolo-parts-weights", type=str)
    parser.add_argument("--siamese-weights", type=str)
    parser.add_argument("--temporal-damage-weights", type=str)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_damage = Path("models/car_damage_v2.pt")
    default_types = Path("models/car_damage_types_v2.pt")
    default_classifier = Path("models/car_damage_classifier_v3.pt")
    damage_weights = args.yolo_damage_weights or (str(default_damage) if default_damage.exists() else None)
    type_weights = args.yolo_damage_type_weights or (str(default_types) if default_types.exists() else None)
    classifier_weights = args.yolo_damage_classifier_weights or (
        str(default_classifier) if default_classifier.exists() else None
    )
    temporal_weights = args.temporal_damage_weights
    if args.demo:
        before, after, _, _ = generate_synthetic_pair()
    else:
        if not args.before or not args.after:
            raise SystemExit("Fournissez --before et --after, ou utilisez --demo.")
        before, after = read_image(args.before), read_image(args.after)
    config = PipelineConfig(
        sensitivity=float(max(0, min(1, args.sensitivity))),
        enable_yolo=bool(damage_weights or type_weights or classifier_weights or args.yolo_parts_weights),
        enable_siamese=bool(args.siamese_weights),
        enable_temporal_segmentation=bool(temporal_weights),
        yolo_damage_weights=damage_weights,
        yolo_damage_type_weights=type_weights,
        yolo_damage_classifier_weights=classifier_weights,
        yolo_parts_weights=args.yolo_parts_weights,
        siamese_weights=args.siamese_weights,
        temporal_segmentation_weights=temporal_weights,
        device=resolve_device(args.device),
    )
    result, images = DamageDetectionPipeline(config).analyze(before, after)
    write_result_directory(args.output, result, images)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(f"Résultats écrits dans: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
