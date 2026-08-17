from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable


# Barème volontairement prudent pour une démonstration produit. Ces montants ne
# remplacent pas les tarifs d'un réparateur et pourront ensuite être externalisés
# dans une table administrable.
PRICE_RANGES_EUR: dict[str, dict[str, tuple[int, int, str]]] = {
    "scratch": {
        "faible": (90, 190, "Polissage ou retouche localisée"),
        "moyenne": (180, 380, "Retouche et raccord de peinture"),
        "forte": (350, 700, "Préparation et peinture de l'élément"),
    },
    "dent": {
        "faible": (120, 260, "Débosselage sans peinture"),
        "moyenne": (250, 520, "Débosselage et retouche de peinture"),
        "forte": (500, 950, "Redressage et peinture de l'élément"),
    },
    "hail": {
        "faible": (220, 450, "Débosselage localisé des impacts"),
        "moyenne": (420, 950, "Débosselage sans peinture multi-impact"),
        "forte": (850, 1_800, "Débosselage étendu et reprise de finition"),
    },
    "crack": {
        "faible": (100, 240, "Réparation localisée ou résine"),
        "moyenne": (280, 650, "Réparation renforcée ou remplacement partiel"),
        "forte": (600, 1_300, "Remplacement probable de l'élément"),
    },
    "paint_chip": {
        "faible": (70, 160, "Retouche peinture localisée"),
        "moyenne": (140, 320, "Raccord de peinture"),
        "forte": (300, 600, "Préparation et peinture de l'élément"),
    },
    "broken_part": {
        "faible": (180, 450, "Réparation ou fixation de l'élément"),
        "moyenne": (400, 900, "Réparation lourde ou remplacement"),
        "forte": (800, 1_800, "Remplacement et finition"),
    },
    "corrosion": {
        "faible": (160, 340, "Traitement local et retouche"),
        "moyenne": (320, 700, "Traitement, préparation et peinture"),
        "forte": (650, 1_400, "Réfection approfondie de l'élément"),
    },
    "surface_change": {
        "faible": (100, 240, "Contrôle et correction localisée"),
        "moyenne": (220, 520, "Réparation et retouche de finition"),
        "forte": (480, 1_000, "Réparation étendue et peinture"),
    },
}


def _ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character)).lower()


def _damage_family(value: str) -> str:
    name = _ascii(value).replace("_probable", "")
    if "rayure" in name or "scratch" in name:
        return "scratch"
    if "grele" in name or "hail" in name:
        return "hail"
    if "bosse" in name or "dent" in name:
        return "dent"
    if "fissure" in name or "crack" in name:
        return "crack"
    if "eclat" in name or "paint_chip" in name or "peinture_ecaillee" in name:
        return "paint_chip"
    if "cassee" in name or "manquante" in name or "broken" in name or "missing" in name:
        return "broken_part"
    if "corrosion" in name:
        return "corrosion"
    return "surface_change"


def _part_multiplier(labels: Iterable[str]) -> float:
    parts = " ".join(_ascii(str(label)) for label in labels)
    if any(token in parts for token in ("windshield", "pare-brise", "window", "vitre", "light", "phare")):
        return 1.20
    if any(token in parts for token in ("roof", "toit", "quarter", "aile", "rocker", "bas de caisse")):
        return 1.12
    if any(token in parts for token in ("bumper", "pare-chocs")):
        return 0.92
    return 1.0


def _round_low(value: float) -> int:
    return max(0, int(math.floor(value / 10.0) * 10))


def _round_high(value: float) -> int:
    return max(0, int(math.ceil(value / 10.0) * 10))


def estimate_repair_cost(
    regions: Iterable[dict],
    *,
    has_new_damage: bool,
    alignment_reliability: float,
    hail_impact_count: int = 0,
) -> dict:
    """Produit une fourchette indicative à partir des seuls dégâts acceptés."""

    accepted = list(regions)[:100]
    disclaimer = (
        "Estimation indicative issue d'un barème de démonstration. Elle ne constitue pas un devis "
        "et doit être confirmée après inspection par un professionnel."
    )
    if not has_new_damage or not accepted:
        return {
            "available": False,
            "currency": "EUR",
            "low": None,
            "high": None,
            "display": "Non applicable",
            "confidence": "non calculée",
            "repair_methods": [],
            "region_estimates": [],
            "pricing_basis": "Barème de démonstration configurable",
            "assumptions": ["Aucun nouveau dégât n'a été suffisamment confirmé pour produire une estimation."],
            "disclaimer": disclaimer,
        }

    estimates = []
    methods: list[str] = []
    missing_part = False
    confidence_values = []
    for region in accepted:
        family = _damage_family(str(region.get("damage_type", "")))
        severity = str(region.get("severity", "moyenne")).lower()
        if severity not in PRICE_RANGES_EUR[family]:
            severity = "moyenne"
        low, high, method = PRICE_RANGES_EUR[family][severity]
        labels = [str(label) for label in (region.get("part_labels") or [])]
        missing_part = missing_part or not labels
        multiplier = _part_multiplier(labels)

        # L'aire relative ne donne pas des cm². Elle ne sert donc qu'à une petite
        # modulation à l'intérieur d'une fourchette déjà déterminée par la gravité.
        try:
            area_ratio = float(region.get("area_ratio", 0.0))
        except (TypeError, ValueError):
            area_ratio = 0.0
        area_ratio = max(0.0, area_ratio) if math.isfinite(area_ratio) else 0.0
        multiplier *= 1.0 + min(area_ratio / 0.03, 1.0) * 0.12
        if family == "hail" and hail_impact_count >= 12:
            multiplier *= 1.15 if hail_impact_count < 30 else 1.30

        region_low = _round_low(low * multiplier)
        region_high = _round_high(high * multiplier)
        if method not in methods:
            methods.append(method)
        try:
            confidence = float(region.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence_values.append(min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else 0.0)
        estimates.append(
            {
                "region_id": region.get("id"),
                "damage_type": region.get("damage_type", "changement_surface_probable"),
                "severity": severity,
                "part_labels": labels,
                "method": method,
                "low": region_low,
                "high": region_high,
            }
        )

    total_low = sum(item["low"] for item in estimates)
    total_high = sum(item["high"] for item in estimates)
    mean_confidence = sum(confidence_values) / max(len(confidence_values), 1)
    if alignment_reliability >= 0.75 and mean_confidence >= 0.80 and not missing_part:
        estimate_confidence = "élevée"
    elif alignment_reliability >= 0.50 and mean_confidence >= 0.65:
        estimate_confidence = "moyenne"
    else:
        estimate_confidence = "faible"

    assumptions = [
        "Montants exprimés en euros TTC, pièces et main-d'œuvre comprises à titre indicatif.",
        "La surface détectée dans l'image n'est pas une mesure physique en cm².",
    ]
    if missing_part:
        assumptions.append("La pièce automobile n'étant pas identifiée avec certitude, aucun coefficient précis de pièce n'est appliqué.")

    return {
        "available": True,
        "currency": "EUR",
        "low": total_low,
        "high": total_high,
        "display": f"{total_low:,} € – {total_high:,} €".replace(",", " "),
        "confidence": estimate_confidence,
        "repair_methods": methods,
        "region_estimates": estimates,
        "pricing_basis": "Barème de démonstration configurable",
        "assumptions": assumptions,
        "disclaimer": disclaimer,
    }
