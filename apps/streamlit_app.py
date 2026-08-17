from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodamage.exporting import result_zip_bytes  # noqa: E402
from autodamage.device import resolve_device  # noqa: E402
from autodamage.io_utils import decode_image_bytes  # noqa: E402
from autodamage.pipeline import DamageDetectionPipeline, PipelineConfig  # noqa: E402
from autodamage.synthetic import generate_synthetic_pair  # noqa: E402

st.set_page_config(page_title="AutoDamage — Inspection visuelle", page_icon="AD", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --ink: #f5f7fa; --muted: #8d98a8; --line: rgba(255,255,255,.10);
      --panel: rgba(13,17,23,.78); --accent: #7dd3fc; --success: #52d3a2;
    }
    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(circle at 75% -10%, rgba(42,94,125,.22), transparent 34rem),
        linear-gradient(180deg, #05070a 0%, #080b10 45%, #06080c 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer {display:none !important;}
    :focus-visible {outline:3px solid #f7c948 !important; outline-offset:3px !important;}
    button:focus-visible, input:focus-visible, [role="tab"]:focus-visible, [role="radio"]:focus-visible {box-shadow:0 0 0 4px rgba(247,201,72,.28) !important;}
    .block-container {max-width: 1240px; padding: 2rem 2.4rem 4rem;}
    .ad-nav {display:flex; align-items:center; justify-content:space-between; margin-bottom:5.5rem;}
    .ad-brand {font-size:.78rem; font-weight:700; letter-spacing:.18em; color:#fff;}
    .ad-nav-meta {font-size:.70rem; letter-spacing:.12em; color:var(--muted); text-transform:uppercase;}
    .ad-hero {max-width:960px; margin-bottom:3.5rem;}
    .ad-kicker {color:var(--accent); text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700;}
    .ad-hero h1 {font-size:clamp(3rem,7vw,6.25rem); line-height:.94; letter-spacing:-.065em; font-weight:590; margin:.65rem 0 1.5rem;}
    .ad-hero p {font-size:clamp(1.05rem,2vw,1.35rem); line-height:1.55; color:#aab3c0; max-width:700px;}
    .ad-pills {display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.6rem;}
    .ad-pill {border:1px solid var(--line); border-radius:99px; padding:.42rem .72rem; font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:#b7c0cc; background:rgba(255,255,255,.025);}
    .ad-pill.live::before {content:""; display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--success); margin-right:.5rem; box-shadow:0 0 12px rgba(82,211,162,.8);}
    .ad-section-title {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:2.8rem 0 1rem;}
    [data-testid="stRadio"] > label {display:none;}
    [data-testid="stRadio"] div[role="radiogroup"] {gap:.5rem;}
    [data-testid="stRadio"] label {background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:99px; padding:.52rem 1rem;}
    [data-testid="stFileUploader"] {border:1px solid var(--line); border-radius:18px; padding:.8rem 1rem .25rem; background:var(--panel);}
    [data-testid="stFileUploaderDropzone"] {background:rgba(255,255,255,.025); border:1px dashed rgba(255,255,255,.18); border-radius:13px; min-height:150px;}
    [data-testid="stExpander"] {border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.018);}
    .stButton > button[kind="primary"] {height:3.6rem; border-radius:99px; border:0; background:#f5f7fa; color:#05070a; font-weight:750; letter-spacing:.08em; text-transform:uppercase; transition:all .22s ease;}
    .stButton > button[kind="primary"]:hover {background:#fff; transform:translateY(-1px); box-shadow:0 14px 40px rgba(125,211,252,.13);}
    [data-testid="stImage"] img {border-radius:16px; border:1px solid var(--line);}
    [data-testid="stMetric"] {border-top:1px solid var(--line); padding:1rem .15rem 1.2rem; background:transparent;}
    [data-testid="stMetricLabel"] {color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:.68rem;}
    [data-testid="stMetricValue"] {font-size:1.65rem; letter-spacing:-.04em;}
    .ad-result {border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.4rem; margin:2.5rem 0 1rem; background:var(--panel);}
    .ad-result strong {font-size:1.05rem; font-weight:620;}
    .ad-result span {display:block; color:var(--muted); font-size:.85rem; margin-top:.3rem;}
    .ad-result.alert {border-color:rgba(255,116,116,.35); box-shadow:inset 3px 0 #ff7474;}
    .ad-result.review {border-color:rgba(255,190,92,.32); box-shadow:inset 3px 0 #ffbe5c;}
    .ad-result.clear {border-color:rgba(82,211,162,.30); box-shadow:inset 3px 0 var(--success);}
    .ad-estimate {border:1px solid rgba(125,211,252,.22); border-radius:18px; padding:1.35rem 1.45rem; background:linear-gradient(135deg,rgba(125,211,252,.08),rgba(255,255,255,.02));}
    .ad-estimate-price {font-size:clamp(2rem,4vw,3.2rem); font-weight:610; letter-spacing:-.055em; margin:.25rem 0;}
    .ad-estimate-meta {color:var(--muted); font-size:.82rem; line-height:1.55;}
    .stTabs [data-baseweb="tab-list"] {gap:1.6rem; border-bottom:1px solid var(--line);}
    .stTabs [data-baseweb="tab"] {padding:.7rem 0; color:var(--muted);}
    .stTabs [aria-selected="true"] {color:#fff !important;}
    .ad-footer {margin-top:5rem; padding-top:1.2rem; border-top:1px solid var(--line); color:#697382; font-size:.73rem; line-height:1.6;}
    @media (max-width: 720px) {.block-container{padding:1.2rem 1rem 3rem}.ad-nav{margin-bottom:3.5rem}.ad-hero h1{font-size:3.35rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DAMAGE_WEIGHTS = ROOT / "models" / "car_damage_v3.pt"
if not DEFAULT_DAMAGE_WEIGHTS.exists():
    DEFAULT_DAMAGE_WEIGHTS = ROOT / "models" / "car_damage_v2.pt"
DEFAULT_TYPE_WEIGHTS = ROOT / "models" / "car_damage_types_v2.pt"
DEFAULT_CLASSIFIER_WEIGHTS = ROOT / "models" / "car_damage_classifier_v3.pt"
DEFAULT_TEMPORAL_WEIGHTS = ROOT / "models" / "temporal_damage_v4.pt"

damage_weights = Path(os.getenv("YOLO_DAMAGE_WEIGHTS", str(DEFAULT_DAMAGE_WEIGHTS)))
type_weights = Path(os.getenv("YOLO_DAMAGE_TYPE_WEIGHTS", str(DEFAULT_TYPE_WEIGHTS)))
classifier_weights = Path(os.getenv("YOLO_DAMAGE_CLASSIFIER_WEIGHTS", str(DEFAULT_CLASSIFIER_WEIGHTS)))
temporal_weights = Path(os.getenv("TEMPORAL_DAMAGE_WEIGHTS", str(DEFAULT_TEMPORAL_WEIGHTS)))
trained_model_available = damage_weights.exists()

model_label = "Modèle hybride prêt" if trained_model_available else "Mode temporel"
st.markdown(
    f"""
    <div class="ad-nav"><div class="ad-brand">AUTODAMAGE / 04</div><div class="ad-nav-meta">Inspection locale · confidentielle</div></div>
    <section class="ad-hero">
      <div class="ad-kicker">Vision automobile augmentée</div>
      <h1>Voir ce qui<br>vient de changer.</h1>
      <p>Comparez deux prises de vue. Le système aligne la carrosserie, isole les variations et documente les dégâts qui nécessitent votre attention.</p>
      <div class="ad-pills">
        <span class="ad-pill live">{model_label}</span>
        <span class="ad-pill">Rayures fines</span><span class="ad-pill">Grêle</span><span class="ad-pill">Analyse sur l’appareil</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="ad-section-title">01 — Choisir les images</div>', unsafe_allow_html=True)
mode = st.radio(
    "Source des images",
    ["Téléverser deux photos", "Essayer la démonstration"],
    horizontal=True,
    help="Choisissez une paire personnelle ou la démonstration locale.",
)

before = after = None
if mode == "Téléverser deux photos":
    upload_col1, upload_col2 = st.columns(2, gap="large")
    with upload_col1:
        upload_before = st.file_uploader(
            "AVANT — état de référence", type=["jpg", "jpeg", "png", "webp"], key="before"
        )
    with upload_col2:
        upload_after = st.file_uploader(
            "APRÈS — état au retour", type=["jpg", "jpeg", "png", "webp"], key="after"
        )
    if upload_before and upload_after:
        try:
            before = decode_image_bytes(upload_before.getvalue())
            after = decode_image_bytes(upload_after.getvalue())
        except Exception as exc:
            st.error(str(exc))
else:
    before, after, gt, _ = generate_synthetic_pair()
    st.caption("Scénario de démonstration chargé — perspective, lumière, reflet, rayure et enfoncement.")
    with st.expander("Afficher le masque attendu"):
        st.image(cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB), caption="Masque attendu : zones blanches correspondant aux dégâts synthétiques")

with st.expander("Réglages avancés"):
    setting1, setting2, setting3 = st.columns(3)
    with setting1:
        sensitivity = st.slider("Sensibilité", 0.0, 1.0, 0.55, 0.05)
    with setting2:
        min_area = st.number_input("Surface minimale (px)", min_value=20, max_value=5000, value=70, step=10)
    with setting3:
        enable_yolo = st.checkbox(
            "Modèles appris", value=trained_model_available, disabled=not trained_model_available
        )
    enable_siamese = st.checkbox("Vérificateur siamois externe", value=False)
    enable_temporal = st.checkbox(
        "Segmenter temporel expérimental", value=False, disabled=not temporal_weights.exists()
    )

if before is not None and after is not None:
    st.markdown('<div class="ad-section-title">02 — Vérifier la paire</div>', unsafe_allow_html=True)
    preview1, preview2 = st.columns(2, gap="large")
    preview1.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), caption="Avant", width="stretch")
    preview2.image(cv2.cvtColor(after, cv2.COLOR_BGR2RGB), caption="Après", width="stretch")

    run_requested = st.button("Lancer l’inspection", type="primary", width="stretch", help="Analyse les deux images et affiche un résultat textuel détaillé.")
    if run_requested:
        config = PipelineConfig(
            sensitivity=sensitivity,
            min_region_area=int(min_area),
            enable_yolo=enable_yolo,
            enable_siamese=enable_siamese,
            enable_temporal_segmentation=enable_temporal and temporal_weights.exists(),
            yolo_damage_weights=str(damage_weights) if damage_weights.exists() else None,
            yolo_damage_type_weights=str(type_weights) if type_weights.exists() else None,
            yolo_damage_classifier_weights=str(classifier_weights) if classifier_weights.exists() else None,
            yolo_parts_weights=os.getenv("YOLO_PARTS_WEIGHTS"),
            siamese_weights=os.getenv("SIAMESE_WEIGHTS"),
            temporal_segmentation_weights=str(temporal_weights) if temporal_weights.exists() else None,
            device=resolve_device(os.getenv("MODEL_DEVICE")),
        )
        with st.spinner("Alignement et analyse des surfaces…"):
            result, images = DamageDetectionPipeline(config).analyze(before, after)
        st.session_state["last_result"] = result
        st.session_state["last_images"] = images

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    images = st.session_state["last_images"]
    decision = result["decision"]
    st.markdown('<div class="ad-section-title">03 — Résultat</div>', unsafe_allow_html=True)
    if decision["has_new_damage"]:
        result_class, result_title, result_text = (
            "alert", "Nouveau dégât probable",
            f"{decision['accepted_candidate_count']} zone(s) dépassent le seuil de décision.",
        )
    elif decision["review_required"]:
        result_class, result_title, result_text = (
            "review", "Vérification humaine recommandée",
            "La qualité ou le recalage ne permettent pas une conclusion suffisamment robuste.",
        )
    else:
        result_class, result_title, result_text = (
            "clear", "Aucun nouveau dégât confirmé",
            "Aucune variation ne dépasse les seuils de décision configurés.",
        )
    st.markdown(
        f'<div class="ad-result {result_class}"><strong>{result_title}</strong><span>{result_text}</span></div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("Recalage", f"{result['alignment']['reliability_score']:.0%}")
    metric2.metric("Similarité", f"{result['change_detection']['ssim_global']:.3f}")
    metric3.metric("Candidats", decision["candidate_count"])
    metric4.metric("Analyse", f"{result['timings_seconds']['total']:.2f} s")
    metric5.metric("Impacts grêle", result.get("hail_detection", {}).get("micro_impact_count", 0))

    summary_tab, compare_tab, evidence_tab, data_tab = st.tabs(
        ["Synthèse", "Comparaison", "Preuves", "Données"]
    )
    with summary_tab:
        st.image(cv2.cvtColor(images["resultat_superpose"], cv2.COLOR_BGR2RGB), width="stretch")
    with compare_tab:
        compare1, compare2 = st.columns(2, gap="large")
        compare1.image(cv2.cvtColor(images["avant"], cv2.COLOR_BGR2RGB), caption="Référence", width="stretch")
        compare2.image(cv2.cvtColor(images["retour_normalise"], cv2.COLOR_BGR2RGB), caption="Retour aligné et normalisé", width="stretch")
    with evidence_tab:
        evidence1, evidence2 = st.columns(2, gap="large")
        evidence1.image(cv2.cvtColor(images["carte_changement"], cv2.COLOR_BGR2RGB), caption="Carte de changement", width="stretch")
        if "carte_grêle" in images:
            evidence2.image(cv2.cvtColor(images["carte_grêle"], cv2.COLOR_BGR2RGB), caption="Réponse grêle", width="stretch")
        else:
            evidence2.image(cv2.cvtColor(images["masque_changement"], cv2.COLOR_BGR2RGB), caption="Masque retenu", width="stretch")
    with data_tab:
        st.json(result)

    estimate = result.get("repair_estimate", {})
    st.markdown('<div class="ad-section-title">04 — Estimation de réparation</div>', unsafe_allow_html=True)
    if estimate.get("available"):
        methods = " · ".join(estimate.get("repair_methods", []))
        st.markdown(
            f"""
            <div class="ad-estimate">
              <div class="ad-estimate-meta">FOURCHETTE INDICATIVE TTC</div>
              <div class="ad-estimate-price">{estimate['display']}</div>
              <div class="ad-estimate-meta">{methods}<br>Fiabilité de l'estimation : {estimate['confidence']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Voir le détail de l'estimation"):
            for item in estimate.get("region_estimates", []):
                part = ", ".join(item.get("part_labels") or []) or "pièce non identifiée"
                st.write(
                    f"Zone {item['region_id']} — {item['method']} ({part}) : "
                    f"{item['low']} € à {item['high']} €"
                )
            for assumption in estimate.get("assumptions", []):
                st.caption(f"• {assumption}")
        st.caption(estimate["disclaimer"])
    else:
        st.info("Aucune estimation n'est produite tant qu'aucun nouveau dégât n'est suffisamment confirmé.")

    download1, download2 = st.columns(2, gap="large")
    download1.download_button(
        "Télécharger le dossier d’inspection",
        data=result_zip_bytes(result, images),
        file_name="inspection_autodamage.zip",
        mime="application/zip",
        width="stretch",
    )
    download2.download_button(
        "Télécharger les données JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="resultat.json",
        mime="application/json",
        width="stretch",
    )

st.markdown(
    '<div class="ad-footer">AUTODAMAGE TEMPORAL · SYSTÈME D’AIDE À L’INSPECTION<br>Une validation humaine reste nécessaire avant toute décision contractuelle ou assurantielle.</div>',
    unsafe_allow_html=True,
)
