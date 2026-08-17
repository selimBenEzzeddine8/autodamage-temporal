from __future__ import annotations

import hmac
import json
import os
import sqlite3
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "autodamage_demo.db"
RESULT = ROOT / "outputs" / "v4_demo" / "resultat.json"
DEMO_USER = os.getenv("AUTODAMAGE_ADMIN_USER", "admin_demo")
DEMO_PASSWORD = os.getenv("AUTODAMAGE_ADMIN_PASSWORD", "AutoDamageDemo2026!")

st.set_page_config(page_title="AutoDamage — Administration", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#0b0d10;color:#f3f5f7}
    [data-testid="stHeader"], footer {display:none}
    .block-container {max-width:1100px;padding-top:3rem}
    div[data-testid="stMetric"] {background:#14181e;border:1px solid #252b34;border-radius:16px;padding:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)


def authenticated() -> bool:
    if st.session_state.get("admin_authenticated"):
        return True
    st.title("Administration AutoDamage")
    st.caption("Accès local de démonstration — ne pas exposer directement sur Internet.")
    with st.form("login"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", width="stretch")
    if submitted:
        valid = hmac.compare_digest(username, DEMO_USER) and hmac.compare_digest(password, DEMO_PASSWORD)
        if valid:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        st.error("Identifiants incorrects.")
    return False


if not authenticated():
    st.stop()

top_left, top_right = st.columns([4, 1])
with top_left:
    st.title("Tableau de contrôle")
    st.caption("État local du paquet de remise AutoDamage Temporal 4.0")
with top_right:
    if st.button("Déconnexion", width="stretch"):
        st.session_state.clear()
        st.rerun()

model_files = sorted((ROOT / "models").glob("*.pt"))
database_status = "disponible" if DATABASE.exists() else "à initialiser"
result_status = "disponible" if RESULT.exists() else "absent"
m1, m2, m3 = st.columns(3)
m1.metric("Poids livrés", len(model_files))
m2.metric("Base SQLite", database_status)
m3.metric("Résultat de démonstration", result_status)

st.subheader("Inventaire des modèles")
st.dataframe(
    [{"fichier": item.name, "taille_Mo": round(item.stat().st_size / 1024**2, 2)} for item in model_files],
    width="stretch",
    hide_index=True,
)

if DATABASE.exists():
    with sqlite3.connect(DATABASE) as connection:
        rows = connection.execute(
            """
            SELECT r.contract_reference, v.registration, a.decision, a.confidence, a.created_at
            FROM analyses a
            JOIN rentals r ON r.id = a.rental_id
            JOIN vehicles v ON v.id = r.vehicle_id
            ORDER BY a.created_at DESC
            """
        ).fetchall()
    st.subheader("Analyses de démonstration")
    st.dataframe(
        [dict(zip(["contrat", "véhicule", "décision", "confiance", "date"], row)) for row in rows],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Exécuter `python scripts/init_demo_db.py` pour initialiser la base.")

if RESULT.exists():
    with st.expander("Dernier résultat JSON fourni"):
        st.json(json.loads(RESULT.read_text(encoding="utf-8")))
