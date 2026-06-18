"""
streamlit_app.py
=================
Démo interactive du système de recommandation. Réutilise les mêmes classes
que le notebook (src/recommender.py) — aucune logique dupliquée.

Pour lancer en local : streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
from recommender import (
    load_data,
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
)

st.set_page_config(page_title="Système de recommandation de films", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Chargement des données et entraînement des modèles (≈10s, une seule fois)...")
def get_models():
    movies, ratings = load_data()
    content_model = ContentBasedRecommender(movies)
    collab_model = CollaborativeRecommender(ratings, movies)
    hybrid_model = HybridRecommender(content_model, collab_model)
    return movies, ratings, content_model, collab_model, hybrid_model


movies, ratings, content_model, collab_model, hybrid_model = get_models()

st.title("🎬 Système de recommandation de films")
st.caption("Projet portfolio — content-based, collaborative filtering (SVD) et hybride, sur le dataset MovieLens.")

tab1, tab2, tab3 = st.tabs(["🎞️ Par film (content-based)", "👤 Par utilisateur (collaboratif)", "🧬 Hybride"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Trouver des films similaires à un film que vous aimez")
    st.write("Cette approche compare les **genres** des films entre eux (TF-IDF + similarité cosinus).")

    titles = movies["title"].sort_values().tolist()
    selected_title = st.selectbox("Choisissez un film", titles, index=titles.index("Toy Story (1995)") if "Toy Story (1995)" in titles else 0)
    top_n = st.slider("Nombre de recommandations", 5, 20, 10, key="cb_topn")

    if st.button("Recommander", key="cb_button"):
        results = content_model.recommend(selected_title, top_n=top_n)
        st.dataframe(
            results.rename(columns={"title": "Film", "genres": "Genres", "score_similarite": "Score de similarité"}),
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Recommandations personnalisées pour un utilisateur")
    st.write(
        "Cette approche se base sur les notes historiques de **tous** les utilisateurs "
        "(factorisation matricielle SVD) pour prédire ce qu'un utilisateur donné aimerait."
    )

    user_ids = sorted(ratings["userId"].unique().tolist())
    selected_user = st.selectbox("Choisissez un utilisateur (ID)", user_ids, key="cf_user")
    top_n_cf = st.slider("Nombre de recommandations", 5, 20, 10, key="cf_topn")

    with st.expander("Voir l'historique de notation de cet utilisateur"):
        history = (
            ratings[ratings.userId == selected_user]
            .merge(movies, on="movieId")
            .sort_values("rating", ascending=False)[["title", "rating"]]
        )
        st.dataframe(history.rename(columns={"title": "Film", "rating": "Note donnée"}), hide_index=True, use_container_width=True)

    if st.button("Recommander", key="cf_button"):
        results = collab_model.recommend(selected_user, top_n=top_n_cf)
        st.dataframe(
            results.rename(columns={"title": "Film", "genres": "Genres", "note_predite": "Note prédite"}),
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Approche hybride : collaboratif + contenu")
    st.write(
        "Combine la prédiction collaborative avec la ressemblance de contenu vis-à-vis des films "
        "déjà aimés par l'utilisateur (notes ≥ 4)."
    )

    selected_user_h = st.selectbox("Choisissez un utilisateur (ID)", user_ids, key="hy_user")
    top_n_h = st.slider("Nombre de recommandations", 5, 20, 10, key="hy_topn")
    alpha = st.slider(
        "Poids du signal collaboratif (α)", 0.0, 1.0, 0.6, step=0.1, key="hy_alpha",
        help="α=1 : 100% collaboratif. α=0 : 100% contenu.",
    )

    if st.button("Recommander", key="hy_button"):
        results = hybrid_model.recommend(selected_user_h, top_n=top_n_h, alpha=alpha)
        st.dataframe(
            results.rename(columns={
                "title": "Film", "genres": "Genres", "note_predite": "Note prédite (collab.)",
                "score_contenu": "Score de contenu", "score_hybride": "Score hybride",
            }),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption("Dataset : MovieLens ml-latest-small (GroupLens Research) — 100 836 notes, 9 742 films, 610 utilisateurs.")
