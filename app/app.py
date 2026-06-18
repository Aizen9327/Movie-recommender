"""
Application Streamlit de démonstration : système de recommandation de films
(content-based / collaboratif / hybride).

Les artefacts utilisés (modèle SVD entraîné, matrice TF-IDF, données) sont produits par le
notebook `notebook/recommandation_films.py` et chargés ici tels quels, en lecture seule.
"""

import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Recommandation de films", page_icon="🎬", layout="wide")

MODELS_DIR = "models"


# ----------------------------------------------------------------------------
# Chargement des artefacts (mis en cache : ne s'exécute qu'une fois par session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(f"{MODELS_DIR}/svd_model.pkl", "rb") as f:
        model = pickle.load(f)
    tfidf_matrix = sp.load_npz(f"{MODELS_DIR}/tfidf_matrix.npz")
    movies = pd.read_pickle(f"{MODELS_DIR}/movies.pkl")
    ratings = pd.read_pickle(f"{MODELS_DIR}/ratings.pkl")
    return model, tfidf_matrix, movies, ratings


model, tfidf_matrix, movies, ratings = load_artifacts()

RATING_MIN, RATING_MAX = ratings["rating"].min(), ratings["rating"].max()
movie_id_to_idx = pd.Series(movies.index, index=movies["movieId"])


# ----------------------------------------------------------------------------
# Fonctions de recommandation (mêmes principes que dans le notebook)
# ----------------------------------------------------------------------------
def content_based_scores(ref_movie_id):
    """Similarité entre un film de référence et tous les autres, calculée à la demande
    (on ne stocke jamais la matrice NxN complète, voir notebook section 6)."""
    idx = movie_id_to_idx[ref_movie_id]
    return linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()


@st.cache_data
def collaborative_scores(user_id):
    """Note prédite par le modèle SVD pour CHAQUE film, pour un utilisateur donné."""
    preds = np.array([model.predict(user_id, mid).est for mid in movies["movieId"]])
    return preds


def get_recommendations(mode, user_id=None, ref_movie_id=None, content_weight=0.4, n=10):
    already_seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"]) if user_id else set()
    candidates = movies.copy()

    if mode == "content":
        sim = content_based_scores(ref_movie_id)
        candidates["score"] = sim
        candidates = candidates[candidates["movieId"] != ref_movie_id]

    elif mode == "collaborative":
        candidates["note_predite"] = collaborative_scores(user_id)
        candidates = candidates[~candidates["movieId"].isin(already_seen)]
        candidates["score"] = candidates["note_predite"]

    elif mode == "hybrid":
        candidates["note_predite"] = collaborative_scores(user_id)
        candidates = candidates[~candidates["movieId"].isin(already_seen)]
        score_cf = (candidates["note_predite"] - RATING_MIN) / (RATING_MAX - RATING_MIN)
        if ref_movie_id is not None:
            sim_full = content_based_scores(ref_movie_id)
            score_content = candidates.index.map(lambda i: sim_full[i])
        else:
            score_content = 0.0
        candidates["score_contenu"] = score_content
        candidates["score"] = content_weight * candidates["score_contenu"] + (1 - content_weight) * score_cf

    return candidates.sort_values("score", ascending=False).head(n)


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🎬 Système de recommandation de films")
st.caption(
    "Démo interactive — dataset MovieLens (9 742 films, 610 utilisateurs, 100 836 notes). "
    "Code source et notebook complet sur GitHub."
)

tab_content, tab_collab, tab_hybrid = st.tabs(
    ["📋 Content-based", "👥 Collaboratif", "🔀 Hybride"]
)

# --- Onglet Content-based ---
with tab_content:
    st.subheader("Recommandation par contenu (similarité de genres)")
    st.write("Choisis un film : on te recommande des films aux genres similaires.")
    movie_title = st.selectbox(
        "Film de référence", movies["title"].sort_values().tolist(), key="content_movie"
    )
    if st.button("Recommander", key="btn_content"):
        ref_id = movies.loc[movies["title"] == movie_title, "movieId"].iloc[0]
        recos = get_recommendations("content", ref_movie_id=ref_id, n=10)
        st.dataframe(
            recos[["title", "genres", "score"]].rename(
                columns={"score": "similarité"}
            ),
            hide_index=True,
            use_container_width=True,
        )

# --- Onglet Collaboratif ---
with tab_collab:
    st.subheader("Recommandation collaborative (filtrage par les pairs)")
    st.write(
        "Choisis un utilisateur existant du dataset : on te montre ses films préférés, "
        "puis ce que le modèle (SVD) lui recommande parmi les films qu'il n'a pas encore notés."
    )
    user_id = st.selectbox(
        "Utilisateur", sorted(ratings["userId"].unique().tolist()), key="collab_user"
    )
    user_top = (
        ratings[ratings["userId"] == user_id]
        .merge(movies, on="movieId")
        .sort_values("rating", ascending=False)
        .head(5)[["title", "rating"]]
    )
    st.markdown("**Ses films préférés (dans le dataset) :**")
    st.dataframe(user_top, hide_index=True, use_container_width=True)

    if st.button("Recommander", key="btn_collab"):
        recos = get_recommendations("collaborative", user_id=user_id, n=10)
        st.markdown("**Recommandations pour cet utilisateur :**")
        st.dataframe(
            recos[["title", "genres", "note_predite"]].rename(
                columns={"note_predite": "note prédite"}
            ).round({"note prédite": 2}),
            hide_index=True,
            use_container_width=True,
        )

# --- Onglet Hybride ---
with tab_hybrid:
    st.subheader("Système hybride (contenu + collaboratif)")
    st.write(
        "Combine la similarité avec le film préféré de l'utilisateur et la note prédite "
        "par le modèle collaboratif. Ajuste le curseur pour voir l'effet de la pondération."
    )
    user_id_h = st.selectbox(
        "Utilisateur", sorted(ratings["userId"].unique().tolist()), key="hybrid_user"
    )
    content_weight = st.slider(
        "Poids du contenu (vs. poids collaboratif = 1 - ce curseur)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.1,
    )

    ref_row = (
        ratings[ratings["userId"] == user_id_h]
        .sort_values(["rating", "timestamp"], ascending=[False, False])
        .head(1)
    )
    if not ref_row.empty:
        ref_movie_id_h = ref_row["movieId"].iloc[0]
        ref_title_h = movies.loc[movies["movieId"] == ref_movie_id_h, "title"].iloc[0]
        st.caption(f"Film de référence (le mieux noté par cet utilisateur) : **{ref_title_h}**")

        if st.button("Recommander", key="btn_hybrid"):
            recos = get_recommendations(
                "hybrid", user_id=user_id_h, ref_movie_id=ref_movie_id_h,
                content_weight=content_weight, n=10,
            )
            st.dataframe(
                recos[["title", "genres", "score_contenu", "note_predite", "score"]]
                .rename(columns={
                    "score_contenu": "similarité contenu",
                    "note_predite": "note prédite",
                    "score": "score final",
                })
                .round(3),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning("Cet utilisateur n'a aucune note dans le dataset.")

st.divider()
with st.expander("ℹ️ À propos de ce projet"):
    st.markdown(
        """
        Ce projet compare trois familles de systèmes de recommandation sur le dataset MovieLens :

        - **Content-based** : TF-IDF sur les genres + similarité cosinus.
        - **Collaboratif** : factorisation matricielle (SVD) via la librairie Surprise,
          RMSE ≈ 0.88 sur un split train/test 80/20.
        - **Hybride** : combinaison pondérée des deux scores.

        Le notebook complet (exploration, méthodologie, évaluation comparative avec
        Hit Rate@10) est disponible dans le dépôt GitHub du projet.
        """
    )
