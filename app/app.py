"""
Application Streamlit de démonstration : système de recommandation de films
(content-based / collaboratif / hybride).

Identité visuelle : "archive de cinémathèque" — encre, or, sarcelle. Une serif de
caractère pour les titres de films, du monospace pour les scores (le projet reste un
système de données, pas une vitrine), un badge de rang façon amorce de pellicule 35mm.

Les artefacts utilisés (modèle SVD entraîné, matrice TF-IDF, données) sont produits par le
notebook `notebook/recommandation_films.py` et chargés ici tels quels, en lecture seule.
"""

import html
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Recommandation de films", page_icon="🎬", layout="wide")

MODELS_DIR = "models"


# ----------------------------------------------------------------------------
# Identité visuelle — injectée une seule fois
# ----------------------------------------------------------------------------
def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Manrope:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --ink: #12151C;
            --panel: #1B2030;
            --hairline: #2B3142;
            --text: #EDEFF5;
            --muted: #8C93A8;
            --gold: #D7A34E;
            --teal: #4FB3A6;
        }

        html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; }
        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; }

        .filmstrip {
            height: 12px;
            background-image: radial-gradient(circle, var(--hairline) 2.2px, transparent 2.6px);
            background-size: 20px 12px;
            opacity: 0.7;
            margin: 0 0 1.4rem 0;
        }

        .hero-eyebrow {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--gold); margin-bottom: 0.5rem;
        }
        .hero-title {
            font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.5rem;
            color: var(--text); margin: 0; line-height: 1.1;
        }
        .hero-sub {
            font-family: 'Manrope', sans-serif; color: var(--muted); font-size: 0.98rem;
            margin-top: 0.6rem; max-width: 640px;
        }
        .hero-meta {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--muted); margin-top: 1.1rem;
        }
        .hero-meta b { color: var(--teal); }

        /* Onglets */
        button[data-baseweb="tab"] {
            font-family: 'Manrope', sans-serif !important; font-size: 0.85rem !important;
            letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--gold) !important; }
        div[data-baseweb="tab-highlight"] { background-color: var(--gold) !important; }
        div[data-baseweb="tab-border"] { background-color: var(--hairline) !important; }

        /* Cartes de profil */
        .profile-card {
            background: var(--panel); border: 1px solid var(--hairline); border-radius: 10px;
            padding: 0.9rem 1.1rem; margin-bottom: 1rem;
        }
        .profile-label {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--gold); margin-bottom: 0.6rem;
        }
        .profile-item {
            display: flex; justify-content: space-between; padding: 0.35rem 0;
            border-bottom: 1px solid var(--hairline); font-size: 0.92rem;
        }
        .profile-item:last-child { border-bottom: none; }
        .profile-title { color: var(--text); }
        .profile-rating { font-family: 'IBM Plex Mono', monospace; color: var(--gold); }

        /* Cartes de recommandation */
        .reco-card {
            display: flex; align-items: center; gap: 1.1rem; background: var(--panel);
            border: 1px solid var(--hairline); border-radius: 10px; padding: 0.85rem 1.1rem;
            margin-bottom: 0.55rem; transition: border-color .15s ease;
        }
        .reco-card:hover { border-color: var(--gold); }
        .reco-rank {
            width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--gold);
            display: flex; align-items: center; justify-content: center; flex-shrink: 0;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--gold);
        }
        .reco-body { flex: 1; min-width: 0; }
        .reco-title { font-family: 'Fraunces', serif; font-size: 1.05rem; color: var(--text); }
        .reco-genres { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem; }
        .chip {
            font-family: 'Manrope', sans-serif; font-size: 0.66rem; letter-spacing: 0.03em;
            text-transform: uppercase; color: var(--muted); border: 1px solid var(--hairline);
            border-radius: 999px; padding: 0.12rem 0.55rem;
        }
        .reco-score { width: 170px; flex-shrink: 0; text-align: right; }
        .score-bar {
            width: 100%; height: 6px; border-radius: 999px; background: var(--hairline);
            overflow: hidden; margin-bottom: 0.35rem;
        }
        .score-fill { height: 100%; border-radius: 999px; }
        .score-fill-content { background: var(--gold); }
        .score-fill-collab { background: var(--teal); }
        .score-fill-hybrid { background: linear-gradient(90deg, var(--gold), var(--teal)); }
        .score-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--muted); }

        .credits {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--muted); text-align: center; margin: 1.6rem 0 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(n_movies, n_users, n_ratings):
    st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-eyebrow">Système de recommandation — MovieLens</div>
        <div class="hero-title">🎬 Trois façons de te recommander un film</div>
        <div class="hero-sub">
            Compare en direct trois approches de recommandation entraînées sur le même jeu de
            données : par contenu, par filtrage collaboratif, et un système hybride combinant
            les deux.
        </div>
        <div class="hero-meta">
            <b>{n_movies:,}</b> films &nbsp;·&nbsp; <b>{n_users:,}</b> utilisateurs &nbsp;·&nbsp;
            <b>{n_ratings:,}</b> notes &nbsp;·&nbsp; dataset MovieLens (ml-latest-small)
        </div>
        """.replace(",", "\u2009"),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="filmstrip" style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)


def render_profile(df):
    rows = "".join(
        f'<div class="profile-item"><span class="profile-title">{html.escape(r.title)}</span>'
        f'<span class="profile-rating">★ {r.rating:.1f}</span></div>'
        for r in df.itertuples()
    )
    st.markdown(
        f'<div class="profile-card"><div class="profile-label">Profil — films préférés</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def render_recommendation_cards(df, score_col, score_label, variant, max_score=None):
    if max_score is None:
        max_score = df[score_col].max() or 1
    cards = []
    for i, row in enumerate(df.itertuples(), start=1):
        genres = [g for g in str(row.genres).split("|") if g and g != "(no genres listed)"]
        chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genres[:4])
        score_val = getattr(row, score_col)
        pct = max(0, min(100, (score_val / max_score) * 100)) if max_score else 0
        cards.append(
            f"""
            <div class="reco-card">
                <div class="reco-rank">{i:02d}</div>
                <div class="reco-body">
                    <div class="reco-title">{html.escape(row.title)}</div>
                    <div class="reco-genres">{chips}</div>
                </div>
                <div class="reco-score">
                    <div class="score-bar"><div class="score-fill score-fill-{variant}" style="width:{pct:.0f}%"></div></div>
                    <div class="score-value">{score_label} {score_val:.2f}</div>
                </div>
            </div>
            """
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


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
inject_style()
render_hero(len(movies), ratings["userId"].nunique(), len(ratings))

tab_content, tab_collab, tab_hybrid = st.tabs(
    ["📋 Content-based", "👥 Collaboratif", "🔀 Hybride"]
)

# --- Onglet Content-based ---
with tab_content:
    col_ctrl, col_results = st.columns([1, 2], gap="large")
    with col_ctrl:
        st.markdown("Choisis un film : on te recommande des films aux **genres similaires**.")
        movie_title = st.selectbox(
            "Film de référence", movies["title"].sort_values().tolist(), key="content_movie"
        )
        go_content = st.button("Recommander", key="btn_content", use_container_width=True)
        ref_id = movies.loc[movies["title"] == movie_title, "movieId"].iloc[0]
        st.caption("Genres du film choisi : " + str(movies.loc[movies["movieId"] == ref_id, "genres"].iloc[0]))

    with col_results:
        if go_content:
            recos = get_recommendations("content", ref_movie_id=ref_id, n=10)
            render_recommendation_cards(recos, "score", "similarité", "content", max_score=1.0)
        else:
            st.info("Choisis un film à gauche, puis clique sur « Recommander ».")

# --- Onglet Collaboratif ---
with tab_collab:
    col_ctrl, col_results = st.columns([1, 2], gap="large")
    with col_ctrl:
        st.markdown(
            "Choisis un utilisateur existant du dataset : le modèle (SVD) lui recommande des "
            "films qu'il n'a pas encore notés, à partir des goûts d'utilisateurs similaires."
        )
        user_id = st.selectbox(
            "Utilisateur", sorted(ratings["userId"].unique().tolist()), key="collab_user"
        )
        go_collab = st.button("Recommander", key="btn_collab", use_container_width=True)
        user_top = (
            ratings[ratings["userId"] == user_id]
            .merge(movies, on="movieId")
            .sort_values("rating", ascending=False)
            .head(5)
        )
        render_profile(user_top)

    with col_results:
        if go_collab:
            recos = get_recommendations("collaborative", user_id=user_id, n=10)
            render_recommendation_cards(recos, "note_predite", "note prédite", "collab", max_score=RATING_MAX)
        else:
            st.info("Choisis un utilisateur à gauche, puis clique sur « Recommander ».")

# --- Onglet Hybride ---
with tab_hybrid:
    col_ctrl, col_results = st.columns([1, 2], gap="large")
    with col_ctrl:
        st.markdown(
            "Combine la similarité avec le film préféré de l'utilisateur et la note prédite "
            "par le modèle collaboratif."
        )
        user_id_h = st.selectbox(
            "Utilisateur", sorted(ratings["userId"].unique().tolist()), key="hybrid_user"
        )
        content_weight = st.slider(
            "Poids du contenu (le reste va au collaboratif)",
            min_value=0.0, max_value=1.0, value=0.4, step=0.1,
        )
        go_hybrid = st.button("Recommander", key="btn_hybrid", use_container_width=True)

        ref_row = (
            ratings[ratings["userId"] == user_id_h]
            .sort_values(["rating", "timestamp"], ascending=[False, False])
            .head(1)
        )
        if not ref_row.empty:
            ref_movie_id_h = ref_row["movieId"].iloc[0]
            ref_title_h = movies.loc[movies["movieId"] == ref_movie_id_h, "title"].iloc[0]
            st.caption(f"Film de référence (le mieux noté par cet utilisateur) : **{ref_title_h}**")
        else:
            ref_movie_id_h = None
            st.warning("Cet utilisateur n'a aucune note dans le dataset.")

    with col_results:
        if go_hybrid and ref_movie_id_h is not None:
            recos = get_recommendations(
                "hybrid", user_id=user_id_h, ref_movie_id=ref_movie_id_h,
                content_weight=content_weight, n=10,
            )
            render_recommendation_cards(recos, "score", "score final", "hybrid", max_score=1.0)
        else:
            st.info("Choisis un utilisateur et un poids à gauche, puis clique sur « Recommander ».")

st.markdown('<div class="filmstrip" style="margin-top:2rem;"></div>', unsafe_allow_html=True)

with st.expander("ℹ️ Méthodologie et résultats"):
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (collaboratif)", "0.877")
    c2.metric("Hit Rate@10 — CF seul", "35.0 %")
    c3.metric("Hit Rate@10 — Hybride", "20.0 %")
    st.markdown(
        """
        - **Content-based** : TF-IDF sur les genres + similarité cosinus.
        - **Collaboratif** : factorisation matricielle (SVD) via la librairie Surprise.
        - **Hybride** : combinaison pondérée des deux scores ci-dessus.

        Le notebook complet (exploration, méthodologie détaillée, évaluation comparative)
        est disponible dans le dépôt GitHub du projet.
        """
    )

st.markdown('<div class="credits">Projet portfolio — recommandation de films · MovieLens</div>', unsafe_allow_html=True)
