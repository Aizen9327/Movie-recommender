"""
CinéMatch — application de recommandation de films par contenu (content-based).

Inspirée d'un site de streaming (sidebar de navigation, barre de recherche, grilles
d'affiches avec rangs numérotés). Approche : TF-IDF sur les genres + similarité cosinus.

Les affiches sont récupérées via l'API TMDB (gratuite, clé requise — voir README). Sans
clé configurée, un visuel de remplacement s'affiche à la place, l'app reste utilisable.
"""

import html

import pandas as pd
import requests
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="CinéMatch — Recommandation de films", page_icon="🎬", layout="wide")

MODELS_DIR = "models"
DATA_DIR = "data"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_MEDIUM = "https://image.tmdb.org/t/p/w342"
TMDB_IMG_LARGE = "https://image.tmdb.org/t/p/w500"


# ----------------------------------------------------------------------------
# Identité visuelle
# ----------------------------------------------------------------------------
def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@500&display=swap');

        :root {
            --bg: #0B0C10;
            --panel: #15171D;
            --hairline: #262A33;
            --text: #ECEDEF;
            --muted: #9AA0AC;
            --red: #E2333A;
            --gold: #F2B84B;
        }

        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background-color: var(--bg); }

        section[data-testid="stSidebar"] {
            background-color: var(--panel); border-right: 1px solid var(--hairline);
        }
        section[data-testid="stSidebar"] button {
            font-family: 'Manrope', sans-serif !important; font-weight: 700 !important;
            text-align: left !important; justify-content: flex-start !important;
        }
        .sidebar-logo {
            font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.3rem;
            color: var(--text); letter-spacing: 0.01em; margin-bottom: 1.2rem;
        }
        .sidebar-logo b { color: var(--red); }

        .page-eyebrow {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--red); margin-bottom: 0.4rem;
        }
        .page-title {
            font-family: 'Manrope', sans-serif !important; font-weight: 800 !important;
            font-size: 2.1rem !important; color: var(--text) !important; margin: 0 0 1.1rem 0 !important;
        }
        .section-label {
            font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.05rem;
            color: var(--text); margin: 1.4rem 0 0.8rem 0;
        }

        div[data-testid="stTextInput"] input {
            background: var(--panel) !important; border: 1px solid var(--hairline) !important;
            border-radius: 999px !important; padding: 0.7rem 1.2rem !important;
            color: var(--text) !important; font-size: 0.95rem !important;
        }

        .poster-wrap {
            position: relative; border-radius: 10px; overflow: hidden; background: var(--panel);
            aspect-ratio: 2 / 3; margin-bottom: 0.5rem; border: 1px solid var(--hairline);
        }
        .poster-img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .poster-placeholder {
            width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center;
            justify-content: center; color: var(--muted); text-align: center; padding: 0.6rem;
            font-size: 0.72rem; gap: 0.4rem;
        }
        .poster-placeholder span { font-size: 1.9rem; }
        .poster-rank {
            position: absolute; left: -4px; bottom: -18px; font-family: 'Manrope', sans-serif;
            font-weight: 800; font-size: 4.2rem; color: transparent;
            -webkit-text-stroke: 2px var(--text); line-height: 1; pointer-events: none;
        }
        .poster-badge {
            position: absolute; top: 8px; right: 8px; background: var(--red); color: #fff;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
            padding: 0.15rem 0.5rem; border-radius: 999px;
        }
        .poster-caption {
            font-size: 0.78rem; color: var(--muted); line-height: 1.25; margin-bottom: 0.4rem;
            min-height: 2rem;
        }

        .detail-poster img, .detail-poster .poster-placeholder {
            border-radius: 12px; width: 100%; border: 1px solid var(--hairline);
        }
        .detail-title {
            font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.7rem; color: var(--text);
        }
        .detail-genres { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0 0.8rem 0; }
        .chip {
            font-family: 'Manrope', sans-serif; font-size: 0.7rem; letter-spacing: 0.02em;
            text-transform: uppercase; color: var(--muted); border: 1px solid var(--hairline);
            border-radius: 999px; padding: 0.15rem 0.6rem;
        }
        .tmdb-link { color: var(--gold); font-size: 0.85rem; text-decoration: none; }
        .tmdb-link:hover { text-decoration: underline; }

        .filmstrip {
            height: 12px;
            background-image: radial-gradient(circle, var(--hairline) 2.2px, transparent 2.6px);
            background-size: 20px 12px; opacity: 0.7; margin: 1.6rem 0;
        }
        .credits {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--muted); text-align: center; margin: 2rem 0 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Données et modèle (content-based uniquement)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    tfidf_matrix = sp.load_npz(f"{MODELS_DIR}/tfidf_matrix.npz")
    movies = pd.read_pickle(f"{MODELS_DIR}/movies.pkl")
    ratings = pd.read_pickle(f"{MODELS_DIR}/ratings.pkl")
    links = pd.read_csv(f"{DATA_DIR}/links.csv")
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return tfidf_matrix, movies, ratings


tfidf_matrix, movies, ratings = load_artifacts()
movie_id_to_idx = pd.Series(movies.index, index=movies["movieId"])


def content_based_scores(ref_movie_id):
    idx = movie_id_to_idx[ref_movie_id]
    return linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()


@st.cache_data
def compute_popular_movies(min_votes=20, top_n=10):
    """Classement façon IMDb : pondère la note moyenne par le nombre de votes, pour éviter
    qu'un film noté 5/5 par 2 personnes seulement écrase des films plébiscités par des centaines."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    stats.columns = ["movieId", "avg_rating", "n_ratings"]
    C = stats["avg_rating"].mean()
    m = min_votes
    stats["weighted"] = (stats["n_ratings"] / (stats["n_ratings"] + m)) * stats["avg_rating"] + (
        m / (stats["n_ratings"] + m)
    ) * C
    merged = stats.merge(movies, on="movieId")
    return merged.sort_values("weighted", ascending=False).head(top_n)


# ----------------------------------------------------------------------------
# Affiches (API TMDB, avec repli propre si pas de clé / film introuvable)
# ----------------------------------------------------------------------------
def get_tmdb_api_key():
    try:
        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_path(tmdb_id):
    api_key = get_tmdb_api_key()
    if not api_key or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{int(tmdb_id)}",
            params={"api_key": api_key, "language": "fr-FR"},
            timeout=4,
        )
        if r.status_code == 200:
            return r.json().get("poster_path")
    except requests.RequestException:
        pass
    return None


def poster_img_html(tmdb_id, title, size_base=TMDB_IMG_MEDIUM):
    path = fetch_poster_path(tmdb_id)
    if path:
        return f'<img src="{size_base}{path}" class="poster-img" alt="{html.escape(title)}"/>'
    return (
        '<div class="poster-placeholder"><span>🎞️</span>'
        f"<div>{html.escape(title)}</div></div>"
    )


# ----------------------------------------------------------------------------
# Composants d'affichage
# ----------------------------------------------------------------------------
def render_grid(df, columns=5, show_rank=False, show_score=False, key_prefix="grid"):
    df = df.reset_index(drop=True)
    for start in range(0, len(df), columns):
        row_df = df.iloc[start : start + columns]
        cols = st.columns(columns)
        for col, (i, movie) in zip(cols, row_df.iterrows()):
            with col:
                img_html = poster_img_html(movie.get("tmdbId"), movie["title"])
                overlay = ""
                if show_rank:
                    overlay += f'<div class="poster-rank">{i + 1}</div>'
                if show_score and "score" in movie:
                    overlay += f'<div class="poster-badge">{movie["score"]:.0%}</div>'
                st.markdown(f'<div class="poster-wrap">{img_html}{overlay}</div>', unsafe_allow_html=True)
                title_short = movie["title"] if len(movie["title"]) <= 38 else movie["title"][:36] + "…"
                st.markdown(f'<div class="poster-caption">{html.escape(title_short)}</div>', unsafe_allow_html=True)
                if st.button("Voir", key=f"{key_prefix}_{movie['movieId']}", use_container_width=True):
                    st.session_state["selected_movie_id"] = int(movie["movieId"])
                    st.rerun()


def render_movie_detail(movie_id):
    movie = movies.loc[movies["movieId"] == movie_id].iloc[0]
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        st.markdown(
            f'<div class="detail-poster">{poster_img_html(movie.get("tmdbId"), movie["title"], TMDB_IMG_LARGE)}</div>',
            unsafe_allow_html=True,
        )
        if st.button("✕ Fermer", use_container_width=True):
            st.session_state["selected_movie_id"] = None
            st.rerun()
    with col2:
        st.markdown(f'<div class="detail-title">{html.escape(movie["title"])}</div>', unsafe_allow_html=True)
        genres = [g for g in str(movie["genres"]).split("|") if g and g != "(no genres listed)"]
        chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genres)
        st.markdown(f'<div class="detail-genres">{chips}</div>', unsafe_allow_html=True)
        if pd.notna(movie.get("tmdbId")):
            st.markdown(
                f'<a class="tmdb-link" href="https://www.themoviedb.org/movie/{int(movie["tmdbId"])}" '
                'target="_blank">Voir la fiche complète sur TMDB ↗</a>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="section-label">Parce que tu as choisi ce film, voici des films similaires :</div>',
            unsafe_allow_html=True,
        )

    sim = content_based_scores(movie_id)
    recos = movies.copy()
    recos["score"] = sim
    recos = recos[recos["movieId"] != movie_id].sort_values("score", ascending=False).head(10)
    render_grid(recos, columns=5, show_score=True, key_prefix="reco")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_decouvrir():
    st.markdown('<div class="page-eyebrow">Content-based filtering</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Trouve ton prochain film</h1>', unsafe_allow_html=True)

    if not get_tmdb_api_key():
        st.info(
            "💡 Aucune clé API TMDB configurée : les films s'affichent avec un visuel de "
            "remplacement à la place de leur affiche. Voir le README pour l'activer (gratuit, 2 minutes)."
        )

    query = st.text_input(
        "Recherche", placeholder="🔍  Recherche un film par titre…", label_visibility="collapsed"
    )

    if query.strip():
        matches = movies[movies["title"].str.contains(query, case=False, na=False, regex=False)].head(12)
        st.markdown(f'<div class="section-label">Résultats pour « {html.escape(query)} »</div>', unsafe_allow_html=True)
        if matches.empty:
            st.warning("Aucun film trouvé avec ce titre.")
        else:
            render_grid(matches, columns=6, key_prefix="search")
    else:
        st.markdown('<div class="section-label">🔥 Les mieux notés du catalogue</div>', unsafe_allow_html=True)
        render_grid(compute_popular_movies(), columns=5, show_rank=True, key_prefix="top")

    sel = st.session_state.get("selected_movie_id")
    if sel:
        st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
        render_movie_detail(sel)


def page_methodo():
    st.markdown('<div class="page-eyebrow">Comment ça marche</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Recommandation par contenu</h1>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Films au catalogue", f"{len(movies):,}".replace(",", "\u2009"))
    c2.metric("Genres distincts", "20")
    c3.metric("Méthode", "TF-IDF + cosinus")

    st.markdown(
        """
        Chaque film est décrit par ses genres. On transforme cette description en vecteur
        numérique avec **TF-IDF** (qui donne plus de poids aux genres rares, donc plus
        discriminants), puis on mesure l'angle entre deux vecteurs avec la **similarité
        cosinus** : deux films avec les mêmes genres ont un score de 1, deux films sans genre
        commun ont un score de 0.

        Quand tu choisis un film, l'app calcule sa similarité avec tous les autres films du
        catalogue et te montre les 10 scores les plus élevés.

        **Limite à avoir en tête** : ce modèle ne regarde que les genres, donc deux films très
        différents en ton (ex. deux comédies, l'une familiale et l'autre plus noire) peuvent être
        jugés proches. C'est le compromis du content-based : disponible dès le premier jour
        (pas besoin de données utilisateur), mais moins fin qu'une approche collaborative.

        Le projet complet — incluant une comparaison chiffrée avec un système collaboratif
        (filtrage par les goûts d'autres utilisateurs, RMSE ≈ 0.88) et un système hybride —
        est détaillé dans le notebook du dépôt GitHub.
        """
    )


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
inject_style()

if "page" not in st.session_state:
    st.session_state.page = "decouvrir"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎬 CINÉ<b>MATCH</b></div>', unsafe_allow_html=True)
    if st.button(
        "🏠  Découvrir", use_container_width=True,
        type="primary" if st.session_state.page == "decouvrir" else "secondary",
    ):
        st.session_state.page = "decouvrir"
        st.rerun()
    if st.button(
        "📖  Méthodologie", use_container_width=True,
        type="primary" if st.session_state.page == "methodo" else "secondary",
    ):
        st.session_state.page = "methodo"
        st.rerun()
    st.markdown("---")
    st.caption("Projet portfolio · Recommandation de films\n\nDataset MovieLens · Approche content-based")

if st.session_state.page == "decouvrir":
    page_decouvrir()
else:
    page_methodo()

st.markdown('<div class="credits">CinéMatch — projet portfolio · MovieLens + TMDB</div>', unsafe_allow_html=True)
