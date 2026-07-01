"""
CinéMatch — application de recommandation de films par contenu (content-based).

Architecture : le catalogue (recherche, tendances, métadonnées, affiches) vient EN DIRECT de
l'API TMDB — donc toujours à jour, y compris pour les sorties les plus récentes. Le scoring de
recommandation, lui, est calculé par nous : TF-IDF sur les genres + similarité cosinus,
recalculé à la volée sur le bassin de films renvoyé par TMDB à chaque recherche. Le catalogue
change tout le temps, l'algorithme reste le nôtre.

Pourquoi content-based uniquement ici : un film sorti hier n'a par définition aucune note
utilisateur, donc le filtrage collaboratif ne peut rien en dire (problème classique du cold
start). Le content-based n'a besoin que des métadonnées du film (genres), disponibles dès sa
sortie. La comparaison complète avec le collaboratif et l'hybride reste dans le notebook, sur le
jeu de données statique MovieLens (méthodologie reproductible, RMSE / Hit Rate@10).
"""

import html

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CinéMatch — Recommandation de films", page_icon="🎬", layout="wide")

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
            --bg: #0B0C10; --panel: #15171D; --hairline: #262A33;
            --text: #ECEDEF; --muted: #9AA0AC; --red: #E2333A; --gold: #F2B84B;
        }
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background-color: var(--bg); }

        section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--hairline); }
        section[data-testid="stSidebar"] button {
            font-family: 'Manrope', sans-serif !important; font-weight: 700 !important;
            text-align: left !important; justify-content: flex-start !important;
        }
        .sidebar-logo { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.3rem; color:var(--text); margin-bottom:1.2rem; }
        .sidebar-logo b { color: var(--red); }

        .page-eyebrow {
            font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.14em;
            text-transform:uppercase; color:var(--red); margin-bottom:0.4rem;
        }
        .page-title { font-family:'Manrope',sans-serif !important; font-weight:800 !important; font-size:2.1rem !important; color:var(--text) !important; margin:0 0 1.1rem 0 !important; }
        .section-label { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.05rem; color:var(--text); margin:1.4rem 0 0.8rem 0; }
        .live-tag {
            font-family:'IBM Plex Mono',monospace; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase;
            color: var(--gold); border:1px solid var(--hairline); border-radius:999px; padding:0.1rem 0.5rem; margin-left:0.5rem;
        }

        div[data-testid="stTextInput"] input {
            background: var(--panel) !important; border:1px solid var(--hairline) !important;
            border-radius:999px !important; padding:0.7rem 1.2rem !important; color:var(--text) !important; font-size:0.95rem !important;
        }

        .poster-wrap { position:relative; border-radius:10px; overflow:hidden; background:var(--panel); aspect-ratio:2/3; margin-bottom:0.5rem; border:1px solid var(--hairline); }
        .poster-img { width:100%; height:100%; object-fit:cover; display:block; }
        .poster-placeholder { width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; color:var(--muted); text-align:center; padding:0.6rem; font-size:0.72rem; gap:0.4rem; }
        .poster-placeholder span { font-size:1.9rem; }
        .poster-rank { position:absolute; left:-4px; bottom:-18px; font-family:'Manrope',sans-serif; font-weight:800; font-size:4.2rem; color:transparent; -webkit-text-stroke:2px var(--text); line-height:1; pointer-events:none; }
        .poster-badge { position:absolute; top:8px; right:8px; background:var(--red); color:#fff; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; font-weight:600; padding:0.15rem 0.5rem; border-radius:999px; }
        .poster-caption { font-size:0.78rem; color:var(--muted); line-height:1.25; margin-bottom:0.4rem; min-height:2rem; }

        .detail-poster img, .detail-poster .poster-placeholder { border-radius:12px; width:100%; border:1px solid var(--hairline); }
        .detail-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.7rem; color:var(--text); }
        .detail-overview { color: var(--muted); font-size:0.92rem; line-height:1.5; margin-top:0.6rem; max-width:680px; }
        .detail-genres { display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.6rem 0 0.8rem 0; }
        .chip { font-family:'Manrope',sans-serif; font-size:0.7rem; letter-spacing:0.02em; text-transform:uppercase; color:var(--muted); border:1px solid var(--hairline); border-radius:999px; padding:0.15rem 0.6rem; }
        .tmdb-link { color: var(--gold); font-size:0.85rem; text-decoration:none; }
        .tmdb-link:hover { text-decoration:underline; }

        .filmstrip { height:12px; background-image:radial-gradient(circle, var(--hairline) 2.2px, transparent 2.6px); background-size:20px 12px; opacity:0.7; margin:1.6rem 0; }
        .credits { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); text-align:center; margin:2rem 0 0.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Accès TMDB — catalogue toujours à jour (recherche, tendances, genres, affiches)
# ----------------------------------------------------------------------------
def get_tmdb_api_key():
    try:
        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None


def _tmdb_get(path, params=None, timeout=5):
    api_key = get_tmdb_api_key()
    if not api_key:
        return None
    params = dict(params or {})
    params["api_key"] = api_key
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_genre_map():
    data = _tmdb_get("/genre/movie/list", {"language": "fr-FR"})
    if not data:
        return {}
    return {g["id"]: g["name"] for g in data.get("genres", [])}


@st.cache_data(ttl=600, show_spinner=False)
def tmdb_search(query):
    data = _tmdb_get("/search/movie", {"query": query, "language": "fr-FR", "include_adult": "false"})
    if not data:
        return []
    return data.get("results", [])[:12]


@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_popular():
    data = _tmdb_get("/movie/popular", {"language": "fr-FR", "page": 1})
    if not data:
        return []
    return data.get("results", [])[:10]


@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_discover_by_genres(genre_ids_tuple, exclude_id=None):
    """Bassin de films candidats partageant au moins un genre avec le film cible — vient en
    direct de TMDB, donc inclut nativement les sorties les plus récentes."""
    if not genre_ids_tuple:
        return []
    results = []
    for page in (1, 2):
        data = _tmdb_get(
            "/discover/movie",
            {
                "language": "fr-FR", "sort_by": "popularity.desc", "page": page,
                "with_genres": "|".join(str(g) for g in genre_ids_tuple),
            },
        )
        if data:
            results.extend(data.get("results", []))
    seen, uniq = set(), []
    for m in results:
        if m["id"] != exclude_id and m["id"] not in seen:
            uniq.append(m)
            seen.add(m["id"])
    return uniq[:40]


def poster_img_html(poster_path, title, size_base=TMDB_IMG_MEDIUM):
    if poster_path:
        return f'<img src="{size_base}{poster_path}" class="poster-img" alt="{html.escape(title)}"/>'
    return f'<div class="poster-placeholder"><span>🎞️</span><div>{html.escape(title)}</div></div>'


def year_of(movie):
    date = movie.get("release_date") or ""
    return date[:4] if date else "—"


# ----------------------------------------------------------------------------
# Scoring content-based — calculé par nous, à la volée, sur le bassin renvoyé par TMDB
# ----------------------------------------------------------------------------
def rank_by_content_similarity(query_genre_names, candidates, genre_map, top_n=10):
    """TF-IDF sur les genres + similarité cosinus — même principe que dans le notebook, mais
    réajusté ici sur le petit bassin de candidats à chaque requête (le catalogue change tout le
    temps, l'algorithme reste fixe)."""
    if not candidates:
        return []
    docs = [" ".join(query_genre_names)]
    for c in candidates:
        names = [genre_map.get(gid, "") for gid in c.get("genre_ids", [])]
        docs.append(" ".join(n for n in names if n))

    if not any(docs):
        return []
    try:
        tfidf = TfidfVectorizer().fit_transform(docs)
    except ValueError:
        return []
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    scored = list(zip(candidates, sims))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ----------------------------------------------------------------------------
# Composants d'affichage
# ----------------------------------------------------------------------------
def render_grid(movies_list, columns=5, show_rank=False, scores=None, key_prefix="grid"):
    for start in range(0, len(movies_list), columns):
        row = movies_list[start : start + columns]
        cols = st.columns(columns)
        for j, (col, movie) in enumerate(zip(cols, row)):
            i = start + j
            with col:
                img_html = poster_img_html(movie.get("poster_path"), movie.get("title", ""))
                overlay = ""
                if show_rank:
                    overlay += f'<div class="poster-rank">{i + 1}</div>'
                if scores is not None:
                    overlay += f'<div class="poster-badge">{scores[i]:.0%}</div>'
                st.markdown(f'<div class="poster-wrap">{img_html}{overlay}</div>', unsafe_allow_html=True)
                title = movie.get("title", "Sans titre")
                cap = f"{title} ({year_of(movie)})"
                cap = cap if len(cap) <= 42 else cap[:40] + "…"
                st.markdown(f'<div class="poster-caption">{html.escape(cap)}</div>', unsafe_allow_html=True)
                if st.button("Voir", key=f"{key_prefix}_{movie['id']}", use_container_width=True):
                    st.session_state["selected_movie"] = movie
                    st.rerun()


def render_movie_detail(movie, genre_map):
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        st.markdown(
            f'<div class="detail-poster">{poster_img_html(movie.get("poster_path"), movie.get("title", ""), TMDB_IMG_LARGE)}</div>',
            unsafe_allow_html=True,
        )
        if st.button("✕ Fermer", use_container_width=True):
            st.session_state["selected_movie"] = None
            st.rerun()
    with col2:
        st.markdown(
            f'<div class="detail-title">{html.escape(movie.get("title",""))} '
            f'<span style="color:var(--muted);font-weight:600;">({year_of(movie)})</span></div>',
            unsafe_allow_html=True,
        )
        genre_names = [genre_map.get(gid, "") for gid in movie.get("genre_ids", []) if genre_map.get(gid)]
        chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genre_names)
        st.markdown(f'<div class="detail-genres">{chips}</div>', unsafe_allow_html=True)
        if movie.get("overview"):
            st.markdown(f'<div class="detail-overview">{html.escape(movie["overview"])}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<a class="tmdb-link" href="https://www.themoviedb.org/movie/{movie["id"]}" target="_blank">'
            "Voir la fiche complète sur TMDB ↗</a>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Films similaires <span class="live-tag">live · TMDB</span></div>', unsafe_allow_html=True)

    candidates = tmdb_discover_by_genres(tuple(movie.get("genre_ids", [])), exclude_id=movie["id"])
    if not candidates:
        st.info("Pas assez de films partageant les mêmes genres pour générer des recommandations.")
        return
    genre_names_query = [genre_map.get(gid, "") for gid in movie.get("genre_ids", [])]
    ranked = rank_by_content_similarity(genre_names_query, candidates, genre_map, top_n=10)
    if not ranked:
        st.info("Impossible de calculer un score de similarité pour ce film.")
        return
    ranked_movies = [m for m, _ in ranked]
    ranked_scores = [s for _, s in ranked]
    render_grid(ranked_movies, columns=5, scores=ranked_scores, key_prefix="reco")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_decouvrir():
    st.markdown('<div class="page-eyebrow">Content-based filtering</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Trouve ton prochain film</h1>', unsafe_allow_html=True)

    genre_map = get_genre_map()

    if not get_tmdb_api_key():
        st.error(
            "🔑 Cette application a besoin d'une clé API TMDB pour fonctionner (recherche, "
            "tendances et affiches viennent toutes de TMDB en direct). Ajoute `TMDB_API_KEY` "
            "dans `.streamlit/secrets.toml` (en local) ou dans Settings → Secrets (sur Streamlit "
            "Cloud). Voir le README pour les instructions."
        )
        return

    query = st.text_input("Recherche", placeholder="🔍  Recherche un film, même tout récent…", label_visibility="collapsed")

    if query.strip():
        results = tmdb_search(query)
        st.markdown(f'<div class="section-label">Résultats pour « {html.escape(query)} »</div>', unsafe_allow_html=True)
        if not results:
            st.warning("Aucun film trouvé avec ce titre.")
        else:
            render_grid(results, columns=6, key_prefix="search")
    else:
        st.markdown('<div class="section-label">🔥 Tendances du moment <span class="live-tag">live · TMDB</span></div>', unsafe_allow_html=True)
        popular = tmdb_popular()
        if not popular:
            st.warning("Impossible de récupérer les tendances pour le moment.")
        else:
            render_grid(popular, columns=5, show_rank=True, key_prefix="top")

    sel = st.session_state.get("selected_movie")
    if sel:
        st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
        render_movie_detail(sel, genre_map)


def page_methodo():
    st.markdown('<div class="page-eyebrow">Comment ça marche</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Recommandation par contenu, sur catalogue vivant</h1>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Catalogue", "TMDB (live)")
    c2.metric("Méthode", "TF-IDF + cosinus")
    c3.metric("Calcul du score", "À la volée")

    st.markdown(
        """
        **Le catalogue et l'algorithme sont deux choses séparées.** La recherche, les tendances,
        les genres et les affiches viennent en direct de l'API TMDB — donc toujours à jour, y
        compris pour les films sortis hier. Le score de recommandation, lui, n'est pas calculé
        par TMDB : il est recalculé par nous à chaque recherche, avec la même méthode que dans le
        notebook (TF-IDF sur les genres + similarité cosinus), simplement appliquée à un petit
        bassin de candidats (films partageant au moins un genre, eux aussi récupérés en direct)
        plutôt qu'à une matrice figée à l'avance.

        **Pourquoi pas de filtrage collaboratif ici ?** Un film qui vient de sortir n'a, par
        définition, encore aucune note d'utilisateur — c'est le problème classique du *cold
        start*. Le collaboratif est mathématiquement incapable de le recommander ou de le faire
        apparaître dans des recommandations tant qu'il n'a pas accumulé d'historique. Le
        content-based, lui, n'a besoin que des métadonnées du film (ses genres), disponibles dès
        le jour de sortie — c'est ce qui en fait la seule approche réellement viable pour un
        catalogue qui s'actualise en permanence.

        La comparaison rigoureuse des trois approches (content-based, collaboratif, hybride),
        avec leur évaluation chiffrée (RMSE, Hit Rate@10), reste documentée dans le notebook —
        elle est faite sur le jeu de données statique MovieLens, comme c'est l'usage pour une
        évaluation reproductible en recherche.
        """
    )


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
inject_style()

if "page" not in st.session_state:
    st.session_state.page = "decouvrir"
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎬 CINÉ<b>MATCH</b></div>', unsafe_allow_html=True)
    if st.button("🏠  Découvrir", use_container_width=True, type="primary" if st.session_state.page == "decouvrir" else "secondary"):
        st.session_state.page = "decouvrir"
        st.rerun()
    if st.button("📖  Méthodologie", use_container_width=True, type="primary" if st.session_state.page == "methodo" else "secondary"):
        st.session_state.page = "methodo"
        st.rerun()
    st.markdown("---")
    st.caption("Projet portfolio · Recommandation de films\n\nCatalogue live TMDB · Scoring content-based maison")

if st.session_state.page == "decouvrir":
    page_decouvrir()
else:
    page_methodo()

st.markdown('<div class="credits">CinéMatch — projet portfolio · TMDB live + TF-IDF maison</div>', unsafe_allow_html=True)
