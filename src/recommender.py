"""
recommender.py
================
Coeur logique du système de recommandation de films.

Ce module est volontairement séparé du notebook et de l'app Streamlit :
c'est une bonne pratique d'ingénierie logicielle (principe DRY - Don't Repeat
Yourself). Le notebook l'utilise pour explorer/évaluer, l'app Streamlit
l'utilise pour servir les recommandations en direct. Un seul endroit à
maintenir.

Trois approches sont implémentées :
1. Content-based   : recommande des films similaires en CONTENU (genres)
2. Collaborative    : recommande en se basant sur le COMPORTEMENT des autres
                      utilisateurs (matrice user x film, factorisée via SVD)
3. Hybride          : combine les deux scores précédents
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR):
    """Charge movies.csv et ratings.csv dans deux DataFrames pandas."""
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    return movies, ratings


# ---------------------------------------------------------------------------
# 2. CONTENT-BASED FILTERING
# ---------------------------------------------------------------------------
#
# Idée : transformer la liste de genres de chaque film ("Comedy|Romance")
# en un vecteur numérique, puis mesurer à quel point deux vecteurs
# "pointent dans la même direction" (similarité cosinus).
#
# TF-IDF (Term Frequency - Inverse Document Frequency) est une technique
# qui vient du traitement du texte : elle donne un poids à chaque mot
# (ici, chaque genre) en fonction de sa fréquence. Un genre très commun
# (ex: "Drama", présent dans des milliers de films) pèsera moins dans la
# comparaison qu'un genre rare (ex: "Film-Noir"), parce qu'un genre rare
# est plus discriminant pour repérer une vraie similarité.

class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        # Remplace les "|" par des espaces pour que TF-IDF traite chaque
        # genre comme un "mot" séparé : "Comedy|Romance" -> "Comedy Romance"
        genres_text = self.movies["genres"].fillna("").str.replace("|", " ", regex=False)

        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(genres_text)

        # Matrice de similarité cosinus entre TOUS les films, deux à deux.
        # similarity[i][j] = à quel point le film i ressemble au film j (0 à 1)
        self.similarity = cosine_similarity(self.tfidf_matrix)

        # Pour retrouver l'index d'un film à partir de son titre
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"]
        )

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Retourne les top_n films les plus similaires (en genres) à `title`."""
        if title not in self.title_to_index:
            raise ValueError(f"Film inconnu : {title!r}")

        idx = self.title_to_index[title]
        # On récupère les scores de similarité du film avec tous les autres
        scores = list(enumerate(self.similarity[idx]))
        # Tri par score décroissant, on exclut le film lui-même (score=1.0)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        indices = [i for i, _ in scores]
        result = self.movies.iloc[indices][["movieId", "title", "genres"]].copy()
        result["score_similarite"] = [s for _, s in scores]
        return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. COLLABORATIVE FILTERING (SVD)
# ---------------------------------------------------------------------------
#
# Idée : on construit une immense matrice (utilisateurs x films) où chaque
# case contient la note donnée par un utilisateur à un film (la plupart des
# cases sont vides, car personne n'a vu tous les films). La factorisation
# matricielle (SVD = Singular Value Decomposition) "compresse" cette matrice
# en quelques dizaines de facteurs latents par utilisateur et par film
# (des goûts abstraits, pas directement interprétables, un peu comme des
# "axes de personnalité cinématographique"). En recombinant ces facteurs,
# le modèle peut PRÉDIRE la note qu'un utilisateur donnerait à un film
# qu'il n'a jamais vu.

class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.movies = movies

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

        # On entraîne sur 80% des notes, on garde 20% pour évaluer le modèle
        self.trainset, self.testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

        self.model = SVD(n_factors=50, random_state=42)
        self.model.fit(self.trainset)

    def evaluate(self) -> float:
        """Retourne le RMSE (Root Mean Squared Error) sur le jeu de test.
        Plus c'est bas, mieux le modèle prédit les notes réelles."""
        predictions = self.model.test(self.testset)
        return accuracy.rmse(predictions, verbose=False)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Prédit la note de l'utilisateur pour tous les films qu'il n'a pas
        encore notés, et retourne les top_n films avec la meilleure note prédite."""
        rated_movie_ids = set(self.ratings.loc[self.ratings.userId == user_id, "movieId"])
        all_movie_ids = set(self.movies["movieId"])
        unrated = all_movie_ids - rated_movie_ids

        predictions = [
            (movie_id, self.model.predict(user_id, movie_id).est)
            for movie_id in unrated
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:top_n]

        result = self.movies.set_index("movieId").loc[[m for m, _ in top]][["title", "genres"]].copy()
        result = result.reset_index()
        result["note_predite"] = [round(score, 2) for _, score in top]
        return result


# ---------------------------------------------------------------------------
# 4. HYBRIDE
# ---------------------------------------------------------------------------
#
# Idée simple et efficace : on prend les films déjà bien notés (prédiction
# collaborative) par l'utilisateur, PUIS on les re-classe en tenant compte
# de leur ressemblance de contenu avec ce que l'utilisateur a déjà aimé.
# Cela évite les deux travers : le collaboratif seul peut suggérer un film
# culte mais hors-sujet, le content-based seul peut tourner en rond sur
# toujours le même genre.

class HybridRecommender:
    def __init__(self, content_model: ContentBasedRecommender, collab_model: CollaborativeRecommender):
        self.content_model = content_model
        self.collab_model = collab_model

    def recommend(self, user_id: int, top_n: int = 10, n_candidates: int = 50, alpha: float = 0.6) -> pd.DataFrame:
        """
        alpha : poids donné au score collaboratif (entre 0 et 1).
                (1 - alpha) est le poids donné au score de contenu.
        """
        # Étape 1 : on prend un plus large pool de candidats côté collaboratif
        candidates = self.collab_model.recommend(user_id, top_n=n_candidates)

        # Étape 2 : on récupère les films préférés de l'utilisateur (notes >= 4)
        # pour calculer une similarité moyenne de contenu avec ces favoris
        liked_ids = self.collab_model.ratings.loc[
            (self.collab_model.ratings.userId == user_id) & (self.collab_model.ratings.rating >= 4),
            "movieId",
        ]
        liked_titles = self.collab_model.movies.set_index("movieId").loc[
            self.collab_model.movies.set_index("movieId").index.intersection(liked_ids)
        ]["title"].tolist()

        title_to_idx = self.content_model.title_to_index

        def content_score(title):
            if title not in title_to_idx or not liked_titles:
                return 0.0
            idx = title_to_idx[title]
            sims = [
                self.content_model.similarity[idx][title_to_idx[t]]
                for t in liked_titles if t in title_to_idx
            ]
            return float(np.mean(sims)) if sims else 0.0

        # Normalisation simple de la note prédite collaborative (0.5-5 -> 0-1)
        candidates["score_collaboratif_norm"] = (candidates["note_predite"] - 0.5) / 4.5
        candidates["score_contenu"] = candidates["title"].apply(content_score)
        candidates["score_hybride"] = (
            alpha * candidates["score_collaboratif_norm"] + (1 - alpha) * candidates["score_contenu"]
        )

        result = candidates.sort_values("score_hybride", ascending=False).head(top_n)
        return result[["movieId", "title", "genres", "note_predite", "score_contenu", "score_hybride"]].reset_index(drop=True)
