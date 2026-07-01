# 🎬 CinéMatch — Recommandation de films par contenu

Application de recommandation de films avec affiches, inspirée des interfaces de sites de
streaming, construite sur le jeu de données
[MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/) (100 836 notes,
9 742 films, 610 utilisateurs).

**L'application déployée se concentre sur l'approche content-based** (recherche d'un film,
affichage de son affiche, recommandations de films aux genres similaires) : c'est celle qui se
prête le mieux à une expérience visuelle et explicable. Le **notebook**, lui, garde la
comparaison complète des trois approches (content-based, collaboratif, hybride) avec leur
évaluation chiffrée — c'est ce qu'il faut montrer pour prouver la rigueur méthodologique du
projet au-delà de l'app elle-même.

## Approches implémentées

| Approche | Où la voir | Principe | Force | Limite |
|---|---|---|---|---|
| **Content-based** | App + notebook | TF-IDF sur les genres + similarité cosinus | Disponible dès le 1er jour, explicable, base de l'app déployée | Reste limité aux genres comme métadonnées |
| **Collaborative filtering** | Notebook | SVD (Surprise) sur la matrice utilisateur × film | RMSE ≈ 0.88, capture des goûts fins | Cold start (nouvel utilisateur/film) |
| **Hybride** | Notebook | Combinaison pondérée des deux scores | Cohérence thématique + personnalisation | Hit Rate@10 plus faible (compromis volontaire) |

## Affiches de films (API TMDB)

Les affiches viennent de [TMDB](https://www.themoviedb.org/) (The Movie Database), gratuit pour
un usage non commercial. Sans clé configurée, l'app fonctionne quand même : un visuel de
remplacement s'affiche à la place de l'affiche.

**Obtenir une clé (2 minutes)** :
1. Crée un compte gratuit sur [themoviedb.org](https://www.themoviedb.org/signup).
2. Une fois connecté, va dans *Paramètres → API* puis *Créer* une clé API ("API Key (v3 auth)").
3. Accepte les conditions (usage non commercial), remplis le formulaire (tu peux mettre "projet
   portfolio personnel / étudiant" comme description).

**Configurer la clé en local** :
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# puis édite .streamlit/secrets.toml et remplace par ta vraie clé
```

**Configurer la clé sur Streamlit Community Cloud** : dans le tableau de bord de ton app, va dans
*Settings → Secrets*, et colle :
```toml
TMDB_API_KEY = "ta_cle_ici"
```
Ce fichier `secrets.toml` ne doit **jamais** être poussé sur GitHub (il est exclu par le
`.gitignore`) — c'est pour ça qu'on le configure séparément en local et sur Streamlit Cloud.

## Structure du projet

```
movie-recommender/
├── data/                              # Dataset MovieLens (CSV : movies, ratings, links, tags)
├── notebook/
│   ├── recommandation_films.py        # Source du notebook (format jupytext, lisible en diff Git)
│   └── recommandation_films.ipynb     # Notebook exécuté : les 3 approches + évaluation complète
├── models/                            # Artefacts générés par le notebook, utilisés par l'app
│   ├── tfidf_matrix.npz                # Matrice TF-IDF creuse (utilisée par l'app)
│   ├── svd_model.pkl                   # Modèle collaboratif (utilisé par le notebook uniquement)
│   ├── movies.pkl / ratings.pkl
│   └── eda_distributions.png
├── app/
│   └── app.py                         # Application Streamlit (content-based + affiches TMDB)
├── .streamlit/
│   ├── config.toml                    # Thème (couleurs)
│   └── secrets.toml.example           # Modèle pour ta clé TMDB (à copier, jamais à committer)
├── requirements.txt
└── README.md
```

## Installation et lancement en local

```bash
# 1. Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé TMDB (voir section ci-dessus) — optionnel mais recommandé
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 4a. Ouvrir le notebook (les 3 approches, évaluation complète)
jupyter notebook notebook/recommandation_films.ipynb

# 4b. Ou lancer l'app interactive
streamlit run app/app.py
```

## Déploiement de l'app (gratuit)

1. Pousse ce dépôt sur GitHub (le dossier `models/` doit être inclus, `secrets.toml` ne doit
   PAS l'être — il ne l'est pas par défaut grâce au `.gitignore`).
2. Sur [Streamlit Community Cloud](https://share.streamlit.io), connecte le dépôt et indique
   `app/app.py` comme fichier principal.
3. Une fois l'app créée, va dans *Settings → Secrets* et ajoute ta clé `TMDB_API_KEY` (voir
   section ci-dessus).
4. Redéploie (ou attends le prochain `git push`) pour que les affiches s'affichent.

## Pour aller plus loin

- Enrichir le content-based avec les résumés (déjà disponibles via TMDB) plutôt que les seuls
  genres.
- Ajouter une page "Genres" pour parcourir le catalogue par catégorie.
- Mettre en cache les chemins d'affiches sur disque pour limiter les appels à l'API TMDB.
- Réintégrer le collaboratif/hybride dans l'app pour les utilisateurs existants du dataset.

## Source des données

MovieLens ml-latest-small, [GroupLens Research](https://grouplens.org/datasets/movielens/),
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

Affiches et métadonnées additionnelles : [TMDB](https://www.themoviedb.org/) (ce produit utilise
l'API TMDB mais n'est ni approuvé ni certifié par TMDB).
