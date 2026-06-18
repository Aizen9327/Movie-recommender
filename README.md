# 🎬 Système de recommandation de films — MovieLens

Projet portfolio démontrant trois approches classiques de systèmes de recommandation sur le
jeu de données [MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/)
(GroupLens Research, Université du Minnesota) : 100 836 notes, 9 742 films, 610 utilisateurs.

## Approches implémentées

| Approche | Principe | Force principale | Limite principale |
|---|---|---|---|
| **Content-based** | TF-IDF sur les genres + similarité cosinus entre films | Fonctionne dès le premier jour, explicable | Reste enfermé dans les métadonnées (ici, seulement les genres) |
| **Collaborative filtering** | Factorisation matricielle (SVD, via [Surprise](http://surpriselib.com/)) sur la matrice utilisateur × film | Capture des goûts fins, RMSE ≈ 0.88 | Problème du *cold start* (nouvel utilisateur ou nouveau film) |
| **Hybride** | Combinaison pondérée des deux scores précédents | Cohérence thématique + personnalisation, recommandations explicables | Hit Rate@10 plus faible que le collaboratif seul (compromis volontaire, voir notebook) |

Le notebook (`notebook/recommandation_films.ipynb`) détaille la méthodologie complète,
étape par étape, avec une vraie évaluation chiffrée (RMSE, MAE, Hit Rate@10 comparé entre
collaboratif et hybride) — pas seulement le code, mais le raisonnement et les compromis derrière
chaque choix.

## Résultats clés

- **RMSE** (collaboratif, SVD, split 80/20) : **0.877** — **MAE** : 0.674
- **Hit Rate@10** : 35.0% (collaboratif seul) vs 20.0% (hybride, poids contenu = 0.4)
- L'écart entre les deux Hit Rate s'explique et se discute dans le notebook (section 5) : ce
  n'est pas un bug, c'est le prix payé pour des recommandations plus explicables.

## Structure du projet

```
movie-recommender/
├── data/                              # Dataset MovieLens (CSV : movies, ratings, links, tags)
├── notebook/
│   ├── recommandation_films.py        # Source du notebook (format jupytext, lisible en diff Git)
│   └── recommandation_films.ipynb     # Notebook exécuté, avec tous les résultats et graphiques
├── models/                            # Artefacts générés par le notebook, utilisés par l'app
│   ├── svd_model.pkl                  # Modèle collaboratif entraîné (~7 Mo)
│   ├── tfidf_matrix.npz               # Matrice TF-IDF creuse (~80 Ko, PAS la matrice NxN complète)
│   ├── movies.pkl / ratings.pkl
│   └── eda_distributions.png
├── app/
│   └── app.py                         # Application Streamlit interactive
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

# 3a. Ouvrir le notebook
jupyter notebook notebook/recommandation_films.ipynb

# 3b. Ou lancer directement l'app interactive (les artefacts dans models/ sont déjà fournis)
streamlit run app/app.py
```

Si tu modifies le notebook (`.py` ou `.ipynb`), régénère les artefacts pour l'app avec :

```bash
cd notebook
jupytext --to notebook recommandation_films.py
jupyter nbconvert --to notebook --execute --inplace recommandation_films.ipynb
```

## Déploiement de l'app (gratuit)

L'app est conçue pour être déployée directement sur
[Streamlit Community Cloud](https://streamlit.io/cloud) :

1. Pousse ce dépôt sur GitHub (le dossier `models/` doit être inclus — l'app en a besoin pour
   fonctionner, pas seulement le code).
2. Sur Streamlit Community Cloud, connecte le dépôt et indique `app/app.py` comme fichier
   principal.
3. Le déploiement installe automatiquement `requirements.txt`. Note : `scikit-surprise` nécessite
   une compilation à l'installation, ce qui peut rendre le premier déploiement un peu plus lent
   (quelques minutes) — c'est normal.

## Pour aller plus loin

- Utiliser des métadonnées plus riches en content-based (résumés via l'API TMDB, acteurs,
  réalisateur) plutôt que les seuls genres.
- Tester d'autres algorithmes collaboratifs (NMF, embeddings par réseau de neurones).
- Gérer explicitement le cold start (nouvel utilisateur → recommandations par popularité, ou
  questionnaire de goûts initial).
- Apprendre la pondération hybride plutôt que la fixer manuellement (ex : régression logistique
  sur un jeu de validation).
- Introduire une notion de diversité dans le top-N pour éviter de recommander dix films presque
  identiques.

## Source des données

MovieLens ml-latest-small, [GroupLens Research](https://grouplens.org/datasets/movielens/),
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
