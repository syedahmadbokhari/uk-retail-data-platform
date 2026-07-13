"""
Hyperparameter tuning for K-means clustering using Optuna.

Searches for the optimal number of clusters (k) by maximising the
silhouette score — a measure of how well each point fits its own
cluster vs the nearest neighbouring cluster (range: -1 to 1, higher is better).

Usage:
    python -m src.tune_clusters
"""

import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.db import get_connection
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger("tune_clusters")

CLUSTER_COLS = ["revenue", "listing_price", "discount", "rating"]
N_TRIALS = 20


def objective(trial: optuna.Trial, X) -> float:
    k = trial.suggest_int("n_clusters", 2, 8)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return silhouette_score(X, labels)


def tune() -> None:
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM features_products", conn)

    X = StandardScaler().fit_transform(df[CLUSTER_COLS].dropna())

    print("\n" + "=" * 55)
    print("  Optuna K-means Hyperparameter Tuning")
    print(f"  Metric: Silhouette Score (higher = better)")
    print(f"  Search space: n_clusters in [2, 8]")
    print(f"  Trials: {N_TRIALS}")
    print("=" * 55)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X), n_trials=N_TRIALS)

    print(f"\n{'Trial':>6}  {'k':>4}  {'Silhouette Score':>18}")
    print("-" * 35)
    for t in sorted(study.trials, key=lambda t: t.number):
        marker = " <-- best" if t.number == study.best_trial.number else ""
        print(f"{t.number:>6}  {t.params['n_clusters']:>4}  {t.value:>18.4f}{marker}")

    best = study.best_trial
    print("\n" + "=" * 55)
    print(f"  Best trial:        #{best.number}")
    print(f"  Best n_clusters:   {best.params['n_clusters']}")
    print(f"  Best silhouette:   {best.value:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    tune()
