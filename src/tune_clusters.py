"""
Hyperparameter tuning for K-means clustering using Optuna, tracked with MLflow.

Searches for the optimal number of clusters (k) by maximising the
silhouette score — a measure of how well each point fits its own
cluster vs the nearest neighbouring cluster (range: -1 to 1, higher is better).

MLflow tracking
---------------
Each Optuna trial is logged as its own MLflow run: the tuned parameter
(n_clusters), the optimisation metric (silhouette_score), and the fitted
KMeans model itself as an artifact — previously this study's results only
ever existed in the console output of whoever ran it, with no history and
no saved model. After the study finishes, the run with the best silhouette
score is registered as a new version of REGISTERED_MODEL_NAME and tagged
stage=production, so there is a single, unambiguous record of which
version is the one meant to be used downstream (build_clusters() in
clustering.py does not yet consume this registered model automatically —
see the README's Model Tracking section for why that wiring is a
follow-up, not part of this change).

Tracking store: a local SQLite file (mlruns/mlflow.db), not a remote
server — consistent with this project's local/CI-friendly philosophy.
MLflow's plain filesystem store is in maintenance mode as of MLflow 3.x
and no longer supports the Model Registry, so SQLite is the local-only
option that still supports registration.

Usage:
    python -m src.tune_clusters
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # view results
"""
import os

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.db import get_connection, get_root
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger("tune_clusters")

CLUSTER_COLS = ["revenue", "listing_price", "discount", "rating"]
N_TRIALS = 20

_MLFLOW_DB_PATH = os.path.join(get_root(), "mlruns", "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{_MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT_NAME = "kmeans_cluster_tuning"
REGISTERED_MODEL_NAME = "kmeans_cluster_model"


def configure_mlflow(tracking_uri: str = None) -> None:
    """
    Points MLflow at the local SQLite tracking store and selects the experiment.

    tracking_uri defaults to the current value of the module-level
    MLFLOW_TRACKING_URI (read at call time, not bound at import time) so
    that tests can point this at a temp file by monkeypatching that
    module attribute before calling configure_mlflow() with no argument.
    """
    # Telemetry is a network call this project has no use for, and it adds
    # real latency to every run — see the ~28s -> ~11s log_model timing
    # difference measured during development.
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

    tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri[len("sqlite:///"):]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def objective(trial: optuna.Trial, X) -> float:
    k = trial.suggest_int("n_clusters", 2, 8)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_param("n_clusters", k)
        mlflow.log_metric("silhouette_score", score)
        # Explicit pip_requirements skips log_model's default dependency
        # auto-inference, which otherwise re-scans the environment on every
        # single trial (~28s each — 20 trials would add ~9 minutes of pure
        # logging overhead with no benefit, since the environment is fixed).
        mlflow.sklearn.log_model(km, name="model", pip_requirements=["scikit-learn"])

    return score


def _get_best_run(experiment_name: str = None) -> pd.Series:
    """
    Returns the logged run with the highest silhouette_score.

    experiment_name defaults to the current value of MLFLOW_EXPERIMENT_NAME
    (read at call time, not bound at import time) — same reasoning as
    configure_mlflow()'s tracking_uri default.
    """
    experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.silhouette_score DESC"],
    )
    if runs.empty:
        raise RuntimeError(f"No MLflow runs found in experiment '{experiment_name}' — run tune() first.")
    return runs.iloc[0]


def register_best_model(experiment_name: str = None, model_name: str = None) -> str:
    """
    Registers the best-scoring run's model as a new version of model_name,
    tagged stage=production. Returns the new version number (as a string).
    """
    model_name = model_name or REGISTERED_MODEL_NAME
    best_run = _get_best_run(experiment_name)
    model_uri = f"runs:/{best_run['run_id']}/model"

    result = mlflow.register_model(model_uri, model_name)

    client = mlflow.MlflowClient()
    client.set_model_version_tag(model_name, result.version, "stage", "production")

    logger.info(
        f"Registered {model_name} v{result.version} (run {best_run['run_id']}, "
        f"n_clusters={int(best_run['params.n_clusters'])}, "
        f"silhouette={best_run['metrics.silhouette_score']:.4f}) as production"
    )
    return result.version


def tune(n_trials: int = N_TRIALS) -> optuna.Study:
    configure_mlflow()

    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM features_products", conn)

    X = StandardScaler().fit_transform(df[CLUSTER_COLS].dropna())

    print("\n" + "=" * 55)
    print("  Optuna K-means Hyperparameter Tuning (tracked in MLflow)")
    print(f"  Metric: Silhouette Score (higher = better)")
    print(f"  Search space: n_clusters in [2, 8]")
    print(f"  Trials: {n_trials}")
    print("=" * 55)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X), n_trials=n_trials)

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

    version = register_best_model()
    print(f"  Registered as {REGISTERED_MODEL_NAME} v{version} (stage=production)")
    print(f"  View all runs: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}\n")

    return study


if __name__ == "__main__":
    tune()
