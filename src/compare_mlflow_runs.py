"""
Queries the MLflow tracking store to compare all logged K-means tuning
trials — genuinely runnable, not just described.

Usage:
    python -m src.compare_mlflow_runs
"""
import pandas as pd
import mlflow

from src.utils.logger import get_logger
from src.tune_clusters import configure_mlflow, MLFLOW_EXPERIMENT_NAME

logger = get_logger("compare_mlflow_runs")


def get_all_runs(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """
    Returns every logged trial for experiment_name as a DataFrame — trial name,
    n_clusters, silhouette_score — sorted best (highest silhouette) first.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.silhouette_score DESC"],
    )
    if runs.empty:
        return pd.DataFrame(columns=["run_name", "n_clusters", "silhouette_score"])

    return pd.DataFrame({
        "run_name":         runs["tags.mlflow.runName"],
        "n_clusters":       runs["params.n_clusters"],
        "silhouette_score": runs["metrics.silhouette_score"],
    }).reset_index(drop=True)


def print_comparison(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    configure_mlflow()
    df = get_all_runs(experiment_name)

    if df.empty:
        logger.info(f"No runs found in experiment '{experiment_name}' — run `python -m src.tune_clusters` first.")
        return df

    print(f"\nAll logged trials — experiment '{experiment_name}' (best first):\n")
    print(df.to_string(index=False))
    print(f"\nBest: n_clusters={df.iloc[0]['n_clusters']}, silhouette={float(df.iloc[0]['silhouette_score']):.4f}\n")
    return df


if __name__ == "__main__":
    print_comparison()
