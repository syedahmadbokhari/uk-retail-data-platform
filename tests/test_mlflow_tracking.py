"""
Unit tests for src/tune_clusters.py MLflow integration and
src/compare_mlflow_runs.py.

Uses a REAL local MLflow tracking store (SQLite, in a pytest tmp_path) rather
than mocking mlflow itself — this exercises the actual integration (params,
metrics, model artifacts, and Model Registry all really persist and can be
queried back), while the database layer (get_connection/pd.read_sql) is
mocked with synthetic data, consistent with the existing test patterns in
this repo (see test_clustering.py). No live/remote MLflow server is used.

mlflow.sklearn.log_model() takes ~10s per call in this environment even with
pip_requirements pinned (see tune_clusters.py's comment on this) — tests are
consolidated where it doesn't cost test clarity, to keep this file's total
runtime reasonable rather than logging redundant runs per assertion.
"""
import numpy as np
import pandas as pd
import pytest
import mlflow
from sklearn.cluster import KMeans
from unittest.mock import patch, MagicMock

from src.tune_clusters import (
    tune,
    configure_mlflow,
    register_best_model,
    _get_best_run,
)
from src.compare_mlflow_runs import get_all_runs, print_comparison

# Same shape as test_clustering.py's fixture — enough rows and spread for
# KMeans with k up to 8 to fit without error.
_SYNTHETIC = pd.DataFrame({
    "product_id":    [f"P{i:03d}" for i in range(1, 13)],
    "brand_encoded": [0, 1] * 6,
    "listing_price": [50, 55, 52, 105, 110, 108, 200, 205, 210, 195, 198, 202],
    "discount":      [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "revenue":       [2.0, 2.1, 1.9, 5.0, 5.1, 4.9, 9.0, 9.1, 8.9, 8.8, 9.2, 9.0],
    "rating":        [3.0, 3.1, 2.9, 4.0, 4.1, 3.9, 4.9, 5.0, 4.8, 4.7, 5.0, 4.9],
})


@pytest.fixture(autouse=True)
def _local_mlflow_store(tmp_path, monkeypatch):
    """Every test gets its own throwaway local SQLite tracking store."""
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setattr("src.tune_clusters.MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    yield


def _run_tune(n_trials, experiment_name):
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.tune_clusters.get_connection", return_value=mock_conn), \
         patch("src.tune_clusters.pd.read_sql", return_value=_SYNTHETIC.copy()), \
         patch("src.tune_clusters.MLFLOW_EXPERIMENT_NAME", experiment_name):
        return tune(n_trials=n_trials)


def _log_manual_run(experiment_name, run_name, n_clusters, score):
    configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", score)
        mlflow.sklearn.log_model(
            KMeans(n_clusters=2, n_init=1).fit(np.random.rand(6, 2)),
            name="model", pip_requirements=["scikit-learn"],
        )


# ── Per-trial logging (one real tune() run, several assertions on it) ────────

def test_tune_logs_params_metrics_and_model_artifact_per_trial():
    experiment_name = "exp_tune_end_to_end"
    _run_tune(n_trials=2, experiment_name=experiment_name)

    runs = mlflow.search_runs(experiment_names=[experiment_name])
    assert len(runs) == 2

    n_clusters = runs["params.n_clusters"].astype(int)
    assert n_clusters.between(2, 8).all()

    assert runs["metrics.silhouette_score"].notna().all()
    assert runs["metrics.silhouette_score"].between(-1, 1).all()

    # MLflow 3.x logs models as a decoupled "LoggedModel" entity (not a
    # traditional run-artifact file), linked to the run via run.outputs —
    # see module docstring / development notes on the models:/m-<id> scheme.
    client = mlflow.MlflowClient()
    for run_id in runs["run_id"]:
        model_outputs = client.get_run(run_id).outputs.model_outputs
        assert model_outputs, f"run {run_id} has no logged model"


# ── Best-model selection + registration ───────────────────────────────────────

def test_get_best_run_picks_highest_silhouette_score():
    exp = "exp_best_selection"
    _log_manual_run(exp, "low", n_clusters=2, score=0.3)
    _log_manual_run(exp, "high", n_clusters=5, score=0.7)
    _log_manual_run(exp, "mid", n_clusters=7, score=0.5)

    best = _get_best_run(experiment_name=exp)
    assert best["params.n_clusters"] == "5"
    assert best["metrics.silhouette_score"] == pytest.approx(0.7)


def test_register_best_model_registers_and_tags_the_top_scoring_run():
    exp = "exp_register"
    _log_manual_run(exp, "low", n_clusters=2, score=0.2)
    _log_manual_run(exp, "high", n_clusters=6, score=0.9)

    version = register_best_model(experiment_name=exp, model_name="test_model_register")

    client = mlflow.MlflowClient()
    mv = client.get_model_version("test_model_register", version)
    source_run = client.get_run(mv.run_id)

    assert source_run.data.params["n_clusters"] == "6"
    assert mv.tags.get("stage") == "production"


def test_get_best_run_raises_clearly_when_no_runs_exist():
    configure_mlflow()
    mlflow.set_experiment("exp_empty_for_best_run")
    with pytest.raises(RuntimeError):
        _get_best_run(experiment_name="exp_empty_for_best_run")


# ── compare_mlflow_runs.py ────────────────────────────────────────────────────

def test_get_all_runs_sorted_best_first():
    exp = "exp_compare"
    _log_manual_run(exp, "low", n_clusters=2, score=0.1)
    _log_manual_run(exp, "high", n_clusters=6, score=0.8)
    _log_manual_run(exp, "mid", n_clusters=4, score=0.4)

    df = get_all_runs(experiment_name=exp)

    assert list(df["silhouette_score"].astype(float)) == sorted(df["silhouette_score"].astype(float), reverse=True)
    assert df.iloc[0]["n_clusters"] == "6"


def test_print_comparison_handles_no_runs_gracefully():
    configure_mlflow()
    mlflow.set_experiment("exp_never_used")
    result = print_comparison(experiment_name="exp_never_used")
    assert result.empty
