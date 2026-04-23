import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.db import get_connection, get_root
from src.utils.logger import get_logger

logger = get_logger("recommender")

_MODELS_DIR = os.path.join(get_root(), "models")
SIMILARITY_PATH = os.path.join(_MODELS_DIR, "similarity.pkl")

FEATURE_COLS = ["brand_encoded", "listing_price", "discount", "revenue", "rating", "review_count"]


def build_similarity_matrix() -> tuple:
    start = time.time()
    logger.info("=== Building similarity matrix ===")

    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM features_products", conn)

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    logger.info(f"Products for similarity: {len(df)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    matrix = cosine_similarity(X)

    os.makedirs(_MODELS_DIR, exist_ok=True)
    artifact = {"matrix": matrix, "df": df, "scaler": scaler}
    with open(SIMILARITY_PATH, "wb") as f:
        pickle.dump(artifact, f)

    logger.info(
        f"Similarity matrix shape: {matrix.shape} — "
        f"saved to {SIMILARITY_PATH} — {time.time() - start:.2f}s"
    )
    return matrix, df


def load_similarity_artifact() -> dict:
    if not os.path.exists(SIMILARITY_PATH):
        raise FileNotFoundError(
            f"Model not found at {SIMILARITY_PATH}. Run: python pipeline/run_pipeline.py"
        )
    with open(SIMILARITY_PATH, "rb") as f:
        return pickle.load(f)


def get_recommendations(
    product_id: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    matches = df.index[df["product_id"] == product_id].tolist()
    if not matches:
        logger.warning(f"product_id '{product_id}' not found in feature table")
        return pd.DataFrame()

    idx = matches[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = [(i, s) for i, s in scores if i != idx][:top_n]

    results = []
    for i, score in top:
        row = df.iloc[i]
        results.append({
            "product_name":     row["product_name"],
            "brand":            row["brand"],
            "listing_price":    round(float(row["listing_price"]), 2),
            "rating":           round(float(row["rating"]), 2),
            "revenue":          round(float(row["revenue"]), 2),
            "similarity_score": round(float(score), 4),
        })

    logger.info(f"Recommendations for '{product_id}': {len(results)} results")
    return pd.DataFrame(results)
