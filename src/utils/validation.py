import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("validation")


class ValidationError(Exception):
    pass


def check_row_count(df: pd.DataFrame, name: str, min_rows: int = 1) -> int:
    n = len(df)
    if n < min_rows:
        raise ValidationError(f"{name}: {n} rows found, need >= {min_rows}")
    logger.info(f"{name}: {n} rows")
    return n


def check_missing(df: pd.DataFrame, name: str, critical: list = None):
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    if missing:
        logger.warning(f"{name} missing values: {missing}")
    if critical:
        for col in critical:
            if col in df.columns and df[col].isnull().any():
                raise ValidationError(
                    f"Critical column '{col}' in {name} contains nulls — pipeline aborted"
                )


def check_duplicates(df: pd.DataFrame, name: str, keys: list) -> int:
    n = int(df.duplicated(subset=keys).sum())
    if n > 0:
        logger.warning(f"{name}: {n} duplicate rows on keys {keys}")
    return n


def validate(
    df: pd.DataFrame,
    name: str,
    critical_cols: list = None,
    key_cols: list = None,
    min_rows: int = 1,
):
    check_row_count(df, name, min_rows)
    check_missing(df, name, critical_cols)
    if key_cols:
        check_duplicates(df, name, key_cols)
