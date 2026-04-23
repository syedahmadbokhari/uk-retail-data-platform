import os
import sqlite3
from contextlib import contextmanager

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_ROOT, "data", "retailDB.sqlite")


def get_root() -> str:
    return _ROOT


def get_db_path() -> str:
    return _DB_PATH


@contextmanager
def get_connection(db_path: str = None):
    path = db_path or _DB_PATH
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
