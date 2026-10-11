"""I/O utilities for data handling."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

__all__ = [
    "save_json", "load_json",
    "save_df", "load_df",
    "save_parquet", "load_parquet",
    "save_pickle", "load_pickle",
    "log",
]


def log(msg: str) -> None:
    """Simple logging function."""
    print(f"[geospatial] {msg}")


def save_json(obj, path: Path | str) -> None:
    """Save object as JSON with numpy type conversion."""
    import numpy as np
    
    def convert_numpy_types(o):
        """Recursively convert numpy types to native Python types."""
        if isinstance(o, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64, np.float32, np.float16)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            return {k: convert_numpy_types(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [convert_numpy_types(item) for item in o]
        elif pd.isna(o):
            return None
        return o
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted_obj = convert_numpy_types(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted_obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path | str):
    """Load object from JSON."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_df(df: pd.DataFrame, path: Path | str) -> None:
    """Save DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def load_df(path: Path | str) -> pd.DataFrame:
    """Load DataFrame from parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, engine="pyarrow")


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Alias for save_df - save DataFrame to parquet."""
    save_df(df, path)


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Alias for load_df - load DataFrame from parquet."""
    return load_df(path)


def save_pickle(obj, path: Path | str) -> None:
    """Save object as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Path | str):
    """Load object from pickle."""
    return joblib.load(path)

