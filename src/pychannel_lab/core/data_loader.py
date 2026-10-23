"""
Utilities for loading experimental CSV data.
CSV format: header row (used as axis labels), then columns x, y[, y_err].
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def _parse(source) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    # The header row is skipped as raw text rather than parsed, so a header
    # with a different separator (e.g. "x (mV).y.SD") does not collapse the
    # table to a single column.
    df = pd.read_csv(source, header=None, skiprows=1)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least 2 columns, got {df.shape[1]}")
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    y_err = df.iloc[:, 2].to_numpy(dtype=float) if df.shape[1] >= 3 else None
    return x, y, y_err


def load_csv(
    filepath: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a two- or three-column CSV file.

    Returns
    -------
    x : ndarray
    y : ndarray
    y_err : ndarray or None
    """
    return _parse(filepath)


def load_from_bytes(
    content: bytes, filename: str = ""
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load from raw bytes (e.g. Streamlit UploadedFile.read())."""
    import io

    return _parse(io.BytesIO(content))
