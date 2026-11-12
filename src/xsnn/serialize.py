"""JSON conversion for numpy and pandas scalars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def json_ready(value: Any) -> Any:
    """Recursively convert values that ``json.dumps`` would reject or emit as NaN."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value
