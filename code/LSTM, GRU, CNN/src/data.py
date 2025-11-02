from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, math
import numpy as np
import pandas as pd

# Columns
INPUT_FEATURES: List[str] = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_FEATURE: str = "OT"

# Horizon config: key -> (target_offset_hours, gap_hours)
HORIZONS: Dict[str, Tuple[int, int]] = {
    "t+1h":   (1,   0),
    "t+24h":  (24,  5),
    "t+168h": (168, 23),
}

# ---------- I/O ----------
def load_dataframe(path: str, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
    return df.reset_index(drop=True)

def load_indices(split_dir: str) -> Dict[str, np.ndarray]:
    def _read(name: str) -> np.ndarray:
        p = os.path.join(split_dir, f"indices_{name}.csv")
        return pd.read_csv(p)["row_index"].to_numpy(dtype=int)
    return {"train": _read("train"), "val": _read("val"), "test": _read("test")}

# ------ standardisation ------
def standardize_train_fit(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        mu = float(train_df[c].mean())
        sd = float(train_df[c].std(ddof=0))
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        stats[c] = (mu, sd)
    return stats

def standardize_apply(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    z = df.copy()
    for c, (mu, sd) in stats.items():
        z[c] = (z[c] - mu) / sd
    return z

def inverse_target(z_values: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (z_values * sd) + mu

# -------- window maker --------
def make_windows(
    df_z: pd.DataFrame,
    indices: np.ndarray,
    window: int,
    horizon_key: str,
    input_cols: List[str] = INPUT_FEATURES,
    target_col: str = TARGET_FEATURE,
):
    offset, gap = HORIZONS[horizon_key]
    rows = np.sort(indices)
    if len(rows) == 0:
        return np.empty((0, window, len(input_cols))), np.empty((0,))
    lo, hi = rows[0], rows[-1]

    Xs, ys = [], []
    for t in rows:
        t_gap_end = t - gap
        t_start   = t_gap_end - (window - 1)
        t_target  = t + offset
        if t_start < lo or t_gap_end > hi or t_target > hi:
            continue
        if not np.all(np.in1d(np.arange(t_start, t_gap_end + 1), rows)):
            continue
        X = df_z.loc[t_start:t_gap_end, input_cols].to_numpy(dtype=float)
        y = float(df_z.loc[t_target, target_col])
        Xs.append(X); ys.append(y)

    if not Xs:
        return np.empty((0, window, len(input_cols))), np.empty((0,))
    return np.stack(Xs, axis=0), np.array(ys, dtype=float)
