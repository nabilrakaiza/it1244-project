from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def callbacks(save_path: str):
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True),
    ]

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    r2   = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
