from __future__ import annotations
from tensorflow.keras import layers, models, optimizers

def _compile(model):
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def build_lstm(input_len: int, n_feats: int = 6, units: int = 64, dropout: float = 0.2):
    m = models.Sequential([
        layers.Input(shape=(input_len, n_feats)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    return _compile(m)

def build_gru(input_len: int, n_feats: int = 6, units: int = 64, dropout: float = 0.2):
    m = models.Sequential([
        layers.Input(shape=(input_len, n_feats)),
        layers.GRU(units),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    return _compile(m)

def build_cnn1d(input_len: int, n_feats: int = 6, channels: int = 64, kernel: int = 5, dropout: float = 0.2):
    m = models.Sequential([
        layers.Input(shape=(input_len, n_feats)),
        layers.Conv1D(channels, kernel, padding="causal", activation="relu"),
        layers.Conv1D(channels, kernel, padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    return _compile(m)
