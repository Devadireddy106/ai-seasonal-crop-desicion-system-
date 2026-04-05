"""
src/lstm_model.py
LSTM-based seasonal price forecasting model.
Trains one LSTM per crop and saves to models/ directory.
"""
import numpy as np
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── Try importing TensorFlow, fall back to sklearn if unavailable ─────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


# ── LSTM Architecture ─────────────────────────────────────────────────────────
def build_lstm(seq_len=12, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(seq_len, 1)),
        Dropout(dropout),
        LSTM(32, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── Fallback: Ridge Regression as LSTM surrogate ─────────────────────────────
def build_ridge_surrogate(X_train, y_train, crop_name):
    """Used when TensorFlow is not available."""
    X_flat = X_train.reshape(X_train.shape[0], -1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_flat, y_train)
    joblib.dump(ridge, MODEL_DIR / f"lstm_{crop_name.lower()}.pkl")
    return ridge


# ── Training ──────────────────────────────────────────────────────────────────
def train(crop_name, epochs=60, batch_size=16, seq_len=12):
    from src.preprocessor import prepare_lstm_data
    print(f"\n{'='*50}")
    print(f"  Training LSTM for {crop_name}")
    print(f"{'='*50}")

    X_tr, y_tr, X_te, y_te, scaler, test_dates = prepare_lstm_data(crop_name)

    if TF_AVAILABLE:
        model = build_lstm(seq_len)
        model.summary()
        cb = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(MODEL_DIR / f"lstm_{crop_name.lower()}.keras"),
                            save_best_only=True, verbose=0)
        ]
        history = model.fit(X_tr, y_tr, validation_split=0.15,
                            epochs=epochs, batch_size=batch_size,
                            callbacks=cb, verbose=1)
        model.save(str(MODEL_DIR / f"lstm_{crop_name.lower()}.keras"))
    else:
        print("  ⚠ TensorFlow not available — using Ridge surrogate")
        model = build_ridge_surrogate(X_tr, y_tr, crop_name)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(model, X_te, y_te, scaler, crop_name, TF_AVAILABLE)
    print(f"  RMSE: {metrics['rmse_pct']:.1f}%  MAE: {metrics['mae_pct']:.1f}%")
    return model, scaler, metrics


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_te, y_te, scaler, crop_name, use_tf=True):
    if use_tf and TF_AVAILABLE:
        y_pred_scaled = model.predict(X_te).flatten()
    else:
        X_flat = X_te.reshape(X_te.shape[0], -1)
        y_pred_scaled = model.predict(X_flat)

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mean_price = y_true.mean()

    metrics = {
        "crop":      crop_name,
        "rmse":      round(float(rmse), 2),
        "mae":       round(float(mae), 2),
        "rmse_pct":  round(float(rmse / mean_price * 100), 2),
        "mae_pct":   round(float(mae  / mean_price * 100), 2),
        "mean_price": round(float(mean_price), 2),
    }

    # Save metrics
    mpath = MODEL_DIR / f"metrics_lstm_{crop_name.lower()}.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_next_price(crop_name, n_steps=3):
    """
    Loads trained model and forecasts next n_steps months.
    Returns list of predicted prices.
    """
    from src.preprocessor import load_data
    import joblib

    df = load_data()
    df = df[df["crop"] == crop_name].sort_values("date")
    prices = df["price_per_q"].values

    scaler_path = MODEL_DIR / f"scaler_lstm_{crop_name.lower()}.pkl"
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(prices.reshape(-1, 1)).flatten()

    # Use last 12 months as seed
    window = scaled[-12:].tolist()
    predictions = []

    model_h5 = MODEL_DIR / f"lstm_{crop_name.lower()}.keras"
    model_pkl = MODEL_DIR / f"lstm_{crop_name.lower()}.pkl"

    use_tf = TF_AVAILABLE and model_h5.exists()

    if use_tf:
        model = load_model(str(model_h5))

    for _ in range(n_steps):
        x = np.array(window[-12:]).reshape(1, 12, 1)
        if use_tf:
            pred_scaled = model.predict(x, verbose=0).flatten()[0]
        else:
            model = joblib.load(model_pkl)
            pred_scaled = model.predict(x.reshape(1, -1))[0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(round(float(pred_price), 0))
        window.append(pred_scaled)

    return predictions


# ── Train all crops ───────────────────────────────────────────────────────────
def train_all():
    from src.preprocessor import prepare_lstm_data
    results = {}
    for crop in ["Groundnut", "Tomato", "Cotton"]:
        model, scaler, metrics = train(crop)
        results[crop] = metrics
    print("\n✅ All LSTM models trained")
    print(f"{'Crop':12} | {'RMSE%':6} | {'MAE%':6}")
    print("-" * 30)
    for c, m in results.items():
        print(f"{c:12} | {m['rmse_pct']:6.1f} | {m['mae_pct']:6.1f}")
    return results


if __name__ == "__main__":
    train_all()
