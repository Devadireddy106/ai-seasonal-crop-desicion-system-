"""
src/preprocessor.py
Handles all data loading, cleaning, feature engineering and
train/test splitting for LSTM and regression models.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib, os

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CROPS = ["Groundnut", "Tomato", "Cotton"]
SEQ_LEN = 12  # LSTM look-back window (months)


# ── Load & Basic Clean ────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_DIR / "crop_data.csv", parse_dates=["date"])
    df = df.sort_values(["crop", "date"]).reset_index(drop=True)
    # Fill any missing values
    df = df.ffill().bfill()
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    for crop in CROPS:
        mask = df["crop"] == crop
        p = df.loc[mask, "price_per_q"]
        df.loc[mask, "price_lag1"]  = p.shift(1)
        df.loc[mask, "price_lag3"]  = p.shift(3)
        df.loc[mask, "price_lag6"]  = p.shift(6)
        df.loc[mask, "price_ma3"]   = p.rolling(3).mean()
        df.loc[mask, "price_ma6"]   = p.rolling(6).mean()
        df.loc[mask, "price_std3"]  = p.rolling(3).std()
        df.loc[mask, "price_std6"]  = p.rolling(6).std()
        df.loc[mask, "price_change"] = p.pct_change()
    # Seasonal encoding
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df = df.dropna().reset_index(drop=True)
    return df


# ── LSTM Data Preparation ─────────────────────────────────────────────────────
def prepare_lstm_data(crop_name, test_size=0.2):
    """
    Returns:
        X_train, y_train, X_test, y_test  (numpy arrays)
        scaler  (fitted MinMaxScaler for inverse transform)
        test_dates
    """
    df = load_data()
    df = df[df["crop"] == crop_name].sort_values("date").reset_index(drop=True)

    prices = df["price_per_q"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / f"scaler_lstm_{crop_name.lower()}.pkl")

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = df["date"].iloc[SEQ_LEN + split:].values

    return X_train, y_train, X_test, y_test, scaler, test_dates


# ── Regression Data Preparation ───────────────────────────────────────────────
def prepare_regression_data(crop_name, test_size=0.25):
    """
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data()
    df = engineer_features(df)
    df = df[df["crop"] == crop_name].reset_index(drop=True)

    features = ["rainfall_mm", "area_ha", "price_per_q",
                "price_ma3", "price_ma6", "sin_month", "cos_month"]
    target = "yield_q_ha"

    X = df[features].values
    y = df[target].values

    # Scale features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, MODEL_DIR / f"scaler_reg_{crop_name.lower()}.pkl")

    split = int(len(X) * (1 - test_size))
    return (X_scaled[:split], X_scaled[split:],
            y[:split], y[split:], features)


# ── Volatility Calculation ────────────────────────────────────────────────────
def compute_volatility(crop_name, window=6):
    df = load_data()
    df = df[df["crop"] == crop_name].sort_values("date")
    prices = df["price_per_q"]
    volatility = prices.rolling(window).std().iloc[-1]
    price_mean = prices.mean()
    return float(volatility / price_mean)   # coefficient of variation


if __name__ == "__main__":
    df = load_data()
    df_fe = engineer_features(df)
    print("✅ Preprocessing OK")
    print(f"   Rows after feature engineering: {len(df_fe)}")
    for c in CROPS:
        v = compute_volatility(c)
        print(f"   {c:12s} volatility (CV): {v:.3f}")
