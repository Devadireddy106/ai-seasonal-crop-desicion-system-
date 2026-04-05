"""
src/regression_model.py
Random Forest regression for yield estimation per crop.
Features: rainfall, area, price lags, seasonal encoding.
"""
import numpy as np
import json, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

MODEL_DIR = Path(__file__).parent.parent / "models"


# ── Train ─────────────────────────────────────────────────────────────────────
def train(crop_name):
    from src.preprocessor import prepare_regression_data
    print(f"\n{'='*50}")
    print(f"  Training Regression (Yield) for {crop_name}")
    print(f"{'='*50}")

    X_tr, X_te, y_tr, y_te, features = prepare_regression_data(crop_name)

    # Compare 3 models
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=8,
                                                   random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
    }

    best_model, best_r2, best_name = None, -999, ""
    results = {}

    for name, clf in candidates.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        cv   = cross_val_score(clf, X_tr, y_tr, cv=5, scoring="r2").mean()
        results[name] = {"r2": round(r2,3), "rmse": round(rmse,2),
                         "mae": round(mae,2), "cv_r2": round(cv,3)}
        print(f"  {name:20}: R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  CV-R²={cv:.3f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, clf, name

    print(f"\n  ✅ Best: {best_name}  (R²={best_r2:.3f})")

    # Feature importance (if RF/GB)
    importance = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        importance = dict(zip(features, [round(float(v),4) for v in imp]))
        print("  Feature importances:")
        for feat, val in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"    {feat:20}: {val:.4f}")

    # Save best model
    joblib.dump(best_model, MODEL_DIR / f"regression_{crop_name.lower()}.pkl")

    metrics = {
        "crop":       crop_name,
        "best_model": best_name,
        "r2":         round(best_r2, 3),
        "all_models": results,
        "feature_importance": importance,
        "features":   features,
    }
    with open(MODEL_DIR / f"metrics_reg_{crop_name.lower()}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return best_model, metrics


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_yield(crop_name, rainfall_mm, area_ha, price_per_q,
                  price_ma3=None, price_ma6=None, month=6):
    """
    Predict yield (Q/Ha) for a crop given input conditions.
    """
    import numpy as np

    model_path  = MODEL_DIR / f"regression_{crop_name.lower()}.pkl"
    scaler_path = MODEL_DIR / f"scaler_reg_{crop_name.lower()}.pkl"

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Use defaults if optional inputs not provided
    if price_ma3 is None:
        price_ma3 = price_per_q * 0.97
    if price_ma6 is None:
        price_ma6 = price_per_q * 0.94

    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)

    X_raw = np.array([[rainfall_mm, area_ha, price_per_q,
                        price_ma3, price_ma6, sin_m, cos_m]])
    X_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_scaled)[0]
    return round(float(y_pred), 2)


# ── Train all crops ───────────────────────────────────────────────────────────
def train_all():
    all_metrics = {}
    for crop in ["Groundnut", "Tomato", "Cotton"]:
        _, m = train(crop)
        all_metrics[crop] = m
    print("\n✅ All regression models trained")
    return all_metrics


if __name__ == "__main__":
    train_all()
