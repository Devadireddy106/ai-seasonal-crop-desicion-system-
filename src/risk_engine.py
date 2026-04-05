"""
src/risk_engine.py
Computes weighted risk score (0-100) per crop.
Risk = W1×Volatility + W2×WeatherDeviation + W3×YieldVariance
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Weights (empirically tuned on Anantapur APMC data) ───────────────────────
W1 = 0.40   # Price Volatility   — dominates (Mandi price drives income most)
W2 = 0.30   # Weather Deviation  — moderate impact
W3 = 0.30   # Yield Variance     — moderate impact

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LOW    = 40
RISK_MEDIUM = 70


def _load(crop):
    df = pd.read_csv(DATA_DIR / "crop_data.csv", parse_dates=["date"])
    return df[df["crop"] == crop].sort_values("date").reset_index(drop=True)


# ── Component calculations ────────────────────────────────────────────────────
def price_volatility(crop, window=6):
    """Coefficient of variation of price over rolling window."""
    df = _load(crop)
    prices = df["price_per_q"]
    cv = prices.rolling(window).std() / prices.rolling(window).mean()
    return float(cv.dropna().mean())


def weather_deviation(crop, window=6):
    """Normalised std of rainfall vs long-term mean."""
    df = _load(crop)
    rain = df["rainfall_mm"]
    long_mean = rain.mean()
    dev = abs(rain - long_mean) / (long_mean + 1e-6)
    return float(dev.mean())


def yield_variance(crop, window=6):
    """Coefficient of variation of yield."""
    df = _load(crop)
    y = df["yield_q_ha"]
    cv = y.rolling(window).std() / y.rolling(window).mean()
    return float(cv.dropna().mean())


def _norm(val, lo, hi):
    """Normalise a raw value to 0–100."""
    return float(np.clip((val - lo) / (hi - lo + 1e-9) * 100, 0, 100))


# ── Main risk score ───────────────────────────────────────────────────────────
def compute_risk(crop):
    pv = price_volatility(crop)
    wd = weather_deviation(crop)
    yv = yield_variance(crop)

    # Raw components → normalised (0-1 mapped to 0-100)
    pv_norm = _norm(pv, 0.0, 0.6)
    wd_norm = _norm(wd, 0.0, 0.5)
    yv_norm = _norm(yv, 0.0, 0.4)

    risk_score = W1 * pv_norm + W2 * wd_norm + W3 * yv_norm

    label = ("Low"    if risk_score < RISK_LOW
             else "Medium" if risk_score < RISK_MEDIUM
             else "High")

    return {
        "crop":           crop,
        "risk_score":     round(risk_score, 1),
        "risk_label":     label,
        "price_vol_raw":  round(pv, 4),
        "weather_dev_raw":round(wd, 4),
        "yield_var_raw":  round(yv, 4),
        "pv_norm":        round(pv_norm, 1),
        "wd_norm":        round(wd_norm, 1),
        "yv_norm":        round(yv_norm, 1),
        "weights":        {"W1": W1, "W2": W2, "W3": W3},
    }


def compute_all_risks():
    results = {}
    for crop in ["Groundnut", "Tomato", "Cotton"]:
        results[crop] = compute_risk(crop)
    return results


# ── Risk breakdown for XAI display ───────────────────────────────────────────
def risk_breakdown_text(risk_dict):
    r = risk_dict
    lines = [
        f"Risk Score: {r['risk_score']:.1f}/100  [{r['risk_label']} Risk]",
        f"",
        f"  Price Volatility  (W=0.40): {r['pv_norm']:.1f}  (raw CV={r['price_vol_raw']:.3f})",
        f"  Weather Deviation (W=0.30): {r['wd_norm']:.1f}  (raw={r['weather_dev_raw']:.3f})",
        f"  Yield Variance    (W=0.30): {r['yv_norm']:.1f}  (raw CV={r['yield_var_raw']:.3f})",
        f"",
        f"Formula: Risk = 0.40×{r['pv_norm']:.1f} + 0.30×{r['wd_norm']:.1f} + 0.30×{r['yv_norm']:.1f} = {r['risk_score']:.1f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 55)
    print("  RISK SCORING ENGINE")
    print("=" * 55)
    all_risks = compute_all_risks()
    for crop, r in all_risks.items():
        print(f"\n{crop}")
        print(risk_breakdown_text(r))
