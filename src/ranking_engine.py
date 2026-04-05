"""
src/ranking_engine.py
Profit calculation + crop ranking system.
FinalScore = alpha × ProfitScore − beta × RiskScore
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Ranking weights ───────────────────────────────────────────────────────────
ALPHA = 0.60   # Profit weight
BETA  = 0.40   # Risk penalty

# ── Cost lookup (₹/Ha) — government benchmarks, Anantapur 2024 ───────────────
COSTS = {
    "Groundnut": 28000,
    "Tomato":    35000,
    "Cotton":    29000,
}


# ── Profit calculation ────────────────────────────────────────────────────────
def compute_profit(crop, forecasted_price, estimated_yield):
    """
    gross_income = yield (Q/Ha) × price (₹/Q)
    net_profit   = gross_income − cost (₹/Ha)
    """
    cost         = COSTS.get(crop, 30000)
    gross_income = round(estimated_yield * forecasted_price, 0)
    net_profit   = round(gross_income - cost, 0)
    profit_margin= round((net_profit / gross_income * 100) if gross_income > 0 else 0, 1)
    return {
        "crop":           crop,
        "yield_q_ha":     round(estimated_yield, 2),
        "price_per_q":    round(forecasted_price, 0),
        "cost_per_ha":    cost,
        "gross_income":   gross_income,
        "net_profit":     net_profit,
        "profit_margin%": profit_margin,
    }


# ── Normalise to 0–100 ────────────────────────────────────────────────────────
def _normalise(values):
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [50.0] * len(values)
    return list((arr - mn) / (mx - mn) * 100)


# ── Main ranking function ─────────────────────────────────────────────────────
def rank_crops(crop_data):
    """
    crop_data: list of dicts with keys:
        crop, forecasted_price, estimated_yield, risk_score
    Returns: sorted list of dicts (best crop first)
    """
    profits = [
        compute_profit(d["crop"], d["forecasted_price"], d["estimated_yield"])
        for d in crop_data
    ]

    net_profits   = [p["net_profit"]  for p in profits]
    risk_scores   = [d["risk_score"]  for d in crop_data]

    profit_scores  = _normalise(net_profits)
    # Risk is already 0-100; higher risk → lower score
    risk_penalties = risk_scores   # direct use

    ranked = []
    for i, p in enumerate(profits):
        final_score = ALPHA * profit_scores[i] - BETA * risk_penalties[i]
        ranked.append({
            **p,
            "risk_score":    round(risk_scores[i], 1),
            "profit_score":  round(profit_scores[i], 1),
            "final_score":   round(final_score, 2),
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    for i, r in enumerate(ranked):
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        r["rank"]       = i + 1
        r["rank_label"] = medals[i] if i < 3 else f"{i+1}th"
        r["recommend"]  = i == 0

    return ranked


# ── Recommendation summary ────────────────────────────────────────────────────
def recommendation_text(ranked):
    top = ranked[0]
    lines = [
        f"✅ RECOMMENDED CROP: {top['crop']}",
        f"",
        f"   Expected Yield    : {top['yield_q_ha']} Q/Ha",
        f"   Forecasted Price  : ₹{top['price_per_q']:,.0f}/Quintal",
        f"   Gross Income      : ₹{top['gross_income']:,.0f}/Ha",
        f"   Net Profit        : ₹{top['net_profit']:,.0f}/Ha",
        f"   Profit Margin     : {top['profit_margin%']}%",
        f"   Risk Score        : {top['risk_score']}/100",
        f"   Final Score       : {top['final_score']}",
        f"",
        f"Score = {ALPHA}×{top['profit_score']:.1f}(Profit) − {BETA}×{top['risk_score']}(Risk) = {top['final_score']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with sample data
    from src.risk_engine import compute_all_risks

    test_data = [
        {"crop": "Groundnut", "forecasted_price": 5800, "estimated_yield": 12.5, "risk_score": 0},
        {"crop": "Tomato",    "forecasted_price": 3200, "estimated_yield": 28.0, "risk_score": 0},
        {"crop": "Cotton",    "forecasted_price": 6500, "estimated_yield": 8.2,  "risk_score": 0},
    ]

    risks = compute_all_risks()
    for d in test_data:
        d["risk_score"] = risks[d["crop"]]["risk_score"]

    ranked = rank_crops(test_data)
    print("\n" + "=" * 60)
    print("  CROP RANKING RESULTS")
    print("=" * 60)
    for r in ranked:
        print(f"\n{r['rank_label']} {r['crop']}")
        print(f"  Profit: ₹{r['net_profit']:,.0f}/Ha  |  Risk: {r['risk_score']}  |  Score: {r['final_score']}")

    print("\n" + recommendation_text(ranked))
