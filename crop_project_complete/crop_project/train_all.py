"""
train_all.py
Master training script — run this ONCE to train all models.
Usage: python train_all.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

print("=" * 60)
print("  AI CROP PLANNING SYSTEM — MODEL TRAINING")
print("  Anantapur District | Dept. AI & ML | 2026")
print("=" * 60)

# ── Step 1: Generate Data ─────────────────────────────────────────────────────
print("\n📦 STEP 1: Generating synthetic dataset...")
t0 = time.time()
exec(open("data/generate_data.py").read())
print(f"   Done in {time.time()-t0:.1f}s")

# ── Step 2: Preprocess ────────────────────────────────────────────────────────
print("\n🔧 STEP 2: Running preprocessor check...")
from src.preprocessor import load_data, engineer_features, compute_volatility

df = load_data()
df_fe = engineer_features(df)
print(f"   Rows loaded: {len(df)}  |  After feature engineering: {len(df_fe)}")
for c in ["Groundnut", "Tomato", "Cotton"]:
    v = compute_volatility(c)
    print(f"   {c:12s} price volatility (CV): {v:.3f}")

# ── Step 3: Train Regression Models ──────────────────────────────────────────
print("\n🌾 STEP 3: Training yield regression models...")
t0 = time.time()
from src.regression_model import train_all as reg_train_all
reg_metrics = reg_train_all()
print(f"\n   Regression training done in {time.time()-t0:.1f}s")

# ── Step 4: Train LSTM Models ─────────────────────────────────────────────────
print("\n🧠 STEP 4: Training LSTM price forecasting models...")
t0 = time.time()
from src.lstm_model import train_all as lstm_train_all
lstm_metrics = lstm_train_all()
print(f"\n   LSTM training done in {time.time()-t0:.1f}s")

# ── Step 5: Risk Scoring ──────────────────────────────────────────────────────
print("\n⚠️  STEP 5: Computing risk scores...")
from src.risk_engine import compute_all_risks, risk_breakdown_text
risks = compute_all_risks()
for crop, r in risks.items():
    print(f"\n   {crop}")
    print(f"   Score: {r['risk_score']} [{r['risk_label']}]  "
          f"PV={r['pv_norm']:.1f}  WD={r['wd_norm']:.1f}  YV={r['yv_norm']:.1f}")

# ── Step 6: Test Full Ranking ─────────────────────────────────────────────────
print("\n🏆 STEP 6: Testing crop ranking pipeline...")
from src.ranking_engine import rank_crops, recommendation_text
from src.lstm_model import predict_next_price
from src.regression_model import predict_yield

crop_data = []
for crop in ["Groundnut", "Tomato", "Cotton"]:
    preds = predict_next_price(crop, n_steps=3)
    price = sum(preds) / len(preds)
    yld   = predict_yield(crop, 95, 90000, price)
    crop_data.append({
        "crop":             crop,
        "forecasted_price": price,
        "estimated_yield":  yld,
        "risk_score":       risks[crop]["risk_score"],
    })

ranked = rank_crops(crop_data)
print("\n" + recommendation_text(ranked))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 60)
print("\n  LSTM Performance:")
for crop, m in lstm_metrics.items():
    print(f"    {crop:12}: RMSE={m['rmse_pct']:.1f}%  MAE={m['mae_pct']:.1f}%")

print("\n  Regression Performance:")
for crop, m in reg_metrics.items():
    print(f"    {crop:12}: R²={m['r2']:.3f}  Best={m['best_model']}")

print("\n  Risk Scores:")
for crop, r in risks.items():
    print(f"    {crop:12}: {r['risk_score']:5.1f}  [{r['risk_label']:6}]")

print("\n  🚀 To launch the UI:")
print("     python ui/app.py")
print("\n  Then open: http://localhost:7860")
print("=" * 60)

# Save summary
summary = {
    "lstm":       {k: {"rmse_pct": v["rmse_pct"], "mae_pct": v["mae_pct"]}
                   for k, v in lstm_metrics.items()},
    "regression": {k: {"r2": v["r2"], "best_model": v["best_model"]}
                   for k, v in reg_metrics.items()},
    "risk":       {k: {"score": v["risk_score"], "label": v["risk_label"]}
                   for k, v in risks.items()},
}
Path("outputs/training_summary.json").write_text(
    json.dumps(summary, indent=2)
)
print("\n  📄 Summary saved → outputs/training_summary.json")
