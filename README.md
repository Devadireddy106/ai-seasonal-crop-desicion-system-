# 🌾 AI-Driven Seasonal Crop Planning & Income Risk Forecasting System

**Anantapur District, Andhra Pradesh


---

## Project Overview

An end-to-end AI/ML platform that helps farmers and agri-businesses make
data-driven seasonal crop decisions by combining:

| Module | Technique | Status |
|--------|-----------|--------|
| Price Forecasting | LSTM (TensorFlow/Keras) | ✅ Complete |
| Yield Estimation | Random Forest Regression | ✅ Complete |
| Risk Scoring | Weighted Formula (W1+W2+W3) | ✅ Complete |
| Profit Calculation | Economic Model | ✅ Complete |
| Crop Ranking | Profit-Risk Optimization | ✅ Complete |
| Web UI | Gradio Dashboard | ✅ Complete |

---

## Project Structure

```
crop_project/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   ├── crop_data.csv        # Generated after running
│   ├── groundnut.csv
│   ├── tomato.csv
│   └── cotton.csv
├── src/
│   ├── preprocessor.py      # Data loading + feature engineering
│   ├── lstm_model.py        # LSTM price forecasting
│   ├── regression_model.py  # Yield estimation (Random Forest)
│   ├── risk_engine.py       # Risk scoring engine
│   ├── ranking_engine.py    # Profit calc + crop ranking
│   └── visualizer.py        # All chart generation
├── models/                  # Saved model files (after training)
│   ├── lstm_groundnut.h5
│   ├── regression_groundnut.pkl
│   └── ...
├── ui/
│   └── app.py               # Gradio web interface
├── outputs/                 # Training summaries, results
├── train_all.py             # ← RUN THIS FIRST
├── requirements.txt
└── README.md
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train all models
```bash
cd crop_project
python train_all.py
```
This will:
- Generate the dataset (2019–2024, 3 crops, Anantapur)
- Train LSTM price forecasting models
- Train Random Forest yield regression models
- Compute risk scores
- Run full pipeline test

### Step 3 — Launch the web UI
```bash
python ui/app.py
```
Open browser → **http://localhost:7860**

---

## Mathematical Models

### LSTM (Price Forecasting)
```
fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf)   Forget Gate
iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi)   Input Gate
Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ    Cell Update
hₜ = oₜ ⊙ tanh(Cₜ)         Hidden State
```

### Risk Score Formula
```
Risk = (W1 × Volatility) + (W2 × WeatherDeviation) + (W3 × YieldVariance)
W1=0.40, W2=0.30, W3=0.30
RiskScore = Normalize(Risk, 0–100)
```

### Profit Calculation
```
GrossIncome = Yield (Q/Ha) × ForecastedPrice (₹/Q)
NetProfit   = GrossIncome − Cost (₹/Ha)
```

### Crop Ranking
```
FinalScore = α × ProfitScore − β × RiskScore
α=0.60 (profit weight), β=0.40 (risk penalty)
Recommendation = argmax(FinalScore)
```

---

## Performance Targets

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| LSTM – Groundnut | RMSE% | ≤12% | ~11.4% ✅ |
| LSTM – Tomato | RMSE% | ≤15% | ~13.2% ✅ |
| LSTM – Cotton | RMSE% | ≤15% | ~15.8% ⟳ |
| Regression (RF) | R² | ≥0.80 | ~0.81 ✅ |
| Risk Scoring | Validation | Historical | ✅ |

---

## Technology Stack

- **Python 3.11**
- **TensorFlow/Keras** — LSTM model
- **Scikit-learn** — Random Forest, preprocessing
- **Pandas/NumPy** — data pipeline
- **Matplotlib** — visualizations
- **Gradio** — web UI
- **Joblib** — model serialization

---

## Future Scope (Phase 2+)

- Real-time Agmarknet API for live Mandi prices
- IMD weather API integration
- Multi-district expansion (all 13 AP districts)
- Mobile app (Telugu language support)
- MLflow for model versioning + retraining pipeline
- Crop insurance risk linkage

---

Dept. of AI & ML*
