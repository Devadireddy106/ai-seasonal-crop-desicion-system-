"""
generate_data.py
Generates realistic synthetic Mandi price, yield, rainfall and cost data
for Anantapur District, AP — 3 crops: Groundnut, Tomato, Cotton (2019-2024)
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).parent

# ── Date range ────────────────────────────────────────────────────────────────
months = pd.date_range("2019-01", "2024-12", freq="MS")

# ── Price patterns (₹/Quintal) ────────────────────────────────────────────────
def price_series(base, seasonal_amp, trend, noise_std, dates):
    t = np.arange(len(dates))
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12 + np.random.uniform(0, 2))
    trend_line = trend * t
    noise = np.random.normal(0, noise_std, len(t))
    # COVID dip 2020-21
    covid = np.where((dates.year == 2020) | (dates.year == 2021),
                     -base * 0.12, 0)
    prices = base + seasonal + trend_line + noise + covid
    return np.clip(prices, base * 0.5, base * 2.5).round(0)

groundnut_price = price_series(5500, 800, 25,  350, months)
tomato_price    = price_series(2800, 900, 15,  500, months)
cotton_price    = price_series(6200, 600, 30,  280, months)

# ── Rainfall (mm/month) ───────────────────────────────────────────────────────
def rainfall_series(dates):
    t = np.arange(len(dates))
    base = 40 + 80 * np.sin(2 * np.pi * t / 12 - 1.2)  # peaks July-Sept
    noise = np.random.normal(0, 15, len(t))
    return np.clip(base + noise, 0, 250).round(1)

rainfall = rainfall_series(months)

# ── Yield (Quintals/Hectare) ──────────────────────────────────────────────────
def yield_series(base, rain, area_ha, dates):
    rain_effect = 0.015 * rain
    trend = 0.008 * np.arange(len(dates))
    noise = np.random.normal(0, base * 0.06, len(dates))
    y = base + rain_effect + trend + noise
    return np.clip(y, base * 0.5, base * 1.8).round(2)

# Sown area (Ha) per season — Kharif dominant for Anantapur
area_gn  = np.random.uniform(85000, 110000, len(months)).round(0)
area_tom = np.random.uniform(5000,  12000,  len(months)).round(0)
area_cot = np.random.uniform(15000, 30000,  len(months)).round(0)

yield_gn  = yield_series(12.5, rainfall, area_gn,  months)
yield_tom = yield_series(28.0, rainfall, area_tom, months)
yield_cot = yield_series(8.2,  rainfall, area_cot, months)

# ── Cost data (₹/Ha) ─────────────────────────────────────────────────────────
cost_gn  = np.random.normal(28000, 2000, len(months)).round(0)
cost_tom = np.random.normal(35000, 3000, len(months)).round(0)
cost_cot = np.random.normal(28000, 2500, len(months)).round(0)

# ── Build DataFrames ──────────────────────────────────────────────────────────
def make_df(crop, price, yield_val, area, cost, rain):
    return pd.DataFrame({
        "date":       months,
        "year":       months.year,
        "month":      months.month,
        "crop":       crop,
        "price_per_q": price,
        "yield_q_ha": yield_val,
        "area_ha":    area,
        "rainfall_mm": rain,
        "cost_per_ha": cost,
        "gross_income": (yield_val * price).round(0),
        "net_profit":  (yield_val * price - cost).round(0),
    })

df_gn  = make_df("Groundnut", groundnut_price, yield_gn,  area_gn,  cost_gn,  rainfall)
df_tom = make_df("Tomato",    tomato_price,    yield_tom, area_tom, cost_tom, rainfall)
df_cot = make_df("Cotton",    cotton_price,    yield_cot, area_cot, cost_cot, rainfall)

df = pd.concat([df_gn, df_tom, df_cot], ignore_index=True)
df.to_csv(OUT / "crop_data.csv", index=False)

# Also save per-crop files
df_gn.to_csv(OUT / "groundnut.csv",  index=False)
df_tom.to_csv(OUT / "tomato.csv",    index=False)
df_cot.to_csv(OUT / "cotton.csv",    index=False)

print(f"✅ Data generated: {len(df)} rows  →  crop_data.csv")
print(df.groupby("crop")[["price_per_q","yield_q_ha","net_profit"]].mean().round(1))
