"""
src/visualizer.py
All chart generation functions for the Gradio UI.
Returns matplotlib Figure objects.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Theme ─────────────────────────────────────────────────────────────────────
COLORS = {
    "Groundnut": "#0D6B72",
    "Tomato":    "#E07B2A",
    "Cotton":    "#0A1628",
}
RISK_COLORS = {"Low": "#1A7A4A", "Medium": "#E07B2A", "High": "#C0392B"}
BG  = "#F4F7F6"
GRID= "#D0E0E0"

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:  ax.set_title(title,  fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)


# ── 1. Price History + Moving Average ────────────────────────────────────────
def plot_price_history(crop):
    df = pd.read_csv(DATA_DIR / "crop_data.csv", parse_dates=["date"])
    df = df[df["crop"] == crop].sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=BG)
    color = COLORS[crop]

    ax.fill_between(df["date"], df["price_per_q"],
                    alpha=0.18, color=color)
    ax.plot(df["date"], df["price_per_q"],
            color=color, linewidth=1.6, label="Monthly Price", alpha=0.9)
    ma6 = df["price_per_q"].rolling(6).mean()
    ax.plot(df["date"], ma6,
            color="#F0A500", linewidth=2.2, linestyle="--", label="6-month MA")

    _style(ax, f"{crop} — Price History & Trend (₹/Quintal)",
           "Date", "Price (₹/Q)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    fig.tight_layout()
    return fig


# ── 2. Volatility Heatmap ─────────────────────────────────────────────────────
def plot_volatility_heatmap():
    df = pd.read_csv(DATA_DIR / "crop_data.csv", parse_dates=["date"])
    pivot = df.pivot_table(index="month", columns="crop",
                           values="price_per_q", aggfunc="std")
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    im = ax.imshow(pivot.values.T, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(pivot.columns, fontsize=10, fontweight="bold")
    ax.set_title("Price Volatility Heatmap (Monthly Std Dev)", fontsize=13,
                 fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="Std Dev (₹/Q)")
    fig.tight_layout()
    return fig


# ── 3. Risk Factor Breakdown Bar ─────────────────────────────────────────────
def plot_risk_breakdown(risk_results):
    crops  = list(risk_results.keys())
    pv     = [risk_results[c]["pv_norm"]  for c in crops]
    wd     = [risk_results[c]["wd_norm"]  for c in crops]
    yv     = [risk_results[c]["yv_norm"]  for c in crops]
    total  = [risk_results[c]["risk_score"] for c in crops]

    x = np.arange(len(crops))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.bar(x - w, pv, w, label="Price Volatility (W=0.40)", color="#0D6B72", alpha=0.88)
    ax.bar(x,     wd, w, label="Weather Deviation (W=0.30)", color="#0E9AA7", alpha=0.88)
    ax.bar(x + w, yv, w, label="Yield Variance   (W=0.30)", color="#F0A500", alpha=0.88)
    # Total line
    ax.plot(x, total, "D--", color="#C0392B", linewidth=2,
            markersize=7, label="Total Risk Score")
    ax.set_xticks(x)
    ax.set_xticklabels(crops, fontsize=11, fontweight="bold")
    _style(ax, "Risk Factor Breakdown by Crop (0–100)", "", "Score (0–100)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    for i, t in enumerate(total):
        ax.annotate(f"{t:.0f}", xy=(i, t + 2), ha="center",
                    fontsize=11, fontweight="bold", color="#C0392B")
    fig.tight_layout()
    return fig


# ── 4. Profit Comparison Bar ─────────────────────────────────────────────────
def plot_profit_comparison(ranked):
    crops   = [r["crop"] for r in ranked]
    profits = [r["net_profit"] for r in ranked]
    colors  = [COLORS[c] for c in crops]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)

    # Profit bars
    bars = axes[0].bar(crops, profits, color=colors, width=0.5, alpha=0.9)
    _style(axes[0], "Net Profit (₹/Ha)", "", "₹/Ha")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    for bar, val in zip(bars, profits):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 500,
                     f"₹{val:,.0f}", ha="center", fontsize=10, fontweight="bold")

    # Final Score bars
    scores = [r["final_score"] for r in ranked]
    risk_s = [r["risk_score"]  for r in ranked]
    x = np.arange(len(crops))
    w = 0.35
    axes[1].bar(x - w/2, [r["profit_score"] for r in ranked], w,
                label="Profit Score", color="#0D6B72", alpha=0.88)
    axes[1].bar(x + w/2, risk_s, w,
                label="Risk Score", color="#C0392B", alpha=0.88)
    axes[1].plot(x, scores, "D--", color="#F0A500", linewidth=2,
                 markersize=7, label="Final Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crops, fontsize=10)
    _style(axes[1], "Profit vs Risk Score", "", "Score (0–100)")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 115)
    for i, s in enumerate(scores):
        axes[1].annotate(f"{s:.1f}", xy=(i, s + 2), ha="center",
                         fontsize=11, fontweight="bold", color="#F0A500")

    fig.tight_layout()
    return fig


# ── 5. Forecast Line Plot ─────────────────────────────────────────────────────
def plot_forecast(crop, forecast_prices):
    df = pd.read_csv(DATA_DIR / "crop_data.csv", parse_dates=["date"])
    df = df[df["crop"] == crop].sort_values("date").tail(18)

    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=BG)
    color = COLORS[crop]

    # Historical
    ax.plot(df["date"], df["price_per_q"],
            color=color, linewidth=1.8, label="Historical Price", marker="o",
            markersize=3)

    # Forecast
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=len(forecast_prices), freq="MS"
    )
    ax.plot(future_dates, forecast_prices,
            color="#F0A500", linewidth=2.5, linestyle="--",
            label="Forecast", marker="D", markersize=6)
    ax.axvline(last_date, color="#888888", linewidth=1, linestyle=":")
    ax.fill_between(future_dates,
                    [p * 0.92 for p in forecast_prices],
                    [p * 1.08 for p in forecast_prices],
                    alpha=0.15, color="#F0A500", label="±8% Confidence Band")

    _style(ax, f"{crop} — Price Forecast (₹/Quintal)", "Date", "Price (₹/Q)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    fig.tight_layout()
    return fig


# ── 6. Ranking Summary ───────────────────────────────────────────────────────
def plot_ranking_summary(ranked):
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
    crops  = [r["crop"]        for r in ranked]
    scores = [r["final_score"] for r in ranked]
    colors = ["#0D6B72", "#0E9AA7", "#334E5E"]

    bars = ax.barh(crops[::-1], scores[::-1], color=colors[::-1],
                   height=0.5, alpha=0.92)
    ax.set_xlim(0, max(scores) * 1.25)
    _style(ax, "Crop Ranking — Final Score (Higher = Better)", "Final Score", "")

    medals = ["🥇", "🥈", "🥉"]
    for i, (bar, sc, crop) in enumerate(zip(bars[::-1], scores, crops)):
        ax.text(sc + 0.5, bar.get_y() + bar.get_height() / 2,
                f"  {medals[i]}  {sc:.1f}", va="center",
                fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig
