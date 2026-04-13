"""
ui/app.py
Complete Gradio web interface for the AI Crop Planning System.
Run: python ui/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path

from src.risk_engine     import compute_all_risks, risk_breakdown_text
from src.ranking_engine  import rank_crops, recommendation_text, COSTS
from src.visualizer      import (plot_price_history, plot_volatility_heatmap,
                                  plot_risk_breakdown, plot_profit_comparison,
                                  plot_forecast, plot_ranking_summary)

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

CROPS    = ["Groundnut", "Tomato", "Cotton"]
SEASONS  = ["Kharif (Jun-Oct)", "Rabi (Nov-Mar)", "Zaid (Apr-May)"]
MONTHS   = {"Kharif (Jun-Oct)": 8, "Rabi (Nov-Mar)": 1, "Zaid (Apr-May)": 4}
DISTRICTS= ["Anantapur", "Kurnool", "Kadapa", "Prakasam"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def try_lstm_predict(crop, n=3):
    """Try LSTM model, fall back to heuristic if model not trained."""
    try:
        from src.lstm_model import predict_next_price
        return predict_next_price(crop, n_steps=n)
    except Exception:
        # Heuristic fallback: last price ± seasonal noise
        df = pd.read_csv(DATA_DIR / "crop_data.csv")
        last = df[df["crop"] == crop]["price_per_q"].iloc[-1]
        return [round(last * np.random.uniform(0.92, 1.12)) for _ in range(n)]


def try_regression_predict(crop, rain, area, price, month):
    """Try regression model, fall back to historical mean."""
    try:
        from src.regression_model import predict_yield
        return predict_yield(crop, rain, area, price, month=month)
    except Exception:
        df = pd.read_csv(DATA_DIR / "crop_data.csv")
        return float(df[df["crop"] == crop]["yield_q_ha"].mean())


# ── Main Analysis Function ────────────────────────────────────────────────────
def run_analysis(district, season, rainfall, selected_crops,
                 custom_gn_price, custom_tom_price, custom_cot_price):
    if not selected_crops:
        return (None,) * 8 + ("⚠ Please select at least one crop.", "")

    month = MONTHS.get(season, 8)
    custom_prices = {
        "Groundnut": custom_gn_price,
        "Tomato":    custom_tom_price,
        "Cotton":    custom_cot_price,
    }

    # ── Step 1: Price Forecast ─────────────────────────────────────────────
    forecasts = {}
    for crop in selected_crops:
        preds = try_lstm_predict(crop, n=3)
        # Blend LSTM forecast with user's custom price if provided
        lstm_avg = np.mean(preds)
        user_p   = custom_prices.get(crop, 0)
        if user_p > 0:
            price = round(0.5 * lstm_avg + 0.5 * user_p, 0)
        else:
            price = round(lstm_avg, 0)
        forecasts[crop] = {"preds": preds, "price": price}

    # ── Step 2: Yield Estimation ───────────────────────────────────────────
    yields = {}
    area_defaults = {"Groundnut": 95000, "Tomato": 8000, "Cotton": 22000}
    for crop in selected_crops:
        y = try_regression_predict(
            crop, rainfall, area_defaults[crop],
            forecasts[crop]["price"], month
        )
        yields[crop] = y

    # ── Step 3: Risk Scoring ───────────────────────────────────────────────
    all_risks = compute_all_risks()

    # ── Step 4: Build crop_data for ranking ───────────────────────────────
    crop_data = []
    for crop in selected_crops:
        crop_data.append({
            "crop":             crop,
            "forecasted_price": forecasts[crop]["price"],
            "estimated_yield":  yields[crop],
            "risk_score":       all_risks[crop]["risk_score"],
        })

    # ── Step 5: Rank ───────────────────────────────────────────────────────
    ranked = rank_crops(crop_data)

    # ── Step 6: Risk info for all selected crops ───────────────────────────
    sel_risks = {c: all_risks[c] for c in selected_crops}

    # ── Figures ────────────────────────────────────────────────────────────
    top_crop = ranked[0]["crop"]
    forecast_prices = forecasts[top_crop]["preds"]

    fig_price    = plot_price_history(top_crop)
    fig_forecast = plot_forecast(top_crop, forecast_prices)
    fig_heatmap  = plot_volatility_heatmap()
    fig_risk     = plot_risk_breakdown(sel_risks)
    fig_profit   = plot_profit_comparison(ranked)
    fig_rank     = plot_ranking_summary(ranked)

    # ── Text outputs ───────────────────────────────────────────────────────
    rec_text = recommendation_text(ranked)
    risk_texts = []
    for crop in selected_crops:
        risk_texts.append(f"── {crop} ──")
        risk_texts.append(risk_breakdown_text(all_risks[crop]))
        risk_texts.append("")
    risk_text = "\n".join(risk_texts)

    # ── Summary table ──────────────────────────────────────────────────────
    table_rows = []
    for r in ranked:
        table_rows.append([
            r["rank_label"], r["crop"],
            f"₹{r['price_per_q']:,.0f}",
            f"{r['yield_q_ha']} Q/Ha",
            f"₹{r['net_profit']:,.0f}",
            f"{r['risk_score']} ({all_risks[r['crop']]['risk_label']})",
            f"{r['final_score']}"
        ])
    df_out = pd.DataFrame(table_rows,
        columns=["Rank","Crop","Price/Q","Yield","Net Profit","Risk","Final Score"])

    return (fig_price, fig_forecast, fig_heatmap,
            fig_risk, fig_profit, fig_rank,
            rec_text, risk_text, df_out)


# ── Gradio Interface ──────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="AI Crop Planning System",
        theme=gr.themes.Base(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css="""
        .header-box {
            background: linear-gradient(135deg, #0A1628 0%, #0D6B72 100%);
            color: white; padding: 24px 32px; border-radius: 10px;
            margin-bottom: 16px;
        }
        .header-box h1 { color: #3DCCC7; font-size: 1.8em; margin:0 0 6px 0; }
        .header-box p  { color: #cce8e8; margin: 0; font-size: 0.95em; }
        .badge {
            display:inline-block; background:#0D6B72; color:white;
            padding:3px 10px; border-radius:12px; font-size:0.82em;
            margin-right:8px; margin-top:6px;
        }
        """
    ) as app:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
          <h1>🌾 AI-Driven Seasonal Crop Planning & Income Risk Forecasting</h1>
          <p>Anantapur District, Andhra Pradesh &nbsp;|&nbsp;
             LSTM Price Forecasting &nbsp;|&nbsp;
             Regression Yield Estimation &nbsp;|&nbsp;
             Risk Scoring &nbsp;|&nbsp;
             Profit-Optimized Ranking</p>
          <span class="badge">Chukkaluri Devadi Reddy</span>
          <span class="badge">Dept. AI & ML</span>
        </div>
        """)

        # ── Input Panel ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Input Parameters")
                district = gr.Dropdown(DISTRICTS, value="Anantapur",
                                       label="District")
                season   = gr.Dropdown(SEASONS, value="Kharif (Jun-Oct)",
                                       label="Season")
                rainfall = gr.Slider(10, 300, value=95, step=5,
                                     label="Expected Rainfall (mm/month)")
                crops_cb = gr.CheckboxGroup(CROPS, value=CROPS,
                                            label="Select Crops to Compare")

                with gr.Accordion("📌 Custom Price Override (optional)", open=False):
                    gr.Markdown("*Leave 0 to use LSTM forecast only*")
                    gn_price  = gr.Number(value=0, label="Groundnut Price (₹/Q)")
                    tom_price = gr.Number(value=0, label="Tomato Price (₹/Q)")
                    cot_price = gr.Number(value=0, label="Cotton Price (₹/Q)")

                run_btn = gr.Button("🚀 Run Analysis", variant="primary",
                                    size="lg")

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():

            with gr.Tab("📊 Recommendation"):
                rec_text  = gr.Textbox(label="Recommendation", lines=14,
                                       show_copy_button=True)
                table_out = gr.Dataframe(label="Crop Comparison Table",
                                         wrap=True)

            with gr.Tab("📈 Price Analysis"):
                fig_price    = gr.Plot(label="Price History & Trend")
                fig_forecast = gr.Plot(label="Price Forecast (Next 3 Months)")

            with gr.Tab("🌡️ Volatility"):
                fig_heatmap = gr.Plot(label="Monthly Price Volatility Heatmap")

            with gr.Tab("⚠️ Risk Breakdown"):
                fig_risk  = gr.Plot(label="Risk Factor Breakdown")
                risk_text = gr.Textbox(label="Risk Explanation (XAI)",
                                       lines=18, show_copy_button=True)

            with gr.Tab("💰 Profit & Ranking"):
                fig_profit = gr.Plot(label="Profit vs Risk Score")
                fig_rank   = gr.Plot(label="Final Crop Ranking")

        # ── Wire up button ─────────────────────────────────────────────────
        run_btn.click(
            fn=run_analysis,
            inputs=[district, season, rainfall, crops_cb,
                    gn_price, tom_price, cot_price],
            outputs=[fig_price, fig_forecast, fig_heatmap,
                     fig_risk, fig_profit, fig_rank,
                     rec_text, risk_text, table_out]
        )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:16px; color:#888; font-size:0.85em; margin-top:24px;">
          AI-Driven Seasonal Crop Planning System &nbsp;|&nbsp;
          Dept. of AI & ML &nbsp;|&nbsp; Third Review 2026 &nbsp;|&nbsp;
          Team: Seema Minds &nbsp;|&nbsp; MSAI24R003
        </div>
        """)

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860,
               share=True, show_error=True)
