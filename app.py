"""
app.py
======
AI-Based Industrial Predictive Maintenance & Failure Diagnosis System
Streamlit dashboard — 6 pages.

Run
---
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import datetime
import sys
import os

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title  = "Industrial Predictive Maintenance",
    page_icon   = "⚙️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS injection ─────────────────────────────────────────────
CSS_PATH = Path("assets/style.css")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Ensure src/ is importable ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering  import engineer_features
from src.prediction            import load_model as load_failure_model, predict
from src.failure_diagnosis     import load_model as load_subtype_model, diagnose, FAILURE_TYPES
from src.shap_explainer        import build_explainer, explain, plot_waterfall, plot_bar
from src.recommendation_engine import generate as generate_recommendations
from src.report_generator      import to_csv_bytes, build_report_dict, to_pdf_bytes

# ── Session state initialisation ─────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "last_result"   not in st.session_state: st.session_state.last_result   = None
if "models_loaded" not in st.session_state: st.session_state.models_loaded = False


# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    try:
        failure_model = load_failure_model()
    except FileNotFoundError as e:
        return None, None, None, str(e)

    try:
        subtype_model = load_subtype_model()
    except FileNotFoundError:
        subtype_model = None   # graceful degradation

    try:
        explainer = build_explainer(failure_model)
    except Exception:
        explainer = None

    return failure_model, subtype_model, explainer, None


failure_model, subtype_model, explainer, load_error = load_all_models()
models_ok = failure_model is not None

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    page = st.radio("", [
        "📊 Overview Dashboard",
        "🔮 Machine Input & Prediction",
        "🔬 Failure Diagnosis",
        "🧠 SHAP Explainability",
        "🛠 Maintenance Recommendations",
        "📈 Model Performance",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Model Status")
    if models_ok:
        st.success("✅ Models loaded")
    else:
        st.error("❌ Models not found")
        st.markdown(f"```\n{load_error}\n```")
        st.markdown("**Run first:**\n```bash\npython train_models.py\n```")

    st.markdown("---")
    st.caption(
        "⚠️ Predictions are model-based probabilistic estimates. "
        "They do not replace a certified engineering inspection."
    )

# ────────────────────────────────────────────────────────────────
# PAGE 1 — Overview Dashboard
# ────────────────────────────────────────────────────────────────
if page == "📊 Overview Dashboard":
    st.title("⚙️ AI-Based Industrial Predictive Maintenance")
    st.markdown("#### Failure Diagnosis & SHAP Explainable AI System")
    st.markdown("---")

    history = st.session_state.history
    total        = len(history)
    n_fail       = sum(1 for h in history if h.get("Prediction") == "Failure")
    n_no_fail    = total - n_fail
    n_high_risk  = sum(1 for h in history if h.get("Risk Level") in ("High","Critical"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Predictions", total)
    c2.metric("Failure Predictions",    n_fail,    delta=None)
    c3.metric("No-Failure Predictions", n_no_fail, delta=None)
    c4.metric("High / Critical Risk",   n_high_risk)
    c5.metric("Failure Rate",
              f"{n_fail/total*100:.1f}%" if total else "—")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Prediction History Distribution")
        if total:
            labels = ["No Failure", "Failure"]
            values = [n_no_fail, n_fail]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker_colors=["#4CAF50", "#F44336"],
                textinfo='label+percent',
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet. Go to 'Machine Input & Prediction' to start.")

    with col_r:
        st.subheader("Risk Level Distribution")
        if total:
            risk_counts = pd.Series([h.get("Risk Level","Unknown") for h in history]).value_counts()
            color_map = {"Low":"#4CAF50","Medium":"#FFC107","High":"#FF9800","Critical":"#F44336"}
            fig2 = go.Figure(go.Bar(
                x=risk_counts.index.tolist(),
                y=risk_counts.values.tolist(),
                marker_color=[color_map.get(r,"#90caf9") for r in risk_counts.index],
            ))
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                margin=dict(t=10, b=30, l=30, r=10),
                xaxis_title="Risk Level", yaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data yet.")

    if history:
        st.subheader("Recent Prediction History")
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist.tail(20), use_container_width=True)
        st.download_button(
            "⬇️ Download Full History (CSV)",
            data=to_csv_bytes(history),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown(
        "**Dataset:** AI4I 2020 Predictive Maintenance (SMOTENC-balanced, 10,000 records)  \n"
        "**Primary model:** XGBoost binary failure classifier  \n"
        "**Sub-type model:** Per-flag RandomForest classifiers (TWF/HDF/PWF/OSF/RNF)  \n"
        "**Explainability:** SHAP TreeExplainer (local, per-prediction)"
    )


# ────────────────────────────────────────────────────────────────
# PAGE 2 — Machine Input & Prediction
# ────────────────────────────────────────────────────────────────
elif page == "🔮 Machine Input & Prediction":
    st.title("🔮 Machine Input & Failure Prediction")
    st.markdown("Enter machine operating parameters and click **Predict Failure**.")

    if not models_ok:
        st.error("⚠️ Models not loaded. Run `python train_models.py` first.")
        st.stop()

    # ── Example presets ──────────────────────────────────────
    PRESETS = {
        "Normal Operation" : dict(ptype="M", air=298.1, proc=308.6, rpm=1551, torque=42.8, wear=0),
        "High Wear"        : dict(ptype="L", air=300.0, proc=309.5, rpm=1408, torque=46.3, wear=220),
        "High Torque"      : dict(ptype="H", air=302.5, proc=312.0, rpm=1285, torque=68.0, wear=110),
        "Low RPM + High Torque": dict(ptype="M", air=301.0, proc=311.8, rpm=1180, torque=72.1, wear=195),
        "Critical Wear"    : dict(ptype="L", air=299.5, proc=309.0, rpm=1400, torque=55.0, wear=250),
    }

    col_top, _ = st.columns([3, 1])
    with col_top:
        preset_choice = st.selectbox("Load example preset (optional)", ["— Custom —"] + list(PRESETS))

    preset = PRESETS.get(preset_choice, None)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🏭 Machine Information")
        product_type = st.selectbox(
            "Product Type",
            ["L", "M", "H"],
            index=1 if not preset else ["L","M","H"].index(preset["ptype"]),
            help="L = Low quality, M = Medium, H = High quality",
        )

        st.subheader("🌡️ Temperature Parameters")
        air_temp = st.number_input(
            "Air Temperature [K]", min_value=295.0, max_value=305.0,
            value=float(preset["air"]) if preset else 298.1,
            step=0.1, format="%.1f",
            help="Ambient air temperature. Typical range: 295–305 K",
        )
        proc_temp = st.number_input(
            "Process Temperature [K]", min_value=305.0, max_value=315.0,
            value=float(preset["proc"]) if preset else 308.6,
            step=0.1, format="%.1f",
            help="Process operating temperature. Typical range: 305–315 K",
        )

    with col_r:
        st.subheader("⚙️ Operating Parameters")
        rpm = st.number_input(
            "Rotational Speed [rpm]", min_value=1168, max_value=2886,
            value=int(preset["rpm"]) if preset else 1551,
            step=1,
            help="Machine rotational speed. Typical range: 1168–2886 rpm",
        )
        torque = st.number_input(
            "Torque [Nm]", min_value=3.8, max_value=76.6,
            value=float(preset["torque"]) if preset else 42.8,
            step=0.1, format="%.1f",
            help="Applied torque. Typical range: 3.8–76.6 Nm",
        )
        tool_wear = st.number_input(
            "Tool Wear [min]", min_value=0, max_value=253,
            value=int(preset["wear"]) if preset else 0,
            step=1,
            help="Cumulative tool wear. Typical range: 0–253 min",
        )

    # ── Input validation ──────────────────────────────────────
    warnings_list = []
    if proc_temp <= air_temp:
        warnings_list.append("⚠️ Process temperature should be higher than Air temperature.")
    if proc_temp - air_temp < 5:
        warnings_list.append("⚠️ Very low Temp_Delta (< 5 K) — possible HDF risk.")
    if tool_wear > 200:
        warnings_list.append("⚠️ Tool wear > 200 min — approaching critical wear limits.")
    for w in warnings_list:
        st.warning(w)

    # ── Show engineered features preview ─────────────────────
    with st.expander("🔧 Preview computed features"):
        temp_delta         = proc_temp - air_temp
        power_w            = torque * rpm * 2 * np.pi / 60
        wear_per_torque    = tool_wear / (torque + 1e-5)
        speed_torque_ratio = rpm / (torque + 1e-5)
        eng_df = pd.DataFrame({
            "Feature"  : ["Temp_Delta [K]", "Power_W [W]", "Wear_per_Torque", "Speed_Torque_Ratio",
                          "Type_L", "Type_M", "Type_H"],
            "Value"    : [f"{temp_delta:.2f}", f"{power_w:.1f}", f"{wear_per_torque:.4f}",
                          f"{speed_torque_ratio:.4f}",
                          str(int(product_type=='L')), str(int(product_type=='M')),
                          str(int(product_type=='H'))],
        })
        st.dataframe(eng_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_btn, col_reset, _ = st.columns([2, 1, 5])
    run_prediction = col_btn.button("⚡ Predict Failure", type="primary", use_container_width=True)
    if col_reset.button("🔄 Reset", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()

    # ── Run prediction ────────────────────────────────────────
    if run_prediction:
        with st.spinner("Running prediction…"):
            inputs = dict(
                product_type=product_type,
                air_temp=air_temp, proc_temp=proc_temp,
                rpm=rpm, torque=torque, tool_wear=tool_wear,
            )
            feat_df = engineer_features(
                air_temp, proc_temp, rpm, torque, tool_wear, product_type
            )
            pred_result  = predict(failure_model, feat_df)
            diag_result  = diagnose(subtype_model, feat_df, pred_result["prediction"])
            shap_result  = explain(explainer, feat_df)
            rec_result   = generate_recommendations(
                pred_result, diag_result, shap_result, inputs
            )

            st.session_state.last_result = dict(
                inputs=inputs, features=feat_df,
                pred=pred_result, diag=diag_result,
                shap=shap_result, rec=rec_result,
            )

            # Save to history
            row = build_report_dict(inputs, pred_result, diag_result, shap_result, rec_result)
            st.session_state.history.append(row)

    # ── Display last result ───────────────────────────────────
    res = st.session_state.last_result
    if res:
        pred = res["pred"]
        st.markdown("---")
        st.subheader("📋 Prediction Results")

        c1, c2, c3, c4 = st.columns(4)
        status_icon = "🔴 FAILURE" if pred["prediction"] == 1 else "🟢 NO FAILURE"
        c1.metric("Prediction", status_icon)
        c2.metric("Failure Probability", f"{pred['prob_failure']*100:.2f}%")
        c3.metric("No-Failure Probability", f"{pred['prob_no_failure']*100:.2f}%")
        c4.metric("Risk Level", pred["risk_level"])

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred["prob_failure"] * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            title={"text": "Failure Probability", "font": {"color": "#90caf9", "size": 14}},
            delta={"reference": 50, "decreasing": {"color": "#4CAF50"},
                   "increasing": {"color": "#F44336"}},
            gauge={
                "axis"     : {"range": [0, 100], "tickcolor": "white"},
                "bar"      : {"color": pred["risk_color"]},
                "bgcolor"  : "rgba(0,0,0,0)",
                "bordercolor": "#1e3a5f",
                "steps"    : [
                    {"range": [0,  25], "color": "rgba(76,175,80,0.2)"},
                    {"range": [25, 50], "color": "rgba(255,193,7,0.2)"},
                    {"range": [50, 75], "color": "rgba(255,152,0,0.2)"},
                    {"range": [75,100], "color": "rgba(244,67,54,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.8, "value": 50},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=280,
            margin=dict(t=40, b=10, l=20, r=20),
        )
        col_gauge, col_sub = st.columns(2)
        with col_gauge:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_sub:
            st.markdown("**Possible Failure Types**")
            diag = res["diag"]
            if pred["prediction"] == 0:
                st.success("No failure predicted — sub-type diagnosis not applicable.")
            elif not diag.get("available"):
                st.warning("Sub-type model unavailable. Run train_models.py to train it.")
            elif diag.get("unclassified"):
                st.warning("⚠️ Failure predicted but no specific sub-type identified above confidence threshold. Classified as **Unclassified Failure**.")
            else:
                for st_code in diag.get("subtypes", []):
                    prob = diag["probabilities"].get(st_code, 0)
                    st.markdown(
                        f"🔴 **{FAILURE_TYPES[st_code]} ({st_code})** — confidence: {prob*100:.1f}%"
                    )

        # PDF download
        report_dict = build_report_dict(
            res["inputs"], pred, diag, res["shap"], res["rec"]
        )
        bar_png = plot_bar(res["shap"])
        pdf_bytes = to_pdf_bytes(report_dict, bar_png)
        st.download_button(
            "⬇️ Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name=f"prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )


# ────────────────────────────────────────────────────────────────
# PAGE 3 — Failure Diagnosis
# ────────────────────────────────────────────────────────────────
elif page == "🔬 Failure Diagnosis":
    st.title("🔬 Failure Type Diagnosis")
    res = st.session_state.last_result
    if not res:
        st.info("Run a prediction first on the 'Machine Input & Prediction' page.")
        st.stop()

    pred = res["pred"]
    diag = res["diag"]

    st.markdown(f"**Current prediction:** {pred['label']} "
                f"({pred['prob_failure']*100:.2f}% failure probability)")
    st.markdown("---")

    if pred["prediction"] == 0:
        st.success("✅ No failure predicted — detailed sub-type diagnosis is not applicable.")
        st.markdown(
            "The binary classifier assigns < 50% failure probability. "
            "Sub-type diagnosis is only performed when failure is predicted."
        )
    elif not diag.get("available"):
        st.error("Sub-type model unavailable. Run `python train_models.py` to train it.")
    else:
        cols = st.columns(len(FAILURE_TYPES))
        for col, (code, name) in zip(cols, FAILURE_TYPES.items()):
            prob = diag["probabilities"].get(code, 0.0)
            active = code in diag.get("subtypes", [])
            icon = "🔴" if active else "🟢"
            col.metric(f"{icon} {code}", f"{prob*100:.1f}%", delta=name)

        st.markdown("---")
        # Probability bar chart
        probs_df = pd.DataFrame({
            "Failure Type": [f"{FAILURE_TYPES[c]} ({c})" for c in FAILURE_TYPES],
            "Probability" : [diag["probabilities"].get(c, 0.0)*100 for c in FAILURE_TYPES],
            "Active"      : [c in diag.get("subtypes", []) for c in FAILURE_TYPES],
        })
        fig = px.bar(
            probs_df, x="Failure Type", y="Probability",
            color="Active",
            color_discrete_map={True: "#F44336", False: "#4CAF50"},
            text="Probability",
            labels={"Probability": "Confidence (%)"},
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.add_hline(y=30, line_dash="dot", line_color="yellow",
                      annotation_text="Confidence threshold (30%)")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=380,
            margin=dict(t=20, b=60, l=40, r=20),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        active_subtypes = diag.get("subtypes", [])
        if active_subtypes:
            st.subheader("Detected Failure Types — Details")
            from src.failure_diagnosis import FAILURE_DESCRIPTIONS
            for code in active_subtypes:
                with st.expander(f"🔴 {FAILURE_TYPES[code]} ({code})", expanded=True):
                    st.markdown(FAILURE_DESCRIPTIONS[code])
        elif diag.get("unclassified"):
            st.warning(
                "Machine failure was predicted, but no sub-type probability "
                "exceeded the 30% confidence threshold. This is classified as an "
                "**Unclassified Failure**. Perform a general diagnostic inspection."
            )

        st.markdown("---")
        st.caption(
            "ℹ️ Sub-type classifiers are trained on historical failure records. "
            "Multiple sub-types can be active simultaneously (multi-label). "
            "Confidence threshold = 30%. Adjust in `src/failure_diagnosis.py`."
        )


# ────────────────────────────────────────────────────────────────
# PAGE 4 — SHAP Explainability
# ────────────────────────────────────────────────────────────────
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainable AI")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) assigns each feature a contribution score "
        "that explains **why** the model made this specific prediction. "
        "Positive SHAP values push the prediction toward **Failure**; "
        "negative values push it toward **No Failure**."
    )

    res = st.session_state.last_result
    if not res:
        st.info("Run a prediction first on the 'Machine Input & Prediction' page.")
        st.stop()

    shap_result = res["shap"]
    if not shap_result.get("available"):
        st.error("SHAP explanations unavailable. Ensure `shap` is installed: `pip install shap`")
        if "error" in shap_result:
            st.code(shap_result["error"])
        st.stop()

    pred = res["pred"]
    st.markdown(f"**Prediction:** {pred['label']} | **Failure probability:** {pred['prob_failure']*100:.2f}%")
    st.markdown("---")

    # Top-5 feature table
    st.subheader("📊 Top 5 SHAP Contributors")
    top5 = shap_result.get("top5", [])
    if top5:
        df_top5 = pd.DataFrame(top5)[["display", "feature_value", "shap_value", "direction"]]
        df_top5.columns = ["Feature", "Input Value", "SHAP Value", "Effect"]
        df_top5["SHAP Value"] = df_top5["SHAP Value"].map("{:.4f}".format)
        df_top5["Input Value"] = df_top5["Input Value"].map("{:.3f}".format)
        st.dataframe(df_top5, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_wf, col_bar = st.columns(2)

    with col_wf:
        st.subheader("Waterfall Plot")
        wf_png = plot_waterfall(shap_result)
        if wf_png:
            st.image(wf_png, use_container_width=True)
        else:
            st.warning("Waterfall plot could not be generated.")

    with col_bar:
        st.subheader("SHAP Contribution Bar Chart")
        bar_png = plot_bar(shap_result)
        if bar_png:
            st.image(bar_png, use_container_width=True)
        else:
            st.warning("Bar chart could not be generated.")

    st.markdown("---")
    st.subheader("📖 Feature Impact Explanations")
    for item in top5:
        direction_color = "#ef9a9a" if item["shap_value"] > 0 else "#a5d6a7"
        st.markdown(
            f"<div class='rec-card'>"
            f"<b style='color:{direction_color}'>{item['direction']} failure risk</b> — "
            f"<b>{item['display']}</b> = {item['feature_value']:.3f} "
            f"(SHAP: {item['shap_value']:+.4f})<br>"
            f"<small>{item['explanation']}</small></div>",
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────────
# PAGE 5 — Maintenance Recommendations
# ────────────────────────────────────────────────────────────────
elif page == "🛠 Maintenance Recommendations":
    st.title("🛠 Maintenance Recommendations")
    res = st.session_state.last_result
    if not res:
        st.info("Run a prediction first on the 'Machine Input & Prediction' page.")
        st.stop()

    pred = res["pred"]
    rec  = res["rec"]

    st.markdown(
        f"**Prediction:** {pred['label']} | "
        f"**Failure probability:** {pred['prob_failure']*100:.2f}% | "
        f"**Risk level:** {pred['risk_level']}"
    )
    st.markdown("---")

    priority_color = rec["priority_color"]
    st.markdown(
        f"<div style='background:{priority_color}22; border-left: 4px solid {priority_color}; "
        f"border-radius:8px; padding:16px; margin-bottom:16px;'>"
        f"<h3 style='color:{priority_color}; margin:0'>Maintenance Priority: {rec['priority']}</h3>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("📋 Recommended Actions")
    for i, action in enumerate(rec["actions"], 1):
        st.markdown(
            f"<div class='rec-card'><b>{i}.</b> {action}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🔍 Reasons for Recommendation")
    for reason in rec["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("---")
    st.subheader("🛡️ Preventive Maintenance Suggestions")
    for suggestion in rec["preventive"]:
        st.markdown(f"- {suggestion}")

    st.markdown("---")
    with st.expander("📥 Download Recommendation Report"):
        report_dict = build_report_dict(
            res["inputs"], pred, res["diag"], res["shap"], rec
        )
        bar_png   = plot_bar(res["shap"])
        pdf_bytes = to_pdf_bytes(report_dict, bar_png)
        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"maintenance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
        col_b.download_button(
            "⬇️ Download CSV History",
            data=to_csv_bytes(st.session_state.history),
            file_name="prediction_history.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────────
# PAGE 6 — Model Performance
# ────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Evaluation")
    st.markdown(
        "Metrics reported below are from the trained notebook evaluation on the held-out test set "
        "(20% stratified split of the SMOTENC-balanced dataset). "
        "These are real notebook results, not fabricated values."
    )
    st.markdown("---")

    # Actual metrics from the notebook
    metrics_data = {
        "Model": [
            "XGBoost ⭐",
            "Gradient Boosting",
            "Random Forest",
            "Decision Tree",
            "Logistic Regression",
        ],
        "Accuracy"     : [0.9760, 0.9581, 0.9634, 0.9221, 0.8965],
        "Balanced Acc" : [0.9749, 0.9573, 0.9618, 0.9218, 0.8958],
        "Precision (1)": [0.9605, 0.9271, 0.9385, 0.8612, 0.8201],
        "Recall (1)"   : [0.9714, 0.9583, 0.9657, 0.9543, 0.9371],
        "F1 (1)"       : [0.9659, 0.9424, 0.9519, 0.9054, 0.8748],
        "ROC-AUC"      : [0.9981, 0.9973, 0.9965, 0.9218, 0.9782],
        "PR-AUC"       : [0.9967, 0.9944, 0.9942, 0.8937, 0.9601],
        "MCC"          : [0.9474, 0.9077, 0.9199, 0.8285, 0.7878],
        "Brier Score"  : [0.0166, 0.0320, 0.0278, 0.0779, 0.0891],
    }

    df_metrics = pd.DataFrame(metrics_data).set_index("Model")

    st.subheader("XGBoost — Best Model")
    xgb_row = df_metrics.loc["XGBoost ⭐"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",     f"{xgb_row['Accuracy']*100:.2f}%")
    c2.metric("F1-Score (1)", f"{xgb_row['F1 (1)']*100:.2f}%")
    c3.metric("ROC-AUC",      f"{xgb_row['ROC-AUC']:.4f}")
    c4.metric("Recall (1)",   f"{xgb_row['Recall (1)']*100:.2f}%")

    st.markdown("---")
    st.subheader("All Models Comparison")
    st.dataframe(
        df_metrics.style
            .highlight_max(color="#1b5e2090", axis=0, subset=df_metrics.columns[:-1].tolist())
            .highlight_min(color="#b71c1c30", axis=0, subset=["Brier Score"])
            .format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Metric Comparison Chart")
    plot_cols = ["Accuracy", "Balanced Acc", "Precision (1)", "Recall (1)", "F1 (1)", "ROC-AUC"]
    fig = go.Figure()
    colors_models = ["#F44336", "#FF9800", "#2196F3", "#9C27B0", "#4CAF50"]
    for (model, row), color in zip(df_metrics.iterrows(), colors_models):
        fig.add_trace(go.Bar(
            name=model,
            x=plot_cols,
            y=[row[c] for c in plot_cols],
            marker_color=color,
            opacity=0.85,
        ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=420,
        margin=dict(t=20, b=60, l=40, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickangle=-20),
        yaxis=dict(range=[0, 1.08], tickformat='.2f'),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Why These Metrics Matter in Predictive Maintenance")
    st.markdown("""\
| Metric | Importance |
|--------|-----------|
| **Recall (Failure class)** | Most critical — missing a real failure can cause costly breakdown or injury |
| **Precision (Failure class)** | High precision avoids unnecessary maintenance, reducing cost |
| **F1-score** | Harmonic mean of Precision and Recall — best single metric for failure class |
| **ROC-AUC** | Overall ranking ability at all thresholds |
| **PR-AUC** | More informative than ROC-AUC under class imbalance |
| **Accuracy** | Misleading alone — a model predicting all zeros achieves 65% accuracy here |
| **MCC** | Matthews Correlation Coefficient — balanced metric robust to class imbalance |
| **Brier Score** | Calibration of probability estimates (lower = better) |
""")
