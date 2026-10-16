"""
app.py
------
Main Streamlit application for the Explainable RAG Agentic AI
Clinical Decision Support System.

Run with:
    streamlit run app.py

Tabs:
    1. Prediction
    2. Why This Prediction? (SHAP)
    3. Clinical Evidence
    4. Evidence-Grounded Explanation
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Decision Support — AI Prototype",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Main background */
    .stApp {
        background: #0f172a;
        color: #e2e8f0;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Positive prediction */
    .pred-positive {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #fecaca;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Negative prediction */
    .pred-negative {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #bbf7d0;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #94a3b8;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Evidence card */
    .evidence-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #1c1917;
        border: 1px solid #78350f;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #fde68a;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }

    /* Feature table */
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .feature-table th {
        background: #334155;
        color: #94a3b8;
        padding: 0.5rem 0.75rem;
        text-align: left;
    }
    .feature-table td {
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid #1e293b;
        color: #cbd5e1;
    }
    .feature-table tr:hover td { background: #1e293b; }

    /* Step indicator */
    .step-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        text-align: center;
        line-height: 22px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    /* Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Info box */
    .info-box {
        background: #172554;
        border: 1px solid #1d4ed8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #bfdbfe;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _is_positive_prediction(prediction: str, disease: str) -> bool:
    positive = {"diabetes": "Diabetic", "anemia": "Anemia"}
    return prediction == positive.get(disease, "")


def _shap_bar_chart(contributions: list[dict], disease: str) -> plt.Figure:
    """Horizontal bar chart of SHAP values (red = positive, blue = negative)."""
    top = contributions[:8]
    features  = [c["feature"] for c in top]
    values    = [c["shap_value"] for c in top]
    colors    = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.55)))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    ax.axvline(0, color="#64748b", linewidth=1, linestyle="--")

    ax.set_xlabel("SHAP Value", color="#94a3b8", fontsize=9)
    ax.set_title(f"Local Feature Contributions — {disease.capitalize()}", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#334155")

    # Value labels
    for bar, val in zip(bars, values[::-1]):
        xpos = bar.get_width() + (0.003 if val >= 0 else -0.003)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                color="#e2e8f0", fontsize=8)

    plt.tight_layout()
    return fig


def _global_shap_chart(importance: list[dict], disease: str) -> plt.Figure:
    """Bar chart of global mean |SHAP| values."""
    features = [d["feature"] for d in importance]
    values   = [d["mean_abs_shap"] for d in importance]

    fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.55)))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    cmap = plt.cm.Blues
    norm_vals = np.array(values)
    norm_vals = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min() + 1e-9)
    colors_g = [cmap(0.4 + 0.5 * v) for v in norm_vals[::-1]]

    ax.barh(features[::-1], values[::-1], color=colors_g, edgecolor="none", height=0.6)
    ax.set_xlabel("Mean |SHAP Value|", color="#94a3b8", fontsize=9)
    ax.set_title(f"Global Feature Importance — {disease.capitalize()}", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#334155")

    plt.tight_layout()
    return fig


def _render_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Academic Prototype — Educational and Decision-Support Use Only</strong><br>
        This prediction is generated by a machine learning model and should not be treated as a
        confirmed medical diagnosis or a substitute for professional medical evaluation.
        All clinical decisions must be made by qualified medical professionals with complete
        patient history and appropriate diagnostic testing.
        The clinical context displayed is from development sample documents and does not represent
        official WHO or other medical guidelines.
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_global_shap_cached(disease: str) -> list:
    """Compute and cache global SHAP importance per disease (persists across reruns)."""
    from explainability.shap_explainer import get_global_shap
    return get_global_shap(disease)


# ─────────────────────────────────────────────
# Sidebar — Disease selection & patient input
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🏥</span>
        <h2 style="color:#e2e8f0; margin:0.3rem 0 0 0; font-size:1.1rem; font-weight:700;">
            Clinical Decision Support
        </h2>
        <p style="color:#64748b; font-size:0.75rem; margin:0.2rem 0 0 0;">
            Explainable AI · RAG · SHAP
        </p>
    </div>
    <hr style="border-color:#334155; margin:1rem 0;">
    """, unsafe_allow_html=True)

    disease = st.selectbox(
        "🔬 Select Disease",
        options=["diabetes", "anemia"],
        format_func=lambda x: x.capitalize(),
        help="Choose the disease model to run.",
    )

    st.markdown("<p style='color:#94a3b8; font-size:0.8rem; margin-top:1rem;'><b>Patient Input</b></p>", unsafe_allow_html=True)

    patient_data = {}

    if disease == "diabetes":
        patient_data["Age"]    = st.slider("Age (years)", 18, 90, 45)
        patient_data["Gender"] = st.selectbox("Gender", ["Female", "Male"])
        patient_data["BMI"]    = st.number_input("BMI (kg/m²)", 10.0, 60.0, 27.5, step=0.1, format="%.1f")
        patient_data["BloodPressure"]            = st.number_input("Blood Pressure (Diastolic mm Hg)", 40, 140, 72, step=1)
        patient_data["Insulin"]                  = st.number_input("Insulin (mu U/ml)", 0.0, 600.0, 80.0, step=1.0)
        patient_data["Glucose"]                  = st.number_input("Glucose (mg/dL)", 50, 300, 120, step=1)
        patient_data["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.35, step=0.01, format="%.3f")

    else:  # anemia
        gender_label           = st.selectbox("Gender", ["Female", "Male"])
        patient_data["Gender"] = 1 if gender_label == "Male" else 0
        patient_data["Hemoglobin"] = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 13.5, step=0.1, format="%.1f")
        patient_data["MCH"]        = st.number_input("MCH (pg)", 10.0, 45.0, 28.0, step=0.1, format="%.1f")
        patient_data["MCHC"]       = st.number_input("MCHC (g/dL)", 15.0, 45.0, 33.0, step=0.1, format="%.1f")
        patient_data["MCV"]        = st.number_input("MCV (fL)", 40.0, 130.0, 85.0, step=0.1, format="%.1f")

    st.markdown("<hr style='border-color:#334155; margin:1rem 0;'>", unsafe_allow_html=True)

    include_llm = st.toggle("🤖 Enable LLM Explanation", value=True,
                            help="Disable if no API key is configured.")

    analyze_btn = st.button("🔍 Analyze Patient", use_container_width=True)

    st.markdown("""
    <hr style="border-color:#334155; margin:1rem 0;">
    <p style="color:#475569; font-size:0.7rem; text-align:center;">
        Academic Prototype<br>
        Explainable RAG · SHAP · LangChain
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="color:#e2e8f0; font-size:1.6rem; font-weight:700; margin-bottom:0.25rem;">
    🏥 Clinical Decision Support System
</h1>
<p style="color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;">
    Explainable Retrieval-Augmented Agentic AI Framework for Trustworthy Disease Prediction
</p>
""", unsafe_allow_html=True)

if not analyze_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.2rem; text-align:center;">
            <div style="font-size:2rem;">🧠</div>
            <div style="color:#e2e8f0; font-weight:600; margin:0.5rem 0 0.3rem 0;">ML Prediction</div>
            <div style="color:#64748b; font-size:0.8rem;">Trained Diabetes & Anemia models with probability scores</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.2rem; text-align:center;">
            <div style="font-size:2rem;">📊</div>
            <div style="color:#e2e8f0; font-weight:600; margin:0.5rem 0 0.3rem 0;">SHAP Explainability</div>
            <div style="color:#64748b; font-size:0.8rem;">Per-patient feature contributions from the actual model</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.2rem; text-align:center;">
            <div style="font-size:2rem;">📚</div>
            <div style="color:#e2e8f0; font-weight:600; margin:0.5rem 0 0.3rem 0;">RAG Evidence</div>
            <div style="color:#64748b; font-size:0.8rem;">ChromaDB retrieval · LLM grounded explanation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem;">
        ℹ️ Select a disease and enter patient values in the sidebar, then click <strong>Analyze Patient</strong>.
    </div>""", unsafe_allow_html=True)

    _render_disclaimer()
    st.stop()


# ─────────────────────────────────────────────
# Run pipeline (cached in session state)
# ─────────────────────────────────────────────

# Build a hashable cache key from the current inputs
_cache_key = (disease, include_llm, str(sorted(patient_data.items())))

# Only recompute when the button is clicked with new inputs
if analyze_btn or "_result" not in st.session_state or st.session_state.get("_cache_key") != _cache_key:
    with st.spinner("🔄 Running analysis pipeline..."):
        try:
            from agent.orchestrator import run_analysis
            _result = run_analysis(
                disease=disease,
                patient_data=patient_data,
                retrieval_k=4,
                skip_llm=not include_llm,
            )
            st.session_state["_result"] = _result
            st.session_state["_cache_key"] = _cache_key
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

result = st.session_state["_result"]


# ─────────────────────────────────────────────
# Check for critical failure
# ─────────────────────────────────────────────
if not result.get("success") or result["prediction"] is None:
    st.error("❌ Prediction failed. Please check that models are trained (`python train_models.py`).")
    if result.get("errors"):
        for k, v in result["errors"].items():
            st.error(f"[{k}] {v}")
    st.stop()


pred_result      = result["prediction"]
shap_result      = result.get("shap_local")
evidence         = result.get("retrieved_evidence", [])
llm_out          = result.get("llm_output")
query            = result.get("clinical_query", "")
errors           = result.get("errors", {})
ood_warnings     = result.get("warnings", [])
exec_profile     = result.get("execution_profile", {})
risk_tier        = pred_result.get("risk_tier", "Moderate Risk")

is_positive = _is_positive_prediction(pred_result["prediction"], disease)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction",
    "📊 Why This Prediction?",
    "📚 Clinical Evidence",
    "💡 Explanation",
])


# ════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Model Prediction Result</div>", unsafe_allow_html=True)

    pred_class = "pred-positive" if is_positive else "pred-negative"
    pred_icon  = "🔴" if is_positive else "🟢"
    st.markdown(
        f'<div class="{pred_class}">'
        f'{pred_icon} <strong>Disease:</strong> {disease.capitalize()}&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<strong>Prediction:</strong> {pred_result["prediction"]}&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<strong>Probability:</strong> {pred_result["probability"]*100:.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disease", disease.capitalize())
    c2.metric("Prediction", pred_result["prediction"])
    c3.metric("Probability", f"{pred_result['probability']*100:.1f}%")
    risk_color = {"Low Risk": "#4ade80", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"}.get(risk_tier, "#94a3b8")
    c4.markdown(
        f'<div data-testid="metric-container"><div style="font-size:0.85rem;color:#94a3b8;">Risk Tier</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:{risk_color};">{risk_tier}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div style="color:#475569; font-size:0.8rem; margin-top:0.75rem;">
        Model used: <strong style="color:#94a3b8;">{pred_result['model_name']}</strong>
    </div>""", unsafe_allow_html=True)

    # OOD / Input Screening Warnings
    if ood_warnings:
        for w in ood_warnings:
            severity = w.get("severity", "warning")
            icon = "⚠️" if severity == "warning" else "🔴"
            st.markdown(
                f'<div style="background:#1c1917;border:1px solid #78350f;border-left:4px solid #f59e0b;'
                f'border-radius:8px;padding:0.6rem 1rem;color:#fde68a;font-size:0.82rem;margin-top:0.4rem;">'
                f'{icon} <strong>Input Screen [{w.get("feature","")}]:</strong> {w.get("message","")}'
                f'</div>',
                unsafe_allow_html=True
            )

    # Pipeline steps status
    st.markdown("<br><div class='section-header'>Pipeline Status</div>", unsafe_allow_html=True)
    steps = [
        ("Prediction",         "prediction" not in errors),
        ("SHAP Analysis",      "shap"       not in errors and shap_result is not None),
        ("Query Builder",      "query"      not in errors),
        ("Evidence Retrieval", "retrieval"  not in errors and len(evidence) > 0),
        ("LLM Explanation",    "llm"        not in errors and (llm_out is not None) and bool(llm_out.get("explanation"))),
    ]
    cols = st.columns(len(steps))
    for col, (step_name, ok) in zip(cols, steps):
        icon  = "OK" if ok else ("SKIP" if not include_llm and step_name == "LLM Explanation" else "FAIL")
        icon_sym = {"OK": "✅", "SKIP": "⏭️", "FAIL": "❌"}[icon]
        color = "#4ade80" if ok else ("#f59e0b" if icon == "SKIP" else "#ef4444")
        col.markdown(
            f'<div style="background:#1e293b; border:1px solid #334155; border-radius:8px; '
            f'padding:0.6rem; text-align:center;">'
            f'<div style="font-size:1.2rem;">{icon_sym}</div>'
            f'<div style="color:{color}; font-size:0.75rem; font-weight:600; margin-top:0.2rem;">{step_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Execution Latency Profile
    if exec_profile and exec_profile.get("total_ms", 0) > 0:
        st.markdown("<br><div class='section-header'>Execution Latency Profile</div>", unsafe_allow_html=True)
        lp1, lp2, lp3, lp4, lp5 = st.columns(5)
        lp1.metric("Total",     f"{exec_profile.get('total_ms', 0):.0f} ms")
        lp2.metric("Inference", f"{exec_profile.get('prediction_ms', 0):.0f} ms")
        lp3.metric("SHAP",      f"{exec_profile.get('shap_ms', 0):.0f} ms")
        lp4.metric("RAG",       f"{exec_profile.get('retrieval_ms', 0):.0f} ms")
        lp5.metric("LLM",       f"{exec_profile.get('llm_ms', 0):.0f} ms")

    _render_disclaimer()


# ════════════════════════════════════════════
# TAB 2 — SHAP Explanation
# ════════════════════════════════════════════
with tab2:
    if "shap" in errors:
        st.error(f"SHAP analysis failed: {errors['shap']}")
    elif shap_result is None:
        st.warning("SHAP results not available.")
    else:
        contributions = shap_result.get("feature_contributions", [])
        base_val      = shap_result.get("base_value", 0)

        st.markdown("<div class='section-header'>Local Feature Contributions (This Patient)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>How to read:</strong> Each bar shows how much a feature pushed the model's output
            toward (<span style="color:#f87171;">↑ red</span>) or away from
            (<span style="color:#60a5fa;">↓ blue</span>) the positive class.
            These are model contributions — not medical causes.
        </div>""", unsafe_allow_html=True)

        # Bar chart
        fig = _shap_bar_chart(contributions, disease)
        st.pyplot(fig)
        plt.close(fig)

        # Feature table
        st.markdown("<br><div class='section-header'>Feature Contribution Table</div>", unsafe_allow_html=True)
        table_data = []
        for c in contributions:
            bar_pct = int(min(abs(c["shap_value"]) / (abs(contributions[0]["shap_value"]) + 1e-9) * 100, 100))
            bar_str = "█" * (bar_pct // 10) + "░" * (10 - bar_pct // 10)
            table_data.append({
                "Rank"       : c["rank"],
                "Feature"    : c["feature"],
                "SHAP Value" : f"{c['shap_value']:+.4f}",
                "Direction"  : c["direction"],
                "Contribution": bar_str,
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Global SHAP
        st.markdown("<br><div class='section-header'>Global Feature Importance (Training Sample)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Global importance shows which features are <em>generally</em> important to the model
            across many patients. This is different from the local contributions above,
            which are specific to this patient.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Computing global SHAP..."):
            try:
                global_imp = _get_global_shap_cached(disease)
                fig2 = _global_shap_chart(global_imp, disease)
                st.pyplot(fig2)
                plt.close(fig2)

                # Global table
                df_global = pd.DataFrame([
                    {"Rank": g["rank"], "Feature": g["feature"], "Mean |SHAP|": f"{g['mean_abs_shap']:.4f}"}
                    for g in global_imp
                ])
                st.dataframe(df_global, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Global SHAP failed: {e}")


# ════════════════════════════════════════════
# TAB 3 — Clinical Evidence
# ════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Retrieved Clinical Evidence</div>", unsafe_allow_html=True)

    if query:
        st.markdown(f"""
        <div class="info-box">
            🔍 <strong>Retrieval query (generated from SHAP features):</strong><br>
            <em style="color:#e2e8f0;">{query}</em>
        </div>""", unsafe_allow_html=True)

    if "retrieval" in errors:
        st.error(f"Evidence retrieval failed: {errors['retrieval']}")
        st.markdown("""
        <div class="info-box">
            Make sure ChromaDB is populated: <code>python rag/ingest.py</code>
        </div>""", unsafe_allow_html=True)
    elif not evidence:
        st.warning("No evidence retrieved. Check that ChromaDB is populated.")
    else:
        st.markdown(f"<p style='color:#64748b; font-size:0.85rem;'>Retrieved {len(evidence)} document chunk(s) from the <strong>{disease}</strong> knowledge base.</p>", unsafe_allow_html=True)

        for i, ev in enumerate(evidence, start=1):
            sim_score = ev.get("similarity_score", 0.0)
            rel_passed = ev.get("is_relevance_passed", True)
            sim_pct = f"{sim_score * 100:.1f}%"
            badge_color = "#4ade80" if rel_passed else "#f59e0b"
            badge_label = "Relevant" if rel_passed else "Low Sim"
            st.markdown(f"""
            <div class="evidence-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <div style="color:#60a5fa; font-weight:600;">
                        Evidence {i} &nbsp;|&nbsp; {ev['source_name']}
                        &nbsp;<span style="color:#475569; font-size:0.78rem;">[{ev['source_type'].upper()}]</span>
                    </div>
                    <div>
                        <span style="background:#1e3a5f;color:#93c5fd;border-radius:4px;padding:2px 8px;font-size:0.75rem;">
                            Similarity: {sim_pct}
                        </span>
                        &nbsp;
                        <span style="color:{badge_color};font-size:0.75rem;font-weight:600;">
                            {badge_label}
                        </span>
                    </div>
                </div>
                <div style="color:#94a3b8; font-size:0.78rem; margin-bottom:0.5rem;">
                    Document: {ev['document_name']} &middot; Chunk #{ev['chunk_id']}
                </div>
                <div style="color:#cbd5e1;">
                    {ev['content']}
                </div>
            </div>""", unsafe_allow_html=True)


    # Note about sample docs
    st.markdown("""
    <div style="color:#475569; font-size:0.75rem; margin-top:1rem; padding:0.75rem;
                background:#0f172a; border-radius:8px; border:1px solid #1e293b;">
        ℹ️ <strong>Note:</strong> The documents above are development sample documents
        created for academic prototype testing. They do not represent official WHO or
        other medical guidelines. Real clinical documents can be added later by replacing
        the files in <code>data/sample_documents/</code> and re-running <code>python rag/ingest.py --reset</code>.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 4 — LLM Explanation
# ════════════════════════════════════════════
with tab4:
    if not include_llm:
        st.info("🤖 LLM explanation is disabled. Enable the toggle in the sidebar to activate it.")
    elif "llm" in errors:
        st.error(f"LLM explanation failed: {errors['llm']}")
        st.markdown("""
        <div class="info-box">
            Check that your API key is set in <code>.env</code> file.<br>
            Required: <code>OPENAI_API_KEY</code> (or configure a different provider in <code>llm/generator.py</code>).
        </div>""", unsafe_allow_html=True)
    elif llm_out is None or not llm_out.get("explanation"):
        st.warning("LLM explanation not available.")
    else:
        explanation_text = llm_out["explanation"]

        # Labels: clearly distinguish ML vs RAG vs LLM
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown("<div class='section-header'>Evidence-Grounded Clinical Explanation</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                The explanation below is generated by the LLM using <strong>only</strong>
                the retrieved evidence passages and SHAP analysis. It is not a diagnosis.
            </div>""", unsafe_allow_html=True)

            # Parse and display sections
            sections = explanation_text.split("## ")
            for section in sections:
                if section.strip():
                    st.markdown("## " + section)

        with col_r:
            st.markdown("<div class='section-header'>Context Provided to LLM</div>", unsafe_allow_html=True)

            with st.expander("📊 SHAP Summary (sent to LLM)", expanded=False):
                st.code(llm_out.get("shap_summary", "N/A"))

            with st.expander("📚 Evidence Summary (sent to LLM)", expanded=False):
                st.text(llm_out.get("evidence_summary", "N/A"))

    _render_disclaimer()
