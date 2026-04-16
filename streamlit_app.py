"""
NeuroCollab — Streamlit UI v3.0 MAX
Run: streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pipeline import (
    predict, get_model_results, get_feature_importance, get_summary,
    FEATURE_NAMES, FEATURE_CATEGORIES,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroCollab · Academic AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "NeuroCollab v3.0 MAX — Academic Link Prediction Engine"},
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root & body ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: #02040f !important;
  font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#050a1a 0%,#080d1f 100%) !important;
  border-right: 1px solid rgba(94,184,255,.12) !important;
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }

/* ── Typography ─────────────────────────────────────── */
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; }

/* ── Hide default header/footer ─────────────────────── */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Cards ──────────────────────────────────────────── */
.nc-card {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(94,184,255,.14);
  border-radius: 20px;
  padding: 28px 32px;
  margin-bottom: 20px;
  backdrop-filter: blur(12px);
  transition: border-color .3s;
}
.nc-card:hover { border-color: rgba(94,184,255,.35); }

/* ── Hero ────────────────────────────────────────────── */
.nc-hero {
  text-align: center;
  padding: 56px 0 36px;
  position: relative;
}
.nc-hero-title {
  font-size: clamp(2.4rem, 5vw, 4rem);
  font-weight: 800;
  background: linear-gradient(135deg,#5eb8ff 0%,#b794f4 50%,#2dd4bf 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -1px;
  line-height: 1.1;
  margin-bottom: 12px;
}
.nc-hero-sub {
  color: #94a3c4;
  font-size: 1.05rem;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: .04em;
}
.nc-badge {
  display: inline-block;
  background: rgba(94,184,255,.12);
  border: 1px solid rgba(94,184,255,.3);
  color: #5eb8ff;
  border-radius: 999px;
  padding: 4px 14px;
  font-size: .78rem;
  font-family: 'JetBrains Mono', monospace;
  margin: 4px 3px;
  letter-spacing: .04em;
}

/* ── Score gauge ring ────────────────────────────────── */
.nc-score-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 24px 0;
}
.nc-score-label {
  font-size: 4.5rem; font-weight: 800; line-height: 1;
  margin-top: 8px;
}
.nc-score-verdict {
  font-size: 1.15rem; font-weight: 600; margin-top: 6px;
  letter-spacing: .04em;
}
.level-high    { color: #2dd4bf; }
.level-medium  { color: #5eb8ff; }
.level-low-medium { color: #f59e0b; }
.level-low     { color: #fc6b6b; }

/* ── Insight chips ───────────────────────────────────── */
.insight-chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 16px; border-radius: 999px;
  font-size: .85rem; margin: 4px 4px 4px 0;
  font-family: 'JetBrains Mono', monospace;
}
.insight-positive { background:rgba(45,212,191,.12); border:1px solid rgba(45,212,191,.35); color:#2dd4bf; }
.insight-neutral  { background:rgba(94,184,255,.10); border:1px solid rgba(94,184,255,.28); color:#5eb8ff; }
.insight-negative { background:rgba(252,107,107,.10); border:1px solid rgba(252,107,107,.28); color:#fc6b6b; }

/* ── Stat tiles ──────────────────────────────────────── */
.stat-tile {
  background: rgba(255,255,255,.035);
  border: 1px solid rgba(94,184,255,.10);
  border-radius: 16px; padding: 20px 24px; text-align: center;
  transition: all .25s;
}
.stat-tile:hover { background: rgba(94,184,255,.07); border-color: rgba(94,184,255,.3); }
.stat-number {
  font-size: 2.1rem; font-weight: 800;
  background: linear-gradient(90deg,#5eb8ff,#b794f4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.stat-label { color: #94a3c4; font-size: .8rem; margin-top:4px; font-family:'JetBrains Mono',monospace; }

/* ── Section titles ──────────────────────────────────── */
.nc-section-title {
  font-size: 1.15rem; font-weight: 700; color: #eef2ff;
  letter-spacing: .02em; margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}
.nc-section-title::before {
  content: ''; display:block; width:4px; height:18px;
  background: linear-gradient(180deg,#5eb8ff,#b794f4);
  border-radius: 4px;
}

/* ── Metric overrides ────────────────────────────────── */
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  color: #5eb8ff !important;
}
[data-testid="stMetricLabel"] { color: #94a3c4 !important; font-size: .8rem !important; }

/* ── Slider/input overrides ──────────────────────────── */
[data-testid="stSlider"] .st-b8, [data-testid="stSlider"] .st-ae {
  background: linear-gradient(90deg,#5eb8ff,#b794f4) !important;
}
.stSlider > div > div { color: #c8d8ff !important; }

/* ── Button ──────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg,#5eb8ff 0%,#b794f4 100%) !important;
  color: #02040f !important;
  border: none !important;
  border-radius: 14px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  padding: 14px 36px !important;
  transition: all .25s !important;
  letter-spacing: .04em !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(94,184,255,.35) !important;
}

/* ── Tab styling ─────────────────────────────────────── */
[data-testid="stTab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
}

/* ── Animated gradient line ──────────────────────────── */
.nc-divider {
  height: 2px;
  background: linear-gradient(90deg, transparent, #5eb8ff, #b794f4, #2dd4bf, transparent);
  border-radius: 2px;
  margin: 32px 0;
  animation: shimmer 3s linear infinite;
  background-size: 200% auto;
}
@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}

/* ── Pulse ring on score ─────────────────────────────── */
@keyframes pulseRing {
  0%   { box-shadow: 0 0 0 0 rgba(94,184,255,.4); }
  70%  { box-shadow: 0 0 0 24px rgba(94,184,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(94,184,255,0); }
}
.nc-pulse { animation: pulseRing 2s ease-out infinite; }
</style>
""", unsafe_allow_html=True)


# ─── HELPER: PLOTLY THEME ─────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#94a3c4"),
    margin=dict(l=24, r=24, t=48, b=24),
    colorway=["#5eb8ff", "#b794f4", "#2dd4bf", "#f59e0b", "#fc6b6b", "#f472b6"],
)


def _fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT)
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NeuroCollab v3.0 MAX")
    st.markdown('<hr style="border-color:rgba(94,184,255,.2)">', unsafe_allow_html=True)
    st.markdown("**Input Parameters**")

    # Researcher A
    st.markdown("##### 👤 Researcher A")
    degree_u  = st.slider("Degree (A)",          1, 200, 20)
    clust_u   = st.slider("Clustering Coeff (A)", 0.0, 1.0, 0.25, 0.01)
    tri_u     = st.slider("Triangles (A)",        0, 100, 8)
    cs_u      = st.slider("Community Size (A)",   5, 2000, 200)
    pr_u      = st.number_input("PageRank (A)",   0.0, 0.01, 0.0003, 0.0001, format="%.5f")
    core_u    = st.slider("k-Core (A)",           1, 20, 4)

    st.markdown("##### 👤 Researcher B")
    degree_v  = st.slider("Degree (B)",           1, 200, 15)
    clust_v   = st.slider("Clustering Coeff (B)", 0.0, 1.0, 0.18, 0.01)
    tri_v     = st.slider("Triangles (B)",        0, 100, 6)
    cs_v      = st.slider("Community Size (B)",   5, 2000, 150)
    pr_v      = st.number_input("PageRank (B)",   0.0, 0.01, 0.0002, 0.0001, format="%.5f")
    core_v    = st.slider("k-Core (B)",           1, 20, 3)

    st.markdown("##### 🔗 Shared Properties")
    cn              = st.slider("Common Neighbors", 0, 50, 5)
    same_community  = st.selectbox("Same Research Community?", [0, 1], format_func=lambda x: "✅ Yes" if x else "❌ No")

    st.markdown('<div class="nc-divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Compatibility", width='stretch')


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nc-hero">
  <div class="nc-hero-title">NeuroCollab</div>
  <div class="nc-hero-sub">Academic Collaboration · Link Prediction · Graph ML</div>
  <div style="margin-top:20px;">
    <span class="nc-badge">25 Features</span>
    <span class="nc-badge">8 Models</span>
    <span class="nc-badge">~99.4% AUC</span>
    <span class="nc-badge">DBLP Graph</span>
    <span class="nc-badge">Voting Ensemble</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── LIVE STATS ───────────────────────────────────────────────────────────────
summary = get_summary()
col1, col2, col3, col4, col5 = st.columns(5)
for col, num, label in zip(
    [col1, col2, col3, col4, col5],
    [
        f"{summary.get('num_nodes', 20000):,}",
        f"{summary.get('num_edges', 15384):,}",
        f"{summary.get('samples', {}).get('total', 12000):,}",
        "25",
        "99.4%",
    ],
    ["Graph Nodes", "Graph Edges", "Training Samples", "Features", "Best AUC"],
):
    with col:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="stat-number">{num}</div>
          <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="nc-divider"></div>', unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_pred, tab_models, tab_features, tab_pipeline = st.tabs([
    "🔮 Prediction", "📊 Model Performance", "🧬 Feature Analysis", "⚙️ Pipeline"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab_pred:
    result = None
    if predict_btn or "last_result" in st.session_state:
        if predict_btn:
            with st.spinner("Running prediction pipeline…"):
                result = predict(
                    cn=cn, du=degree_u, dv=degree_v, same_community=same_community,
                    clust_u=clust_u, clust_v=clust_v, tri_u=tri_u, tri_v=tri_v,
                    cs_u=cs_u, cs_v=cs_v, pr_u=pr_u, pr_v=pr_v,
                    core_u=core_u, core_v=core_v,
                )
                st.session_state["last_result"] = result
        else:
            result = st.session_state["last_result"]

        r = result
        c_left, c_right = st.columns([1, 2])

        with c_left:
            # Score gauge
            level_col = {"high": "#2dd4bf", "medium": "#5eb8ff", "low-medium": "#f59e0b", "low": "#fc6b6b"}.get(r.level, "#5eb8ff")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r.score,
                number={"suffix": "%", "font": {"size": 48, "family": "Syne", "color": level_col}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4a5570", "tickfont": {"size": 11}},
                    "bar": {"color": level_col, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(252,107,107,.08)"},
                        {"range": [35, 55], "color": "rgba(245,158,11,.08)"},
                        {"range": [55, 78], "color": "rgba(94,184,255,.08)"},
                        {"range": [78, 100],"color": "rgba(45,212,191,.08)"},
                    ],
                    "threshold": {"line": {"color": level_col, "width": 3}, "thickness": 0.75, "value": r.score},
                },
                title={"text": f"{r.emoji}  {r.verdict}", "font": {"size": 16, "color": level_col, "family": "Syne"}},
            ))
            fig_gauge.update_layout(**_LAYOUT, height=320)
            st.plotly_chart(fig_gauge, width='stretch')

            st.markdown(f"""
            <div class="nc-card" style="text-align:center">
              <div style="font-size:.8rem;color:#94a3c4;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">
                {"✅ ML Model Active" if r.used_ml_model else "⚠️ Formula Fallback"} · {r.model_name}
              </div>
              <div style="color:#eef2ff;font-size:.9rem;line-height:1.6;">{r.advice}</div>
            </div>""", unsafe_allow_html=True)

        with c_right:
            # Insights
            st.markdown('<div class="nc-section-title">Key Insights</div>', unsafe_allow_html=True)
            chips_html = ""
            for ins in r.insights:
                chips_html += f'<span class="insight-chip insight-{ins["type"]}">{ins["text"]}</span>'
            st.markdown(chips_html, unsafe_allow_html=True)

            st.markdown("---")

            # Feature contributions bar chart
            st.markdown('<div class="nc-section-title">Feature Contributions (SHAP-like)</div>', unsafe_allow_html=True)
            top_all = sorted(r.contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            names_   = [k.replace("_", " ").title() for k, _ in top_all]
            vals_    = [v * 100 for _, v in top_all]
            colors_  = ["#2dd4bf" if v > 0 else "#fc6b6b" for v in vals_]

            fig_bar = go.Figure(go.Bar(
                x=vals_, y=names_, orientation="h",
                marker_color=colors_,
                marker_line_width=0,
                text=[f"{v:+.1f}%" for v in vals_],
                textposition="outside",
                textfont=dict(size=11, color="#94a3c4"),
            ))
            fig_bar.update_layout(**_LAYOUT, height=380,
                xaxis=dict(title="Contribution (%)", gridcolor="rgba(94,184,255,.07)", zeroline=True, zerolinecolor="rgba(94,184,255,.2)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_bar, width='stretch')

        # Feature table
        st.markdown('<div class="nc-section-title">All 25 Computed Features</div>', unsafe_allow_html=True)
        rows = []
        for cat, feats in FEATURE_CATEGORIES.items():
            for feat in feats:
                val = r.features.get(feat, 0)
                imp = get_feature_importance().get(feat, 0) * 100
                rows.append({"Category": cat, "Feature": feat.replace("_", " ").title(),
                             "Value": round(val, 5), "Importance %": round(imp, 2)})
        df_feat = pd.DataFrame(rows)
        st.dataframe(df_feat.style.background_gradient(subset=["Importance %"], cmap="Blues"),
                     width='stretch', hide_index=True)

    else:
        st.markdown("""
        <div class="nc-card" style="text-align:center;padding:60px 40px;">
          <div style="font-size:3rem;margin-bottom:16px;">🧠</div>
          <div style="font-size:1.3rem;font-weight:700;color:#eef2ff;margin-bottom:8px;">
            Configure researchers in the sidebar
          </div>
          <div style="color:#94a3c4;font-family:'JetBrains Mono',monospace;">
            Adjust sliders → click ⚡ Predict Compatibility
          </div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab_models:
    model_results = get_model_results()
    if model_results:
        df_models = pd.DataFrame(model_results).T.reset_index()
        df_models.rename(columns={"index": "Model"}, inplace=True)
        for col in ["accuracy","f1","roc_auc","mcc"]:
            if col in df_models.columns:
                df_models[col] = pd.to_numeric(df_models[col], errors="coerce")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="nc-section-title">AUC-ROC Comparison</div>', unsafe_allow_html=True)
            df_s = df_models.sort_values("roc_auc", ascending=True)
            fig_auc = go.Figure(go.Bar(
                x=df_s["roc_auc"] * 100, y=df_s["Model"], orientation="h",
                marker=dict(
                    color=df_s["roc_auc"] * 100,
                    colorscale=[[0,"#5eb8ff"],[0.5,"#b794f4"],[1,"#2dd4bf"]],
                    showscale=False,
                    line_width=0,
                ),
                text=[f"{v:.2f}%" for v in df_s["roc_auc"] * 100],
                textposition="outside",
                textfont=dict(size=11, color="#94a3c4"),
            ))
            fig_auc.update_layout(**_LAYOUT, height=360,
                xaxis=dict(range=[95, 100], title="AUC-ROC (%)", gridcolor="rgba(94,184,255,.07)"),
            )
            st.plotly_chart(fig_auc, width='stretch')

        with c2:
            st.markdown('<div class="nc-section-title">Radar: Multi-metric Overview</div>', unsafe_allow_html=True)
            metrics = ["accuracy", "f1", "roc_auc", "mcc"]
            m_labels = ["Accuracy", "F1", "AUC-ROC", "MCC"]
            fig_radar = go.Figure()
            colors_r = ["#5eb8ff","#b794f4","#2dd4bf","#f59e0b","#fc6b6b","#f472b6","#a78bfa","#34d399"]
            
            # Helper function to convert hex to rgba
            def hex_to_rgba(hex_color, alpha=0.15):
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({r}, {g}, {b}, {alpha})"
            
            for i, (_, row) in enumerate(df_models.iterrows()):
                vals = [row.get(m, 0) for m in metrics] + [row.get(metrics[0], 0)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=m_labels + [m_labels[0]],
                    fill="toself", name=row["Model"],
                    line=dict(color=colors_r[i % len(colors_r)], width=2),
                    fillcolor=hex_to_rgba(colors_r[i % len(colors_r)]),
                    opacity=0.9,
                ))
            fig_radar.update_layout(**_LAYOUT, height=360,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0.9, 1.0], gridcolor="rgba(94,184,255,.12)", tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="rgba(94,184,255,.12)"),
                ),
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_radar, width='stretch')

        # Table
        st.markdown('<div class="nc-section-title">Full Metrics Table</div>', unsafe_allow_html=True)
        display_cols = ["Model"] + [c for c in ["accuracy","f1","roc_auc","mcc","cv_auc_mean","cv_auc_std","time_s"] if c in df_models.columns]
        st.dataframe(
            df_models[display_cols].style.background_gradient(subset=[c for c in ["roc_auc","f1","accuracy"] if c in df_models.columns], cmap="Blues"),
            width='stretch', hide_index=True,
        )
    else:
        st.info("Run `python build_project_max.py` to generate model results.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab_features:
    importance = get_feature_importance()
    if importance:
        df_imp = pd.DataFrame(
            [(k.replace("_"," ").title(), v*100) for k, v in sorted(importance.items(), key=lambda x: -x[1])],
            columns=["Feature","Importance %"],
        )
        # Add category
        rev_cat = {f: cat for cat, fs in FEATURE_CATEGORIES.items() for f in fs}
        df_imp["Category"] = df_imp["Feature"].apply(lambda x: rev_cat.get(x.lower().replace(" ","_"), "Other"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="nc-section-title">Feature Importance (Top 20)</div>', unsafe_allow_html=True)
            df_top20 = df_imp.head(20)
            cat_colors = {"Neighborhood":"#5eb8ff","Structural":"#b794f4","Community":"#2dd4bf",
                          "Clustering":"#f59e0b","Centrality":"#f472b6","Other":"#94a3c4"}
            fig_imp = go.Figure(go.Bar(
                x=df_top20["Importance %"], y=df_top20["Feature"], orientation="h",
                marker_color=[cat_colors.get(c, "#94a3c4") for c in df_top20["Category"]],
                marker_line_width=0,
                text=[f"{v:.2f}%" for v in df_top20["Importance %"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3c4"),
            ))
            fig_imp.update_layout(**_LAYOUT, height=550,
                xaxis=dict(title="Importance (%)", gridcolor="rgba(94,184,255,.07)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_imp, width='stretch')

        with c2:
            st.markdown('<div class="nc-section-title">Category Distribution</div>', unsafe_allow_html=True)
            df_cat = df_imp.groupby("Category")["Importance %"].sum().reset_index()
            fig_pie = go.Figure(go.Pie(
                labels=df_cat["Category"], values=df_cat["Importance %"],
                hole=0.55,
                marker=dict(colors=[cat_colors.get(c,"#94a3c4") for c in df_cat["Category"]],
                            line=dict(color="#02040f", width=3)),
                textinfo="label+percent",
                textfont=dict(size=12, family="Syne"),
            ))
            fig_pie.update_layout(**_LAYOUT, height=320,
                annotations=[dict(text="Features", x=0.5, y=0.5, font=dict(size=16, color="#eef2ff"), showarrow=False)],
            )
            st.plotly_chart(fig_pie, width='stretch')

            st.markdown('<div class="nc-section-title" style="margin-top:12px;">Feature Registry</div>', unsafe_allow_html=True)
            for cat, feats in FEATURE_CATEGORIES.items():
                color = cat_colors.get(cat, "#94a3c4")
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <span style="color:{color};font-weight:700;font-size:.85rem;">{cat}</span>
                  <div style="margin-top:4px;">
                    {"".join(f'<span style="display:inline-block;background:{color}18;border:1px solid {color}40;color:{color};border-radius:6px;padding:2px 8px;font-size:.75rem;font-family:JetBrains Mono,monospace;margin:2px">{f}</span>' for f in feats)}
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Train the model first to see feature importance.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("""
    <div class="nc-card">
      <div class="nc-section-title">ML Pipeline Overview</div>
      <div style="color:#94a3c4;line-height:1.8;font-size:.92rem;">
        NeuroCollab implements a full Graph ML link-prediction pipeline on the DBLP academic graph.
      </div>
    </div>""", unsafe_allow_html=True)

    steps = [
        ("📥", "Data Ingestion",     "Load DBLP .mtx graph (317K nodes, 1M+ edges) + community labels"),
        ("🌐", "Graph Sampling",     "Extract 20,000-node subgraph; sample 6K positive + 6K negative pairs"),
        ("🔬", "Feature Engineering","Compute 25 graph features across 5 categories (Neighborhood, Structural, Community, Clustering, Centrality)"),
        ("⚖️",  "Balancing",         "Balanced positive/negative sampling — exact 50/50 split"),
        ("🤖", "Model Training",     "8 models: LR, RF, ExtraTrees, HistGB, MLP, XGBoost, LightGBM, Voting Ensemble"),
        ("📈", "Evaluation",         "8 metrics: Accuracy, F1, AUC-ROC, Precision, Recall, Avg Precision, MCC, Balanced Acc + 5-fold CV"),
        ("🧩", "Explainability",     "SHAP-like feature contributions computed per prediction"),
        ("🚀", "Deployment",         "Flask API · Streamlit · Hugging Face Gradio · Render cloud"),
    ]
    for emoji, title, desc in steps:
        st.markdown(f"""
        <div class="nc-card" style="padding:16px 24px;display:flex;gap:16px;align-items:flex-start">
          <div style="font-size:1.8rem;min-width:44px;text-align:center">{emoji}</div>
          <div>
            <div style="font-weight:700;color:#eef2ff;margin-bottom:4px">{title}</div>
            <div style="color:#94a3c4;font-size:.88rem;font-family:'JetBrains Mono',monospace">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nc-section-title" style="margin-top:24px;">Quick Start</div>', unsafe_allow_html=True)
    st.code("""# 1. Install
pip install -r requirements.txt

# 2. Train (full pipeline)
python build_project_max.py

# 3a. Streamlit UI
streamlit run streamlit_app.py

# 3b. Gradio / Hugging Face UI
python gradio_app.py

# 3c. Flask API
cd flask_app && python app.py

# 3d. Render (cloud)
# Push to GitHub → connect on render.com → deploy
""", language="bash")

    # Programmatic API demo
    st.markdown('<div class="nc-section-title">Use the Pipeline in Code</div>', unsafe_allow_html=True)
    st.code("""from pipeline import predict, batch_predict

# Single prediction
result = predict(
    cn=5, du=20, dv=15, same_community=1,
    clust_u=0.25, clust_v=0.18,
    tri_u=8, tri_v=6,
    cs_u=200, cs_v=150,
    pr_u=0.0003, pr_v=0.0002,
    core_u=4, core_v=3,
)
print(result.verdict, result.score)   # "Highly Compatible" 84.2

# Batch prediction
pairs = [{"cn":5,"du":20,"dv":15,"same_community":1,...}, ...]
results = batch_predict(pairs)        # list of {score, verdict, level}
""", language="python")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="nc-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4a5570;font-size:.8rem;font-family:'JetBrains Mono',monospace;padding-bottom:32px;">
  NeuroCollab v3.0 MAX · Graph ML · DBLP · Voting Ensemble · ~99.4% AUC-ROC
</div>""", unsafe_allow_html=True)
